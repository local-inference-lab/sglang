from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, List, Optional

import torch

if TYPE_CHECKING:
    pass


"""
Some comments on the common terms used in DeepSeekV4Backend:

topk_lengths:
    NOTE: TL;DR: topk_lengths == seq_lens
    The sparse decode kernel will attend to `k` tokens for each query.
    `topk_lengths` indicates how many tokens each query will attend to.
    This should be named as `seq_lens`, but we simply follow the naming convention.

page_table:
    The page table indicates which pages each request is assigned to.
    Each value in the page table is the page index in the TokenToKVPool.
    This page index is irrelevant to the actual `page_size`.

page_indices:
    The real indices used to index into the KV cache.
    This can be computed from the `page_table` and `page_size`.
    e.g. page_indices[i, j] = page_table[i, j // page_size] * page_size + (j % page_size)
    For sparse C4 top-512 attention, the indices will be selected from the C4 page indices.
    In implementation, we don't materialize the full C4 `page_indices`,
    but calculate them from `page_table` on-the-fly in the attention kernel.

positions:
    The position of the last token for each request.
    For compress token, the positions must be times of compress ratio.
    For example, for C4, raw_position=11 will trigger a compression,
    But the RoPE's position, during compression, must be 8 instead of 11.

Some other notes:
    c4_ / c128_: means "compressed by 4" / "compressed by 128".
    c4_page_size: page_size // 4
    c4_seq_lens: seq_lens // 4, but bounded by at least 1 for sparse attention.
    c4_sparse: means "compressed by 4" but only attend to top-512 tokens.
               all related length will be clipped to 512.
"""


def copy_metadata(
    *,
    src,
    dst,
    check_eq_fields: List[str],
    copy_fields: List[str],
    assign_fields: Optional[List[str]] = None,
):
    assign_fields = assign_fields or []

    for field_name in check_eq_fields:
        src_val = getattr(src, field_name)
        dst_val = getattr(dst, field_name)
        assert src_val == dst_val, f"{field_name=} {src_val=} {dst_val=}"

    for field_name in copy_fields:
        src_val = getattr(src, field_name)
        dst_val = getattr(dst, field_name)
        if src_val is None and dst_val is None:
            continue
        assert dst_val is not None, f"{field_name=} {src_val=} {dst_val=}"
        if hasattr(dst_val, "copy_"):
            dst_val.copy_(src_val)
        else:
            warnings.warn(
                f"{field_name=} {type(dst_val)=} does not have copy_, use setattr"
            )
            setattr(dst, field_name, src_val)

    for field_name in assign_fields:
        setattr(dst, field_name, getattr(src, field_name))

    provided_fields = check_eq_fields + copy_fields + assign_fields
    provided_fields_unique = set(provided_fields)
    assert len(provided_fields) == len(
        provided_fields_unique
    ), f"{provided_fields=} has dup"
    all_fields = {f.name for f in fields(src)}
    provided_fields = set(provided_fields)
    assert (
        provided_fields == all_fields
    ), f"{provided_fields - all_fields=}, {all_fields - provided_fields=}"


def _is_cuda_graph_capture_active(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


@dataclass
class PagedIndexerMetadata:
    page_size: int
    page_table: torch.Tensor
    c4_seq_lens: torch.Tensor
    expected_num_q_heads: Optional[int] = None
    shared_page_table: bool = False
    b12x_metadata: Any = field(init=False, repr=False)
    b12x_schedule_metadata: Optional[torch.Tensor] = field(
        init=False, repr=False, default=None
    )
    topk_metadata: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        assert self.page_size == 256, "the system hardcodes page_size=256"
        if self.page_table.dtype != torch.int32:
            self.page_table = self.page_table.to(torch.int32)
        if self.page_table.dim() != 2:
            raise ValueError(
                f"paged indexer page_table must be rank-2, got {self.page_table.shape=}"
            )
        if not self.page_table.is_contiguous():
            self.page_table = self.page_table.contiguous()

        c4_seq_lens = self.c4_seq_lens.to(torch.int32)
        if c4_seq_lens.dim() == 2:
            if c4_seq_lens.shape[-1] != 1:
                raise ValueError(
                    "paged indexer c4_seq_lens rank-2 input must have trailing "
                    f"dimension 1, got {tuple(c4_seq_lens.shape)}"
                )
            c4_seq_lens = c4_seq_lens.squeeze(-1)
        if c4_seq_lens.dim() != 1:
            raise ValueError(
                f"paged indexer c4_seq_lens must be rank-1, got {c4_seq_lens.shape=}"
            )
        self.c4_seq_lens = c4_seq_lens.contiguous()
        self._refresh_b12x_metadata()

        self.topk_metadata = torch.empty(
            (0,),
            dtype=torch.int32,
            device=self.c4_seq_lens.device,
        )

    def _refresh_b12x_metadata(
        self,
        *,
        build_schedule: Optional[bool] = None,
        validate_raw_lengths: Optional[bool] = None,
    ) -> None:
        from b12x.integration.compressed_indexer import (
            prepare_compressed_indexer_metadata,
        )

        if validate_raw_lengths is None:
            validate_raw_lengths = self.page_table.device.type != "cuda"
        self.b12x_metadata = prepare_compressed_indexer_metadata(
            real_page_table=self.page_table,
            cache_seqlens_int32=self.c4_seq_lens,
            page_size=self.c4_page_size,
            expected_num_q_heads=self.expected_num_q_heads,
            schedule_metadata=self.b12x_schedule_metadata,
            build_schedule=build_schedule,
            validate_raw_lengths=validate_raw_lengths,
            shared_page_table=self.shared_page_table,
        )
        self.b12x_schedule_metadata = self.b12x_metadata.schedule_metadata

    @property
    def c4_page_size(self) -> int:
        return self.page_size // 4

    @property
    def max_seq_len(self) -> int:
        return self.page_table.shape[1] * self.page_size

    @property
    def max_c4_seq_len(self) -> int:
        return self.page_table.shape[1] * self.c4_page_size

    def copy_(self, other: "PagedIndexerMetadata"):
        assert self.page_size == other.page_size
        assert self.expected_num_q_heads == other.expected_num_q_heads
        self.shared_page_table = other.shared_page_table
        self.page_table.copy_(other.page_table)
        self.c4_seq_lens.copy_(other.c4_seq_lens)
        self.topk_metadata.copy_(other.topk_metadata)

        other_schedule = other.b12x_schedule_metadata
        if other_schedule is None:
            self.b12x_schedule_metadata = None
        elif (
            self.b12x_schedule_metadata is None
            or self.b12x_schedule_metadata.shape != other_schedule.shape
        ):
            if _is_cuda_graph_capture_active(other_schedule.device):
                raise RuntimeError(
                    "b12x compressed-indexer schedule metadata was not allocated before "
                    "CUDA graph capture"
                )
            self.b12x_schedule_metadata = torch.empty_like(other_schedule)
            self.b12x_schedule_metadata.copy_(other_schedule)
        else:
            self.b12x_schedule_metadata.copy_(other_schedule)

        self._refresh_b12x_metadata(
            build_schedule=False,
            validate_raw_lengths=False,
        )


def maybe_copy_inplace(dst, *, src) -> None:
    assert type(src) == type(dst)
    if dst is not None:
        dst.copy_(src)
