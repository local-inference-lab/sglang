from __future__ import annotations

import enum
import functools
import logging
import os
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.b12x_backend import _b12x_get_config_attr
from sglang.srt.layers.attention.dsv4.compressor import (
    CompressorBackendMixin,
    FusedCompressMetadata,
    create_paged_compressor_data,
)
from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin
from sglang.srt.layers.attention.dsv4.metadata import (
    PagedIndexerMetadata,
    copy_metadata,
    maybe_copy_inplace,
)
from sglang.srt.layers.attention.dsv4.metadata_kernel import (
    init_compression_metadata as _init_compression_metadata_triton,
)
from sglang.srt.layers.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.layers.dp_attention import (
    get_attention_cp_rank,
    get_attention_cp_size,
    get_attention_tp_size,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import ceil_align

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

SWA_WINDOW = 128
C4_TOPK = 512
PAGE_INDEX_ALIGNED_SIZE = 64
B12X_COMPRESSED_MLA_HEAD_DIM = 512
B12X_C4_INDEXER_TILE_BLOCK_K = 512
B12X_C4_INDEXER_SUPERTILE_K_ENV = "B12X_COMPRESSED_INDEX_SUPERTILE_K"
B12X_C4_INDEXER_SUPERTILE_K_DEFAULT = 1048576
B12X_C4_INDEXER_UNSCHEDULED_MAX_PAGES = 1023
B12X_C4_INDEXER_TILE_LOGITS_BUDGET_BYTES_ENV = (
    "B12X_DSV4_C4_INDEXER_TILE_LOGITS_BUDGET_BYTES"
)
B12X_C4_INDEXER_TILE_LOGITS_BUDGET_BYTES_DEFAULT = 2 * 1024 * 1024 * 1024


T = TypeVar("T", bound=Optional[torch.Tensor])


@dataclass
class _B12XCompressedMLABundle:
    arena: object
    workspaces: Dict[Tuple[object, ...], object]
    mhc_workspace: object | None = None


_global_b12x_compressed_mla_bundles: Dict[
    Tuple[object, ...], _B12XCompressedMLABundle
] = {}


def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED_SIZE) -> T:
    if x is None:
        return None
    curr_size = x.shape[-1]
    target_size = ceil_align(curr_size, multiples_of)
    return F.pad(x, pad=(0, target_size - curr_size), mode="constant", value=-1)


def _same_device(actual: torch.device, expected: torch.device) -> bool:
    if actual.type != expected.type:
        return False
    return expected.index is None or actual.index == expected.index


def _create_dummy_paged_compress_data(compress_ratio: int):
    return None


@dataclass
class DSV4AttnMetadata:
    page_size: int
    page_table: torch.Tensor
    raw_out_loc: torch.Tensor
    cuda_int32_kwargs: dict

    seq_lens_casual: torch.Tensor
    positions_casual: torch.Tensor

    swa_page_indices: torch.Tensor
    swa_topk_lengths: torch.Tensor

    c4_sparse_topk: int
    c4_out_loc: Optional[torch.Tensor] = None
    c4_topk_lengths_raw: Optional[torch.Tensor] = None
    c4_topk_lengths_clamp1: Optional[torch.Tensor] = None
    c4_sparse_topk_lengths: torch.Tensor = field(init=False)
    c4_sparse_page_indices: torch.Tensor = field(init=False)

    c128_out_loc: Optional[torch.Tensor] = None
    c128_page_indices: Optional[torch.Tensor] = None
    c128_topk_lengths_clamp1: Optional[torch.Tensor] = None
    c128_index_capacity: Optional[int] = None

    @property
    def positions(self) -> torch.Tensor:
        return self.positions_casual

    def copy_(self, other: DSV4AttnMetadata) -> None:
        copy_metadata(
            src=other,
            dst=self,
            check_eq_fields=[
                "c4_sparse_topk",
                "page_size",
                "cuda_int32_kwargs",
                "c128_index_capacity",
            ],
            copy_fields=[
                "raw_out_loc",
                "seq_lens_casual",
                "positions_casual",
                "c4_out_loc",
                "c128_out_loc",
                "page_table",
                "swa_page_indices",
                "swa_topk_lengths",
                "c128_page_indices",
                "c128_topk_lengths_clamp1",
                "c4_topk_lengths_raw",
                "c4_topk_lengths_clamp1",
                "c4_sparse_topk_lengths",
                "c4_sparse_page_indices",
            ],
        )

    def init_compression_metadata(self):
        assert self.page_table.dim() == 2
        assert (
            self.raw_out_loc.shape == self.seq_lens_casual.shape
        ), f"{self.raw_out_loc.shape=}, {self.seq_lens_casual.shape=}"

        (
            self.c4_out_loc,
            _,
            self.c4_topk_lengths_raw,
            self.c4_topk_lengths_clamp1,
            self.c128_out_loc,
            _,
            self.c128_topk_lengths_clamp1,
            self.c128_page_indices,
        ) = _init_compression_metadata_triton(
            self.seq_lens_casual,
            self.positions_casual,
            self.raw_out_loc,
            self.page_table,
            self.page_size,
            compute_page_indices=True,
            max_c128_page_indices=self.c128_index_capacity,
        )

        self.c128_page_indices = _pad_last_dim(self.c128_page_indices)
        self.swa_page_indices = _pad_last_dim(self.swa_page_indices)

    _CP_REINDEX_FIELDS = [
        "seq_lens_casual",
        "positions_casual",
        "swa_page_indices",
        "swa_topk_lengths",
        "page_table",
        "c4_topk_lengths_raw",
        "c4_topk_lengths_clamp1",
        "c128_page_indices",
        "c128_topk_lengths_clamp1",
    ]
    _CP_GLOBAL_FIELDS = [
        "raw_out_loc",
        "c4_out_loc",
        "c128_out_loc",
    ]

    def apply_cp_reindex(self) -> None:
        cp_rank = get_attention_cp_rank()
        cp_size = get_attention_cp_size()
        idx = slice(cp_rank, None, cp_size)
        pre_global_len = self.seq_lens_casual.shape[0]
        assert pre_global_len % cp_size == 0, (
            f"apply_cp_reindex: global token count {pre_global_len} is not divisible by cp_size={cp_size}. "
            "CP round-robin requires padding to ensure divisibility."
        )
        expected_local_len = pre_global_len // cp_size
        for field_name in self._CP_REINDEX_FIELDS:
            val = getattr(self, field_name, None)
            assert isinstance(
                val, torch.Tensor
            ), f"CP reindex: {field_name} is {type(val)}, expected Tensor"
            setattr(self, field_name, val[idx].contiguous())

        for field_name in self._CP_REINDEX_FIELDS:
            val = getattr(self, field_name)
            assert val.shape[0] == expected_local_len, (
                f"apply_cp_reindex post-condition: {field_name}.shape[0]={val.shape[0]} "
                f"!= expected_local_len={expected_local_len} (cp_size={cp_size})"
            )
        for field_name in self._CP_GLOBAL_FIELDS:
            val = getattr(self, field_name, None)
            if val is None:
                continue
            assert val.shape[0] == pre_global_len, (
                f"apply_cp_reindex post-condition: global field {field_name}.shape[0]={val.shape[0]} "
                f"!= pre_global_len={pre_global_len} (must remain global for compressor write path)"
            )

    def init_sparse_mla_related(self):
        # c4_sparse_topk is set from model_config.index_topk per-model
        # (small model: 512, large model: 1024).
        assert self.c4_sparse_topk in (512, 1024), (
            f"unexpected c4_sparse_topk={self.c4_sparse_topk}; "
            "supported: 512 (small) or 1024 (large)"
        )
        assert self.c4_topk_lengths_clamp1 is not None
        self.c4_sparse_topk_lengths = torch.clamp(
            self.c4_topk_lengths_clamp1, max=self.c4_sparse_topk
        )
        self.c4_sparse_page_indices = torch.full(
            (self.c4_topk_lengths_clamp1.size(0), self.c4_sparse_topk),
            -1,
            dtype=torch.int32,
            device=self.c4_topk_lengths_clamp1.device,
        )
        self.c4_sparse_page_indices = _pad_last_dim(self.c4_sparse_page_indices)


@dataclass
class DSV4Metadata:
    core_attn_metadata: DSV4AttnMetadata
    indexer_metadata: Optional[PagedIndexerMetadata]

    c4_compress_metadata: Optional[FusedCompressMetadata] = None
    c128_compress_metadata: Optional[FusedCompressMetadata] = None

    @property
    def core_metadata(self) -> DSV4AttnMetadata:
        return self.core_attn_metadata

    def copy_(self, other: DSV4Metadata):
        self.core_attn_metadata.copy_(other.core_attn_metadata)
        maybe_copy_inplace(self.indexer_metadata, src=other.indexer_metadata)
        maybe_copy_inplace(self.c4_compress_metadata, src=other.c4_compress_metadata)
        maybe_copy_inplace(
            self.c128_compress_metadata, src=other.c128_compress_metadata
        )


@dataclass
class DSV4RawVerifyMetadata:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor

    extend_seq_lens: Optional[torch.Tensor] = None

    def copy_(self, other: DSV4RawVerifyMetadata):
        self.req_pool_indices.copy_(other.req_pool_indices)
        self.seq_lens.copy_(other.seq_lens)
        self.out_cache_loc.copy_(other.out_cache_loc)

        self.extend_seq_lens = other.extend_seq_lens


@dataclass
class DSV4RawDecodeMetadata:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor

    def copy_(self, other: DSV4RawDecodeMetadata):
        self.req_pool_indices.copy_(other.req_pool_indices)
        self.seq_lens.copy_(other.seq_lens)
        self.out_cache_loc.copy_(other.out_cache_loc)


class _GraphBucket(enum.Enum):
    DECODE_OR_IDLE = "decode_or_idle"
    TARGET_VERIFY = "target_verify"
    DRAFT_EXTEND = "draft_extend"

    @classmethod
    def of(cls, forward_mode: ForwardMode) -> _GraphBucket:
        if forward_mode.is_decode_or_idle():
            return cls.DECODE_OR_IDLE
        if forward_mode.is_target_verify():
            return cls.TARGET_VERIFY
        if forward_mode.is_draft_extend(include_v2=True):
            return cls.DRAFT_EXTEND
        raise NotImplementedError(f"unsupported {forward_mode=}")


class DeepseekV4AttnBackend(
    AttentionBackend, C4IndexerBackendMixin, CompressorBackendMixin
):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()
        self.device = torch.device(model_runner.device)
        self.server_args = model_runner.server_args
        self.max_running_requests = int(getattr(model_runner, "max_running_requests", 1))
        self.q_dtype = model_runner.dtype
        head_dim = model_runner.model_config.head_dim
        assert (
            head_dim == 512
        ), "DSV4 MQA head_dim = qk_nope_head_dim(448) + qk_rope_head_dim(64) = 512"
        self.softmax_scale: float = head_dim**-0.5
        self.head_dim_v: int = model_runner.model_config.v_head_dim
        self.cuda_int32_kwargs = {"device": self.device, "dtype": torch.int32}
        self.swa_page_size = 128
        assert model_runner.page_size is not None
        assert model_runner.req_to_token_pool is not None
        self.page_size = model_runner.page_size
        assert self.page_size == 256, "the system hardcodes page_size=256"

        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool: DeepSeekV4TokenToKVPool = model_runner.token_to_kv_pool
        self.MAX_SEQ_LEN_FOR_CAPTURE = self.req_to_token.shape[1]

        assert isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool)
        self.hf_text_config = model_runner.model_config.hf_text_config
        self.c4_topk = getattr(self.hf_text_config, "index_topk", C4_TOPK)
        architectures = tuple(
            str(architecture)
            for architecture in getattr(model_runner.model_config.hf_config, "architectures", ())
        )
        self._b12x_is_nextn_model = any("NextN" in architecture for architecture in architectures)
        self.attn_tp_size = get_attention_tp_size()
        model_config_total_q_heads = getattr(
            model_runner.model_config, "num_attention_heads", None
        )
        total_q_heads = int(
            getattr(self.hf_text_config, "num_attention_heads", model_config_total_q_heads)
        )
        index_total_q_heads = int(
            getattr(self.hf_text_config, "index_n_heads", total_q_heads)
        )
        if total_q_heads % self.attn_tp_size != 0:
            raise ValueError(
                f"num_attention_heads={total_q_heads} must divide by TP={self.attn_tp_size}"
            )
        if index_total_q_heads <= 0:
            raise ValueError(
                f"index_n_heads must be positive, got {index_total_q_heads}"
            )
        self.num_q_heads = total_q_heads // self.attn_tp_size
        self.index_num_q_heads = index_total_q_heads
        self._b12x_compressed_workspaces = {}
        self._b12x_indexer_workspaces = {}
        self._b12x_attention_bundle: Optional[_B12XCompressedMLABundle] = None
        self._use_prep_in_cuda_graph = False

        self.topk = self.server_args.speculative_eagle_topk or 0
        assert self.topk in [0, 1], "MTP Topk > 1 not supported for DeepSeek V4"
        self.mtp_enabled = self.topk > 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens: int = (
            self.server_args.speculative_num_draft_tokens
        )
        self.speculative_step_id = speculative_step_id
        self.forward_metadata: Union[
            DSV4Metadata,
            DSV4RawVerifyMetadata,
            DSV4RawDecodeMetadata,
        ] = None
        self._replay_forward_batch: Optional[ForwardBatch] = None  # FIXME: out-of-band
        self._init_b12x_attention_bundle(model_runner)

    def _b12x_graph_batch_counts(self) -> Tuple[int, ...]:
        candidates = [self.max_running_requests]
        cuda_graph_max_bs = getattr(self.server_args, "cuda_graph_max_bs", None)
        if cuda_graph_max_bs is not None:
            candidates.append(int(cuda_graph_max_bs))
        cuda_graph_bs = getattr(self.server_args, "cuda_graph_bs", None)
        if cuda_graph_bs:
            candidates.extend(int(bs) for bs in cuda_graph_bs)
        return tuple(sorted({int(value) for value in candidates if int(value) > 0}))

    def _b12x_graph_batch_capacity(self) -> int:
        return max(1, *self._b12x_graph_batch_counts())

    def _b12x_graph_q_rows_counts(self) -> Tuple[int, ...]:
        draft_tokens = max(1, int(self.speculative_num_draft_tokens or 1))
        counts = self._b12x_graph_batch_counts()
        if not counts:
            counts = (1,)
        return tuple(max(1, int(count) * draft_tokens) for count in counts)

    def _b12x_graph_q_rows_capacity(self) -> int:
        return max(1, *self._b12x_graph_q_rows_counts())

    def _b12x_compression_ratios(self) -> Tuple[int, ...]:
        ratios = getattr(self.token_to_kv_pool, "compression_ratios", None)
        if ratios is None:
            ratios = getattr(self.hf_text_config, "compress_ratios", ())
        return tuple(int(ratio) for ratio in ratios)

    def _b12x_uses_c4_attention(self) -> bool:
        return any(ratio == 4 for ratio in self._b12x_compression_ratios())

    def _b12x_first_c4_layer_id(self) -> int:
        for layer_id, ratio in enumerate(self._b12x_compression_ratios()):
            if ratio == 4:
                return layer_id
        raise RuntimeError("b12x C4 indexer requested but no C4 layer is present")

    def _b12x_uses_c128_attention(self) -> bool:
        return any(ratio == 128 for ratio in self._b12x_compression_ratios())

    def _b12x_uses_compressed_attention(self) -> bool:
        return self._b12x_uses_c4_attention() or self._b12x_uses_c128_attention()

    def _b12x_prefill_budget_selected_widths(self) -> Tuple[int, ...]:
        widths = set(self._b12x_compressed_selected_widths())
        if self._b12x_is_nextn_model and not self._b12x_uses_compressed_attention():
            swa_width = ceil_align(SWA_WINDOW, PAGE_INDEX_ALIGNED_SIZE)
            c4_width = ceil_align(self.c4_topk, PAGE_INDEX_ALIGNED_SIZE)
            widths.add(swa_width + c4_width)
        return tuple(sorted(widths))

    def _b12x_prefill_budget_split_chunk_capacity(self) -> int:
        from b12x.integration.mla import compressed_mla_split_chunks_for_contract

        max_chunks = 1
        rows = self._b12x_eager_extend_total_q_capacity()
        for selected_width in self._b12x_prefill_budget_selected_widths():
            max_chunks = max(
                max_chunks,
                compressed_mla_split_chunks_for_contract(
                    rows=rows,
                    width=selected_width,
                ),
            )
        return max_chunks

    def _b12x_eager_extend_total_q_capacity(self) -> int:
        chunked_prefill_size = int(
            getattr(self.server_args, "chunked_prefill_size", -1) or -1
        )
        if chunked_prefill_size > 0:
            return chunked_prefill_size
        raise RuntimeError(
            "b12x DeepSeek V4 prefill requires --chunked-prefill-size > 0. "
            "Without chunking, eager prefill would need a max-prefill-token "
            "compressed MLA workspace and cannot reuse the fixed chunk workspace."
        )

    def _b12x_compressed_prefill_q_capacity(self) -> int:
        chunk_capacity = self._b12x_eager_extend_total_q_capacity()
        override = os.environ.get("B12X_COMPRESSED_MLA_PREFILL_Q_CAP")
        if override is not None:
            q_capacity = int(override)
            if q_capacity <= 0:
                raise RuntimeError(
                    "B12X_COMPRESSED_MLA_PREFILL_Q_CAP must be positive when set"
                )
            return max(1, min(chunk_capacity, q_capacity))

        budget_gib = float(
            os.environ.get("B12X_COMPRESSED_MLA_PREFILL_WORKSPACE_GIB", "2.0")
        )
        if budget_gib <= 0:
            raise RuntimeError(
                "B12X_COMPRESSED_MLA_PREFILL_WORKSPACE_GIB must be positive"
            )
        selected_width = max(self._b12x_prefill_budget_selected_widths())
        max_chunks_per_row = self._b12x_prefill_budget_split_chunk_capacity()
        row_nbytes = (
            3 * 4
            + self.num_q_heads * max_chunks_per_row * self.head_dim_v * 2
            + self.num_q_heads * max_chunks_per_row * 4
        )
        q_capacity = int((budget_gib * (1 << 30)) // max(row_nbytes, 1))
        q_capacity = max(1, min(chunk_capacity, q_capacity))
        if q_capacity >= 128:
            q_capacity = (q_capacity // 128) * 128
        return max(1, q_capacity)

    @staticmethod
    def _b12x_uses_chunked_prefill_workspace(forward_mode: ForwardMode) -> bool:
        if forward_mode.is_target_verify() or forward_mode.is_draft_extend(
            include_v2=True
        ):
            return False
        return forward_mode.is_prefill(include_draft_extend_v2=True)

    def _b12x_prefill_q_capacity(
        self,
        *,
        forward_batch: ForwardBatch,
        q_rows: int,
    ) -> Optional[int]:
        if not self._b12x_uses_chunked_prefill_workspace(
            forward_batch.forward_mode
        ):
            return None
        q_capacity = self._b12x_eager_extend_total_q_capacity()
        if int(q_rows) > q_capacity:
            raise RuntimeError(
                "b12x DeepSeek V4 prefill received a chunk larger than the fixed "
                f"workspace capacity: q_rows={int(q_rows)}, "
                f"chunked_prefill_size={q_capacity}. Check that SGLang chunked "
                "prefill is enabled and that the chunk size matches the b12x "
                "workspace capacity."
            )
        return q_capacity

    def _b12x_compressed_selected_widths(self) -> Tuple[int, ...]:
        swa_width = ceil_align(SWA_WINDOW, PAGE_INDEX_ALIGNED_SIZE)
        widths = {swa_width}
        if self._b12x_uses_c4_attention():
            widths.add(swa_width + ceil_align(self.c4_topk, PAGE_INDEX_ALIGNED_SIZE))
        if self._b12x_uses_c128_attention():
            widths.add(swa_width + self._b12x_c128_index_width_capacity())
        return tuple(sorted(widths))

    def _b12x_compressed_selected_width_capacity(self) -> int:
        return max(self._b12x_compressed_selected_widths())

    def _b12x_full_token_capacity(self) -> int:
        candidates = [max(1, int(self.MAX_SEQ_LEN_FOR_CAPTURE))]
        c128_size = int(getattr(self.token_to_kv_pool, "c128_size", 0) or 0)
        if c128_size > 0:
            candidates.append(c128_size * 128)
        c4_logical_size = int(
            getattr(self.token_to_kv_pool, "c4_logical_size", 0) or 0
        )
        if c4_logical_size > 0:
            candidates.append(c4_logical_size * 4)
        return max(1, min(candidates))

    def _b12x_c128_index_width_capacity(self) -> int:
        if not self._b12x_uses_c128_attention():
            return 0
        full_token_capacity = self._b12x_full_token_capacity()
        c128_width = (full_token_capacity + 127) // 128
        c128_pool_size = int(getattr(self.token_to_kv_pool, "c128_size", 0) or 0)
        if c128_pool_size > 0:
            c128_width = min(c128_width, c128_pool_size)
        return ceil_align(c128_width, PAGE_INDEX_ALIGNED_SIZE)

    def _b12x_split_chunk_capacity(self) -> int:
        max_chunks = 1
        for selected_width in self._b12x_compressed_selected_widths():
            max_chunks = max(
                max_chunks,
                self._b12x_compressed_mla_split_chunks(
                    q_rows=self._b12x_graph_q_rows_capacity(),
                    selected_width=selected_width,
                ),
                self._b12x_compressed_mla_split_chunks(
                    q_rows=self._b12x_compressed_prefill_q_capacity(),
                    selected_width=selected_width,
                ),
            )
        return max_chunks

    def _b12x_compressed_mla_split_chunks(
        self,
        *,
        q_rows: int,
        selected_width: int,
    ) -> int:
        from b12x.integration.mla import compressed_mla_split_chunks_for_contract

        return compressed_mla_split_chunks_for_contract(
            rows=max(int(q_rows), 1),
            width=max(int(selected_width), 1),
        )

    def _b12x_compressed_mla_max_q_chunks_capacity(
        self,
        *,
        graph_q_rows: int,
        compressed_prefill_q: int,
        selected_widths: Tuple[int, ...],
    ) -> int:
        max_q_chunks = 1
        for selected_width in selected_widths:
            for q_rows in (graph_q_rows, compressed_prefill_q):
                max_q_chunks = max(
                    max_q_chunks,
                    max(int(q_rows), 1)
                    * self._b12x_compressed_mla_split_chunks(
                        q_rows=q_rows,
                        selected_width=selected_width,
                    ),
                )
        return max_q_chunks

    def _b12x_indexer_page_table_width_capacity(self) -> int:
        if not self._b12x_uses_c4_attention():
            return 1
        capture_width = (
            int(self.MAX_SEQ_LEN_FOR_CAPTURE) + int(self.page_size) - 1
        ) // int(self.page_size)
        return ceil_align(
            max(
                capture_width,
                (self._b12x_full_token_capacity() + self.page_size - 1)
                // self.page_size,
            ),
            PAGE_INDEX_ALIGNED_SIZE,
        )

    def _b12x_c4_indexer_supertile_tokens_capacity(
        self,
        *,
        page_table_width: Optional[int] = None,
        q_rows: Optional[int] = None,
    ) -> int:
        if not self._b12x_uses_c4_attention():
            return 0
        if page_table_width is None:
            page_table_width = self._b12x_indexer_page_table_width_capacity()
        c4_page_size = self.page_size // 4
        max_tokens = max(1, int(page_table_width) * c4_page_size)

        requested = int(
            os.environ.get(
                B12X_C4_INDEXER_SUPERTILE_K_ENV,
                str(B12X_C4_INDEXER_SUPERTILE_K_DEFAULT),
            )
        )
        if requested <= 0:
            raise RuntimeError(
                f"{B12X_C4_INDEXER_SUPERTILE_K_ENV} must be positive when set"
            )
        requested = ceil_align(requested, B12X_C4_INDEXER_TILE_BLOCK_K)

        budget_bytes = int(
            os.environ.get(
                B12X_C4_INDEXER_TILE_LOGITS_BUDGET_BYTES_ENV,
                str(B12X_C4_INDEXER_TILE_LOGITS_BUDGET_BYTES_DEFAULT),
            )
        )
        if budget_bytes <= 0:
            raise RuntimeError(
                f"{B12X_C4_INDEXER_TILE_LOGITS_BUDGET_BYTES_ENV} must be positive when set"
            )
        if q_rows is None:
            q_rows = max(
                self._b12x_graph_q_rows_capacity(),
                self._b12x_eager_extend_total_q_capacity(),
            )
        q_rows_aligned = ceil_align(max(int(q_rows), 1), 32)
        budget_tokens = budget_bytes // max(q_rows_aligned * 4, 1)
        budget_tokens = (
            budget_tokens // B12X_C4_INDEXER_TILE_BLOCK_K
        ) * B12X_C4_INDEXER_TILE_BLOCK_K
        if budget_tokens < B12X_C4_INDEXER_TILE_BLOCK_K:
            raise RuntimeError(
                "b12x DeepSeek V4 C4 indexer tile-logits budget is too small: "
                f"budget_bytes={budget_bytes}, q_rows_aligned={q_rows_aligned}"
            )

        # The current C4 windowed scorer intentionally uses the unscheduled
        # paged path so it can consume the live full page table directly under
        # graph capture. Keep the fixed supertile below b12x's paged-schedule
        # threshold for every captured q_rows contract.
        unscheduled_tokens = B12X_C4_INDEXER_UNSCHEDULED_MAX_PAGES * c4_page_size
        unscheduled_tokens = (
            unscheduled_tokens // B12X_C4_INDEXER_TILE_BLOCK_K
        ) * B12X_C4_INDEXER_TILE_BLOCK_K
        supertile_tokens = min(
            requested,
            max_tokens,
            budget_tokens,
            unscheduled_tokens,
        )
        supertile_tokens = (
            supertile_tokens // B12X_C4_INDEXER_TILE_BLOCK_K
        ) * B12X_C4_INDEXER_TILE_BLOCK_K
        return max(B12X_C4_INDEXER_TILE_BLOCK_K, supertile_tokens)

    def _b12x_c4_indexer_supertile_pages_capacity(
        self,
        *,
        page_table_width: Optional[int] = None,
        q_rows: Optional[int] = None,
    ) -> int:
        if not self._b12x_uses_c4_attention():
            return 1
        c4_page_size = self.page_size // 4
        return max(
            1,
            self._b12x_c4_indexer_supertile_tokens_capacity(
                page_table_width=page_table_width,
                q_rows=q_rows,
            )
            // c4_page_size,
        )

    def _build_b12x_attention_arena_caps(self):
        from b12x.attention.workspace import B12XAttentionArenaCaps

        graph_q_rows = self._b12x_graph_q_rows_capacity()
        prefill_chunk_q = self._b12x_eager_extend_total_q_capacity()
        compressed_prefill_q = self._b12x_compressed_prefill_q_capacity()
        selected_widths = self._b12x_compressed_selected_widths()
        selected_width = self._b12x_compressed_selected_width_capacity()
        indexer_topk = (
            ceil_align(self.c4_topk, PAGE_INDEX_ALIGNED_SIZE)
            if self._b12x_uses_c4_attention()
            else 1
        )
        page_table_width = self._b12x_indexer_page_table_width_capacity()
        c4_supertile_tokens = self._b12x_c4_indexer_supertile_tokens_capacity(
            page_table_width=page_table_width,
            q_rows=max(graph_q_rows, prefill_chunk_q),
        )
        c4_dense_decode_tokens = page_table_width * 64 if self._b12x_uses_c4_attention() else 0
        max_chunks_per_row = self._b12x_split_chunk_capacity()
        mla_max_q_chunks = self._b12x_compressed_mla_max_q_chunks_capacity(
            graph_q_rows=graph_q_rows,
            compressed_prefill_q=compressed_prefill_q,
            selected_widths=selected_widths,
        )
        logger.info(
            "b12x DeepSeek V4 attention arena caps: "
            "full_token_capacity=%s selected_widths=%s indexer_topk=%s "
            "indexer_page_table_width=%s max_chunks_per_row=%s mla_q_chunks=%s "
            "prefill_chunk_q=%s compressed_prefill_q=%s "
            "c4_indexer_supertile_tokens=%s c4_dense_decode_q=%s "
            "c4_dense_decode_tokens=%s mhc_tokens=%s",
            self._b12x_full_token_capacity(),
            list(selected_widths),
            indexer_topk,
            page_table_width,
            max_chunks_per_row,
            mla_max_q_chunks,
            prefill_chunk_q,
            compressed_prefill_q,
            c4_supertile_tokens,
            graph_q_rows,
            c4_dense_decode_tokens,
            max(graph_q_rows, prefill_chunk_q),
        )
        return B12XAttentionArenaCaps(
            device=self.device,
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            num_q_heads=self.num_q_heads,
            indexer_num_q_heads=self.index_num_q_heads,
            head_dim=B12X_COMPRESSED_MLA_HEAD_DIM,
            max_v_head_dim=self.head_dim_v,
            topk=selected_width,
            indexer_topk=indexer_topk,
            max_page_table_width=page_table_width,
            extend_max_total_q=compressed_prefill_q,
            extend_max_batch=compressed_prefill_q,
            extend_max_kv_rows=0,
            indexer_max_k_rows=c4_supertile_tokens,
            paged_max_q_rows=max(graph_q_rows, prefill_chunk_q),
            paged_max_batch=max(graph_q_rows, prefill_chunk_q),
            mla_max_total_q=max(graph_q_rows, compressed_prefill_q),
            mla_max_q_chunks=mla_max_q_chunks,
            page_size=64,
            max_chunks_per_row=max_chunks_per_row,
            reserve_extend_indexer_logits=False,
            reserve_paged_indexer_logits=self._b12x_uses_c4_attention(),
            reserve_mhc=True,
            mhc_max_tokens=max(graph_q_rows, prefill_chunk_q),
            mhc_hidden_size=int(getattr(self.hf_text_config, "hidden_size")),
            paged_indexer_logits_q_rows=graph_q_rows,
            paged_indexer_logits_k_rows=c4_dense_decode_tokens,
            paged_indexer_tile_logits_k_rows=c4_supertile_tokens,
        )

    def _build_b12x_moe_arena_caps(self, model_runner: ModelRunner):
        from b12x.integration import B12XMoEArenaCaps
        from sglang.srt.distributed import get_tensor_model_parallel_world_size
        from sglang.srt.layers.moe import get_moe_runner_backend

        if not get_moe_runner_backend().is_b12x():
            return None

        cfg = model_runner.model_config.hf_config
        weight_E = _b12x_get_config_attr(
            cfg, ("n_routed_experts", "num_experts", "num_local_experts")
        )
        hidden_size = _b12x_get_config_attr(cfg, ("hidden_size",))
        intermediate_size = _b12x_get_config_attr(
            cfg, ("moe_intermediate_size", "intermediate_size")
        )
        num_topk = _b12x_get_config_attr(
            cfg,
            (
                "num_experts_per_tok",
                "top_k",
                "num_experts_per_token",
                "router_topk",
            ),
        )
        missing = [
            name
            for name, value in (
                ("n_routed_experts/num_experts/num_local_experts", weight_E),
                ("hidden_size", hidden_size),
                ("moe_intermediate_size/intermediate_size", intermediate_size),
                (
                    "num_experts_per_tok/top_k/num_experts_per_token/router_topk",
                    num_topk,
                ),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "b12x joint arena cannot size MoE workspace; missing config fields: "
                + ", ".join(missing)
            )

        tp_size = max(1, int(get_tensor_model_parallel_world_size()))
        intermediate_size = int(intermediate_size)
        if intermediate_size % tp_size != 0:
            raise ValueError(
                "b12x joint arena expected MoE intermediate_size to be divisible "
                f"by tensor parallel size, got intermediate_size={intermediate_size}, tp_size={tp_size}"
            )
        graph_q_rows_counts = self._b12x_graph_q_rows_counts()
        graph_batch_counts = self._b12x_graph_batch_counts()
        graph_q_rows = max(1, *graph_q_rows_counts)
        extend_total_q = self._b12x_eager_extend_total_q_capacity()
        max_tokens = max(extend_total_q, graph_q_rows)
        small_graph_token_counts = range(1, graph_q_rows + 1)
        core_token_counts = {
            int(extend_total_q),
            *(int(count) for count in graph_q_rows_counts),
            # EAGLE draft graph capture can run the NextN MoE with raw graph
            # batch counts rather than batch*draft_tokens. Keep those exact
            # small launches preplanned so the frozen b12x MoE pool does not
            # need to compile or allocate inside capture.
            *(int(count) for count in graph_batch_counts),
            # Runtime speculative verify/draft paths can also produce accepted
            # token counts between the captured graph sizes. These are still a
            # fixed graph-contract envelope, not live sequence-derived shapes.
            *(int(count) for count in small_graph_token_counts),
        }
        return B12XMoEArenaCaps(
            device=self.device,
            dtype=self.q_dtype,
            quant_mode="w4a16",
            weight_E=int(weight_E),
            k=int(hidden_size),
            n=intermediate_size // tp_size,
            num_topk=int(num_topk),
            max_tokens=max_tokens,
            core_token_counts=tuple(sorted(core_token_counts)),
            route_num_experts=int(weight_E),
            route_logits_dtype=self.q_dtype,
            activation=str(getattr(cfg, "hidden_act", "silu")).lower(),
            apply_router_weight_on_input=bool(
                getattr(cfg, "moe_apply_router_weight_on_input", False)
            ),
            swiglu_limit=getattr(cfg, "swiglu_limit", None),
        )

    def _build_b12x_attention_bundle_key(self, caps) -> Tuple[object, ...]:
        return (
            caps.device,
            caps.dtype,
            caps.kv_dtype,
            int(caps.num_q_heads),
            int(caps.indexer_num_q_heads),
            int(caps.head_dim),
            int(caps.max_v_head_dim),
            int(caps.topk),
            int(getattr(caps, "indexer_topk", caps.topk)),
            int(caps.max_page_table_width),
            int(caps.extend_max_total_q),
            int(caps.extend_max_batch),
            int(caps.extend_max_kv_rows),
            int(getattr(caps, "indexer_max_k_rows", 0) or 0),
            int(caps.paged_max_q_rows),
            int(caps.paged_max_batch),
            int(getattr(caps, "mla_max_total_q", 0) or 0),
            int(getattr(caps, "mla_max_q_chunks", 0) or 0),
            int(caps.page_size),
            int(caps.padded_heads),
            int(caps.max_chunks_per_row),
            bool(caps.reserve_extend_indexer_logits),
            bool(caps.reserve_paged_indexer_logits),
            bool(getattr(caps, "reserve_mhc", False)),
            int(getattr(caps, "mhc_max_tokens", 0) or 0),
            int(getattr(caps, "mhc_hidden_size", 0) or 0),
            int(getattr(caps, "mhc_split_k", 0) or 0),
            int(caps.extend_indexer_tile_logits_k_rows),
            int(getattr(caps, "paged_indexer_logits_q_rows", 0) or 0),
            int(caps.paged_indexer_logits_k_rows),
            int(caps.paged_indexer_tile_logits_k_rows),
        )

    def _init_b12x_attention_bundle(self, model_runner: ModelRunner) -> None:
        from b12x.integration import (
            B12XJointArenaSpec,
            ensure_b12x_execution_lane_arena,
        )

        if self._b12x_attention_bundle is not None:
            return

        caps = self._build_b12x_attention_arena_caps()
        moe_caps = self._build_b12x_moe_arena_caps(model_runner)
        lane = ensure_b12x_execution_lane_arena(
            B12XJointArenaSpec(
                device=self.device,
                attention_caps=caps,
                moe_caps=moe_caps,
            )
        )
        if lane.arena is None or lane.arena.attention_arena is None:
            raise RuntimeError("b12x execution lane was allocated without attention arena")
        arena = lane.arena.attention_arena
        bundle_key = self._build_b12x_attention_bundle_key(caps)
        bundle = _global_b12x_compressed_mla_bundles.get(bundle_key)
        if bundle is None:
            bundle = _B12XCompressedMLABundle(arena=arena, workspaces={})
            _global_b12x_compressed_mla_bundles[bundle_key] = bundle
        elif bundle.arena is not arena:
            raise RuntimeError(
                "existing b12x compressed MLA bundle is not owned by the active execution lane"
            )
        self._b12x_attention_bundle = bundle
        setattr(model_runner, "_dsv4_b12x_attention_bundle", bundle)
        self._prime_b12x_default_fixed_workspaces()

    def get_b12x_mhc_workspace(self):
        if self._b12x_attention_bundle is None:
            raise RuntimeError("b12x attention bundle is not initialized")
        workspace = self._b12x_attention_bundle.mhc_workspace
        if workspace is None:
            workspace = self._b12x_attention_bundle.arena.make_mhc_workspace()
            self._b12x_attention_bundle.mhc_workspace = workspace
        return workspace

    def _move_to_device(self, x: List[int]) -> torch.Tensor:
        pin_tensor = torch.tensor(x, dtype=torch.int32, pin_memory=True)
        return pin_tensor.to(self.device, non_blocking=True)

    def init_forward_metadata_indexer(
        self,
        core_attn_metadata: DSV4AttnMetadata,
        *,
        shared_page_table: bool = False,
    ):
        return PagedIndexerMetadata(
            page_size=self.page_size,
            page_table=core_attn_metadata.page_table,
            c4_seq_lens=core_attn_metadata.c4_topk_lengths_raw,
            expected_num_q_heads=self.index_num_q_heads,
            shared_page_table=shared_page_table,
        )

    def _use_b12x_fixed_workspace(self, forward_batch: ForwardBatch) -> bool:
        return bool(forward_batch.forward_mode.is_cuda_graph() or get_is_capture_mode())

    def get_b12x_compressed_mla_workspace(
        self,
        *,
        forward_batch: ForwardBatch,
        q_rows: int,
        selected_width: int,
    ):
        fixed = self._use_b12x_fixed_workspace(forward_batch)
        uses_chunked_prefill_workspace = self._b12x_uses_chunked_prefill_workspace(
            forward_batch.forward_mode
        )
        max_fixed_q_rows = None
        if uses_chunked_prefill_workspace:
            self._b12x_prefill_q_capacity(
                forward_batch=forward_batch,
                q_rows=q_rows,
            )
            compressed_q_capacity = self._b12x_compressed_prefill_q_capacity()
            if int(q_rows) > compressed_q_capacity:
                raise RuntimeError(
                    "b12x DeepSeek V4 compressed MLA prefill must be sliced before "
                    f"workspace lookup: q_rows={int(q_rows)}, "
                    f"compressed_q_capacity={compressed_q_capacity}"
                )
            fixed = True
            q_rows = compressed_q_capacity
        else:
            max_fixed_q_rows = self._b12x_graph_q_rows_capacity()
        return self._get_b12x_compressed_mla_workspace(
            q_rows=q_rows,
            selected_width=selected_width,
            fixed=fixed,
            max_fixed_q_rows=max_fixed_q_rows,
        )

    def _get_b12x_compressed_mla_workspace(
        self,
        *,
        q_rows: int,
        selected_width: int,
        fixed: bool,
        max_fixed_q_rows: Optional[int] = None,
    ):
        from b12x.attention.workspace import (
            B12XAttentionWorkspace,
            B12XAttentionWorkspaceContract,
        )

        q_rows = max(int(q_rows), 1)
        selected_width = max(int(selected_width), 1)
        split_chunks = self._b12x_compressed_mla_split_chunks(
            q_rows=q_rows,
            selected_width=selected_width,
        )
        cache = self._b12x_compressed_workspaces
        key = (
            fixed,
            q_rows if fixed else "dynamic",
            selected_width,
            split_chunks,
            self.num_q_heads,
            self.index_num_q_heads,
        )
        workspace = cache.get(key)
        if workspace is None and fixed:
            workspace = self._find_b12x_compressed_mla_workspace(
                q_rows=q_rows,
                selected_width=selected_width,
                split_chunks=split_chunks,
                max_q_rows=max_fixed_q_rows,
            )
        if workspace is not None and not fixed:
            if (
                int(getattr(workspace, "max_total_q", 0)) < q_rows
                or int(getattr(workspace, "max_batch", 0)) < q_rows
                or int(getattr(workspace, "max_chunks_per_row", 0))
                < split_chunks
            ):
                workspace = None
        if workspace is None:
            if fixed and self._b12x_attention_bundle is not None:
                contract = B12XAttentionWorkspaceContract(
                    mode="decode",
                    max_total_q=q_rows,
                    max_batch=q_rows,
                    max_paged_q_rows=q_rows,
                    max_kv_rows=0,
                    v_head_dim=self.head_dim_v,
                    indexer_num_q_heads=self.index_num_q_heads,
                    max_page_table_width=1,
                    topk=selected_width,
                    max_chunks_per_row=split_chunks,
                )
                workspace = self._b12x_attention_bundle.arena.make_workspace(
                    contract,
                    use_cuda_graph=True,
                )
            else:
                if fixed:
                    raise RuntimeError(
                        "b12x DeepSeek V4 fixed compressed MLA workspace requested "
                        "before the joint attention arena was initialized"
                    )
                factory = (
                    B12XAttentionWorkspace.for_fixed_capacity
                    if fixed
                    else B12XAttentionWorkspace.for_contract
                )
                kwargs = {}
                workspace = factory(
                    mode="decode",
                    device=self.device,
                    dtype=torch.bfloat16,
                    kv_dtype=torch.uint8,
                    num_q_heads=self.num_q_heads,
                    indexer_num_q_heads=self.index_num_q_heads,
                    head_dim=B12X_COMPRESSED_MLA_HEAD_DIM,
                    v_head_dim=self.head_dim_v,
                    topk=selected_width,
                    max_page_table_width=1,
                    max_total_q=q_rows,
                    max_batch=q_rows,
                    max_paged_q_rows=q_rows,
                    max_kv_rows=0,
                    page_size=64,
                    use_cuda_graph=fixed,
                    max_chunks_per_row=split_chunks,
                    **kwargs,
                )
            cache[key] = workspace
        return workspace

    def _find_b12x_compressed_mla_workspace(
        self,
        *,
        q_rows: int,
        selected_width: int,
        split_chunks: int,
        max_q_rows: Optional[int] = None,
    ):
        candidates = []
        for (
            fixed,
            _q_key,
            _width_key,
            _chunks_key,
            num_q_heads,
            index_num_q_heads,
        ), workspace in self._b12x_compressed_workspaces.items():
            if not fixed:
                continue
            if num_q_heads != self.num_q_heads or index_num_q_heads != self.index_num_q_heads:
                continue
            if int(getattr(workspace, "max_total_q", 0)) < q_rows:
                continue
            if max_q_rows is not None and int(
                getattr(workspace, "max_total_q", 0)
            ) > int(max_q_rows):
                continue
            if int(getattr(workspace, "topk", 0)) < selected_width:
                continue
            if int(getattr(workspace, "max_chunks_per_row", 0)) < split_chunks:
                continue
            candidates.append(workspace)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda workspace: (
                int(getattr(workspace, "max_total_q", 0)),
                int(getattr(workspace, "topk", 0)),
                int(getattr(workspace, "max_chunks_per_row", 0)),
            ),
        )

    def _prime_b12x_fixed_workspaces_for_metadata(
        self,
        metadata: Union[
            DSV4Metadata,
            DSV4RawVerifyMetadata,
            DSV4RawDecodeMetadata,
        ],
        *,
        compressed_q_rows: int,
        indexer_q_rows: int,
        indexer_logits_mode: Literal["dense", "tiled"] = "tiled",
    ) -> None:
        if not isinstance(metadata, DSV4Metadata):
            return
        core_metadata = metadata.core_attn_metadata
        selected_widths = {core_metadata.swa_page_indices.shape[1]}
        if core_metadata.c4_sparse_page_indices is not None:
            selected_widths.add(
                core_metadata.swa_page_indices.shape[1]
                + core_metadata.c4_sparse_page_indices.shape[1]
            )
        if core_metadata.c128_page_indices is not None:
            selected_widths.add(
                core_metadata.swa_page_indices.shape[1]
                + core_metadata.c128_page_indices.shape[1]
            )

        for selected_width in selected_widths:
            self._get_b12x_compressed_mla_workspace(
                q_rows=compressed_q_rows,
                selected_width=selected_width,
                fixed=True,
            )

        if metadata.indexer_metadata is not None:
            indexer_workspace = self._get_b12x_indexer_paged_workspace(
                q_rows=indexer_q_rows,
                page_table_width=metadata.indexer_metadata.page_table.shape[1],
                fixed=True,
                logits_mode=indexer_logits_mode,
            )
            if indexer_logits_mode == "tiled":
                self._prewarm_b12x_tiled_indexer_workspace(
                    indexer_workspace=indexer_workspace,
                    page_table_width=metadata.indexer_metadata.page_table.shape[1],
                    indexer_q_rows=indexer_q_rows,
                )

    def _prewarm_b12x_tiled_indexer_workspace(
        self,
        *,
        indexer_workspace,
        page_table_width: int,
        indexer_q_rows: int,
    ) -> None:
        if not self._b12x_uses_c4_attention():
            return
        if hasattr(indexer_workspace, "prewarm_paged_indexer_tiled_topk"):
            indexer_workspace.prewarm_paged_indexer_tiled_topk()
        if hasattr(indexer_workspace, "prewarm_paged_indexer_tiled_scorer"):
            supertile_tokens = self._b12x_c4_indexer_supertile_tokens_capacity(
                page_table_width=page_table_width,
                q_rows=indexer_q_rows,
            )
            indexer_workspace.prewarm_paged_indexer_tiled_scorer(
                index_k_cache=self.token_to_kv_pool.get_index_k_with_scale_buffer(
                    self._b12x_first_c4_layer_id()
                ),
                width_tokens=supertile_tokens,
            )

    def _prime_b12x_cuda_graph_workspaces(
        self,
        metadata: Union[
            DSV4Metadata,
            DSV4RawVerifyMetadata,
            DSV4RawDecodeMetadata,
        ],
        *,
        q_rows: int,
    ) -> None:
        self._prime_b12x_fixed_workspaces_for_metadata(
            metadata,
            compressed_q_rows=q_rows,
            indexer_q_rows=q_rows,
            indexer_logits_mode="dense",
        )

    def _prime_b12x_default_fixed_workspaces(
        self,
        *,
        compressed_q_rows: Optional[int] = None,
        indexer_q_rows: Optional[int] = None,
        indexer_logits_mode: Literal["dense", "tiled"] = "tiled",
        log: bool = True,
    ) -> None:
        if compressed_q_rows is None:
            compressed_q_rows = self._b12x_compressed_prefill_q_capacity()
        if indexer_q_rows is None:
            indexer_q_rows = self._b12x_eager_extend_total_q_capacity()
        selected_widths = self._b12x_compressed_selected_widths()
        for selected_width in selected_widths:
            self._get_b12x_compressed_mla_workspace(
                q_rows=compressed_q_rows,
                selected_width=selected_width,
                fixed=True,
            )

        page_table_width = 0
        if self._b12x_uses_c4_attention():
            page_table_width = self._b12x_indexer_page_table_width_capacity()
            indexer_workspace = self._get_b12x_indexer_paged_workspace(
                q_rows=indexer_q_rows,
                page_table_width=page_table_width,
                fixed=True,
                logits_mode=indexer_logits_mode,
            )
            if indexer_logits_mode == "tiled":
                self._prewarm_b12x_tiled_indexer_workspace(
                    indexer_workspace=indexer_workspace,
                    page_table_width=page_table_width,
                    indexer_q_rows=indexer_q_rows,
                )
        if log:
            logger.info(
                "Initialized b12x compressed MLA fixed workspaces: "
                "compressed_q_capacity=%s indexer_q_capacity=%s "
                "selected_widths=%s page_table_width=%s",
                compressed_q_rows,
                indexer_q_rows,
                list(selected_widths),
                page_table_width,
            )

    def _prime_b12x_default_cuda_graph_workspaces(self, *, q_rows: int) -> None:
        self._prime_b12x_default_fixed_workspaces(
            compressed_q_rows=q_rows,
            indexer_q_rows=q_rows,
            indexer_logits_mode="dense",
            log=False,
        )

    def get_b12x_indexer_paged_workspace(
        self,
        *,
        forward_batch: ForwardBatch,
        q_rows: int,
        page_table_width: int,
        logits_mode: Literal["dense", "tiled"] = "tiled",
    ):
        fixed = self._use_b12x_fixed_workspace(forward_batch)
        prefill_q_capacity = self._b12x_prefill_q_capacity(
            forward_batch=forward_batch,
            q_rows=q_rows,
        )
        if prefill_q_capacity is not None:
            fixed = True
            q_rows = prefill_q_capacity
        return self._get_b12x_indexer_paged_workspace(
            q_rows=q_rows,
            page_table_width=page_table_width,
            fixed=fixed,
            logits_mode=logits_mode,
        )

    def _get_b12x_indexer_paged_workspace(
        self,
        *,
        q_rows: int,
        page_table_width: int,
        fixed: bool,
        logits_mode: Literal["dense", "tiled"] = "tiled",
    ):
        if not fixed:
            return None

        from b12x.attention.workspace import B12XAttentionWorkspaceContract

        q_rows = max(int(q_rows), 1)
        page_table_width_capacity = self._b12x_indexer_page_table_width_capacity()
        page_table_width = min(max(int(page_table_width), 1), page_table_width_capacity)
        if logits_mode == "dense":
            workspace_page_table_width = page_table_width_capacity
        elif logits_mode == "tiled":
            workspace_page_table_width = page_table_width_capacity
        else:
            raise ValueError(f"unknown b12x C4 indexer logits mode {logits_mode!r}")
        key = (
            logits_mode,
            q_rows,
            workspace_page_table_width,
            self.num_q_heads,
            self.index_num_q_heads,
        )
        workspace = self._b12x_indexer_workspaces.get(key)
        if workspace is None:
            workspace = self._find_b12x_indexer_paged_workspace(
                q_rows=q_rows,
                page_table_width=workspace_page_table_width,
                logits_mode=logits_mode,
            )
        if workspace is None:
            if self._b12x_attention_bundle is not None:
                contract = B12XAttentionWorkspaceContract(
                    mode="decode",
                    max_total_q=1,
                    max_batch=1,
                    max_paged_q_rows=q_rows,
                    max_kv_rows=0,
                    v_head_dim=self.head_dim_v,
                    indexer_num_q_heads=self.index_num_q_heads,
                    max_page_table_width=workspace_page_table_width,
                    topk=self.c4_topk,
                )
                workspace = self._b12x_attention_bundle.arena.make_workspace(
                    contract,
                    use_cuda_graph=True,
                )
            else:
                raise RuntimeError(
                    "b12x DeepSeek V4 fixed paged indexer workspace requested "
                    "before the joint attention arena was initialized"
                )
            self._b12x_indexer_workspaces[key] = workspace
        return workspace

    def _find_b12x_indexer_paged_workspace(
        self,
        *,
        q_rows: int,
        page_table_width: int,
        logits_mode: Literal["dense", "tiled"] = "tiled",
    ):
        candidates = []
        for (
            mode_key,
            _q_key,
            _width_key,
            num_q_heads,
            index_num_q_heads,
        ), workspace in self._b12x_indexer_workspaces.items():
            if mode_key != logits_mode:
                continue
            if num_q_heads != self.num_q_heads or index_num_q_heads != self.index_num_q_heads:
                continue
            if int(getattr(workspace, "max_paged_q_rows", 0)) < q_rows:
                continue
            if int(getattr(workspace, "max_page_table_width", 0)) < page_table_width:
                continue
            if logits_mode == "dense" and getattr(workspace, "indexer_paged_logits", None) is None:
                continue
            if logits_mode == "tiled" and getattr(workspace, "indexer_extend_tile_logits", None) is None:
                continue
            candidates.append(workspace)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda workspace: (
                int(getattr(workspace, "max_paged_q_rows", 0)),
                int(getattr(workspace, "max_page_table_width", 0)),
            ),
        )

    def init_forward_metadata_decode(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> Union[DSV4Metadata, DSV4RawDecodeMetadata]:
        assert (
            req_pool_indices.shape[0] == seq_lens.shape[0] == out_cache_loc.shape[0]
        ), f"{req_pool_indices.shape=} {seq_lens.shape=} {out_cache_loc.shape=}"

        if self._use_prep_in_cuda_graph and envs.SGLANG_PREP_IN_CUDA_GRAPH.get():
            return DSV4RawDecodeMetadata(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
            )

        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices,
            seq_lens_casual=seq_lens,
            max_seq_len=max_seq_len,
            out_loc=out_cache_loc,
            need_compress=self._b12x_uses_compressed_attention(),
        )

        indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata)
            if self._b12x_uses_c4_attention()
            else None
        )

        create = functools.partial(
            create_paged_compressor_data,
            is_prefill=False,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
        )

        need_compress = self._b12x_uses_compressed_attention()
        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4) if need_compress else None,
            c128_compress_metadata=create(compress_ratio=128) if need_compress else None,
        )

    def init_forward_metadata_prefill(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: List[int],
        out_cache_loc: torch.Tensor,
        num_tokens: int,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: List[int],
        need_compress: bool = True,
        use_prefill_cuda_graph: bool = False,
    ) -> DSV4Metadata:
        seq_lens_casual, req_pool_indices_repeated = self.expand_prefill_casually(
            num_tokens=num_tokens,
            seq_lens=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens_cpu,
            req_pool_indices=req_pool_indices,
            padded_num_tokens=out_cache_loc.shape[0],
        )
        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=max_seq_len,
            out_loc=out_cache_loc,
            need_compress=need_compress,
            is_prefill=True,
        )
        indexer_metadata = (
            self.init_forward_metadata_indexer(
                core_attn_metadata,
                shared_page_table=len(seq_lens_cpu) == 1,
            )
            if need_compress and self._b12x_uses_c4_attention()
            else None
        )
        if not need_compress:
            create = _create_dummy_paged_compress_data
        else:
            create = functools.partial(
                create_paged_compressor_data,
                is_prefill=True,
                token_to_kv_pool=self.token_to_kv_pool,
                req_to_token=self.req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                extend_lens=extend_seq_lens,
                extend_lens_cpu=extend_seq_lens_cpu,
                use_prefill_cuda_graph=use_prefill_cuda_graph,
            )
        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4),
            c128_compress_metadata=create(compress_ratio=128),
        )

    def init_forward_metadata_target_verify(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
    ) -> Union[DSV4Metadata, DSV4RawVerifyMetadata]:
        if self._use_prep_in_cuda_graph and envs.SGLANG_PREP_IN_CUDA_GRAPH.get():
            assert out_cache_loc is not None
            if not hasattr(self, "extend_seq_lens_buffer"):
                self.extend_seq_lens_buffer = torch.tensor(
                    [self.speculative_num_draft_tokens] * 1025, device=self.device
                )
            extend_seq_lens = self.extend_seq_lens_buffer[: len(seq_lens)]

            return DSV4RawVerifyMetadata(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
                extend_seq_lens=extend_seq_lens,
            )
        else:
            seq_lens_cpu = seq_lens.tolist()
            return self.init_forward_metadata_target_verify_old(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                out_cache_loc=out_cache_loc,
                use_prefill_cuda_graph=use_prefill_cuda_graph,
            )

    def init_forward_metadata_target_verify_old(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[List[int]] = None,
        out_cache_loc: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
    ) -> DSV4Metadata:
        batch_size = len(seq_lens)
        seq_lens = seq_lens + self.speculative_num_draft_tokens
        seq_lens_cpu = [x + self.speculative_num_draft_tokens for x in seq_lens_cpu]
        extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * batch_size
        extend_seq_lens = self._move_to_device(extend_seq_lens_cpu)
        num_tokens = self.speculative_num_draft_tokens * batch_size
        if out_cache_loc is None:
            out_cache_loc = seq_lens.new_zeros(num_tokens)
        return self.init_forward_metadata_prefill(
            max_seq_len=max_seq_len,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            num_tokens=num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            need_compress=self._b12x_uses_compressed_attention(),
            use_prefill_cuda_graph=use_prefill_cuda_graph,
        )

    def make_forward_metadata_from_raw_verify(
        self, raw_metadata: DSV4RawVerifyMetadata
    ) -> DSV4Metadata:
        req_pool_indices = raw_metadata.req_pool_indices
        seq_lens = raw_metadata.seq_lens
        out_cache_loc = raw_metadata.out_cache_loc

        bs, num_draft_tokens = len(seq_lens), self.speculative_num_draft_tokens
        seq_lens = seq_lens + self.speculative_num_draft_tokens
        extend_seq_lens = raw_metadata.extend_seq_lens

        seq_lens_casual, req_pool_indices_repeated = (
            self.expand_extend_with_same_length(
                bs, num_draft_tokens, seq_lens, req_pool_indices
            )
        )
        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
            out_loc=out_cache_loc,
            need_compress=self._b12x_uses_compressed_attention(),
        )
        indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata)
            if self._b12x_uses_c4_attention()
            else None
        )
        create = functools.partial(
            create_paged_compressor_data,
            is_prefill=True,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_lens=extend_seq_lens,
            seq_lens_cpu=None,
            extend_lens_cpu=None,
            use_prefill_cuda_graph=True,
            num_q_tokens=num_draft_tokens * bs,
        )
        need_compress = self._b12x_uses_compressed_attention()
        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4) if need_compress else None,
            c128_compress_metadata=create(compress_ratio=128) if need_compress else None,
        )

    def make_forward_metadata_from_raw_decode(
        self, raw_metadata: DSV4RawDecodeMetadata
    ) -> DSV4Metadata:
        req_pool_indices = raw_metadata.req_pool_indices
        seq_lens = raw_metadata.seq_lens
        out_cache_loc = raw_metadata.out_cache_loc

        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices,
            seq_lens_casual=seq_lens,
            max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
            out_loc=out_cache_loc,
            need_compress=self._b12x_uses_compressed_attention(),
        )
        indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata)
            if self._b12x_uses_c4_attention()
            else None
        )

        create = functools.partial(
            create_paged_compressor_data,
            is_prefill=False,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
        )

        need_compress = self._b12x_uses_compressed_attention()
        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4) if need_compress else None,
            c128_compress_metadata=create(compress_ratio=128) if need_compress else None,
        )

    def init_forward_metadata_draft_extend(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: List[int],
        num_tokens_per_bs: int,
        out_cache_loc: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
    ) -> DSV4Metadata:
        batch_size = len(seq_lens)
        extend_seq_lens_cpu = [num_tokens_per_bs] * batch_size
        extend_seq_lens = self._move_to_device(extend_seq_lens_cpu)
        num_tokens = num_tokens_per_bs * batch_size
        if out_cache_loc is None:
            out_cache_loc = seq_lens.new_zeros(num_tokens)
        return self.init_forward_metadata_prefill(
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            req_pool_indices=req_pool_indices,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            num_tokens=num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            need_compress=False,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            return

        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        seq_lens_cpu = forward_batch.seq_lens_cpu
        assert forward_batch.req_to_token_pool.req_to_token is self.req_to_token

        assert self.swa_page_size % SWA_WINDOW == 0 and self.page_size % 128 == 0
        assert seq_lens_cpu is not None
        max_seq_len = int(seq_lens_cpu.max().item())
        prefill_num_tokens: Optional[int] = None

        if forward_batch.forward_mode.is_decode_or_idle():
            metadata = self.init_forward_metadata_decode(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=forward_batch.out_cache_loc,
            )
        elif forward_batch.forward_mode.is_target_verify():
            metadata = self.init_forward_metadata_target_verify(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=forward_batch.out_cache_loc,
            )
        elif forward_batch.forward_mode.is_prefill(include_draft_extend_v2=True):
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            extend_seq_lens = forward_batch.extend_seq_lens
            assert (
                seq_lens is not None
                and seq_lens_cpu is not None
                and extend_seq_lens is not None
                and extend_seq_lens_cpu is not None
            )
            is_draft = forward_batch.forward_mode.is_draft_extend(include_v2=True)
            prefill_num_tokens = sum(extend_seq_lens_cpu)
            metadata = self.init_forward_metadata_prefill(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu.tolist(),
                out_cache_loc=forward_batch.out_cache_loc,
                num_tokens=prefill_num_tokens,
                extend_seq_lens=extend_seq_lens,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                need_compress=(not is_draft) and self._b12x_uses_compressed_attention(),
            )
        else:
            raise NotImplementedError(f"unsupported mode {forward_batch.forward_mode=}")

        self.forward_metadata = metadata
        if prefill_num_tokens is not None and self._b12x_uses_chunked_prefill_workspace(
            forward_batch.forward_mode
        ):
            prefill_q_capacity = self._b12x_prefill_q_capacity(
                forward_batch=forward_batch,
                q_rows=prefill_num_tokens,
            )
            assert prefill_q_capacity is not None
            self._prime_b12x_fixed_workspaces_for_metadata(
                metadata,
                compressed_q_rows=self._b12x_compressed_prefill_q_capacity(),
                indexer_q_rows=prefill_q_capacity,
            )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        self.cuda_graph_metadata_of_bucket_and_bs: Dict[
            _GraphBucket,
            Dict[
                int,
                Union[DSV4Metadata, DSV4RawDecodeMetadata, DSV4RawVerifyMetadata],
            ],
        ] = {bucket: {} for bucket in _GraphBucket}
        self.draft_extend_num_tokens_per_bs = (
            max_num_tokens // max_bs if max_bs > 0 else 1
        )
        self._prime_b12x_default_cuda_graph_workspaces(q_rows=max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ) -> None:
        assert req_pool_indices.size(0) == bs
        assert seq_lens.size(0) == bs

        bucket = _GraphBucket.of(forward_mode)
        raw_type: Optional[type] = None
        if bucket == _GraphBucket.DECODE_OR_IDLE:
            metadata = self.init_forward_metadata_decode(
                max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=torch.zeros_like(seq_lens),
            )
            raw_type = DSV4RawDecodeMetadata
        elif bucket == _GraphBucket.TARGET_VERIFY:
            out_cache_loc = torch.zeros(num_tokens, **self.cuda_int32_kwargs)
            metadata = self.init_forward_metadata_target_verify(
                max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
                use_prefill_cuda_graph=True,
            )
            raw_type = DSV4RawVerifyMetadata
        elif bucket == _GraphBucket.DRAFT_EXTEND:
            num_tokens_per_bs = num_tokens // bs
            metadata = self.init_forward_metadata_draft_extend(
                max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens.tolist(),
                num_tokens_per_bs=num_tokens_per_bs,
                use_prefill_cuda_graph=True,
            )
        else:
            raise NotImplementedError(f"{forward_mode=} not supported yet")

        self._prime_b12x_cuda_graph_workspaces(metadata, q_rows=num_tokens)
        self.cuda_graph_metadata_of_bucket_and_bs[bucket][bs] = metadata
        self.forward_metadata = metadata
        if raw_type is not None:
            self._current_capture_raw = (
                metadata if isinstance(metadata, raw_type) else None
            )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ) -> None:
        bucket = _GraphBucket.of(forward_mode)

        # FIXME: see cuda_graph_runner — this attribute is set out-of-band.
        fb = self._replay_forward_batch
        out_cache_loc = fb.out_cache_loc
        actual_forward_mode = fb.forward_mode

        if actual_forward_mode == ForwardMode.IDLE:
            logger.debug(
                f"[IDLE replay] bs={bs}, "
                f"local_seq_lens_len={len(seq_lens)}, "
                f"has_graph={bs in self.cuda_graph_metadata_of_bucket_and_bs[_GraphBucket.DECODE_OR_IDLE]}"
            )
            device = seq_lens.device
            seq_lens = torch.ones(bs, dtype=seq_lens.dtype, device=device)
            seq_lens_cpu = torch.ones(bs, dtype=torch.int64)
            seq_lens_sum = bs
            req_pool_indices = torch.zeros(
                bs, dtype=req_pool_indices.dtype, device=device
            )
            out_cache_loc = torch.zeros(bs, dtype=torch.int64, device=device)

        assert seq_lens_cpu is not None
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]

        actual_max_seq_len = seq_lens_cpu.max().item()
        chosen_max_seq_len = self.MAX_SEQ_LEN_FOR_CAPTURE
        assert actual_max_seq_len <= chosen_max_seq_len

        if bucket == _GraphBucket.DECODE_OR_IDLE:
            assert out_cache_loc is not None
            assert len(out_cache_loc.shape) == 1, f"{out_cache_loc.shape=}"
            out_cache_loc_padded = torch.nn.functional.pad(
                out_cache_loc,
                pad=(0, bs - len(out_cache_loc)),
                mode="constant",
                value=0,
            )
            temp_metadata = self.init_forward_metadata_decode(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc_padded,
            )
        elif bucket == _GraphBucket.TARGET_VERIFY:
            assert out_cache_loc is not None
            num_tokens = self.speculative_num_draft_tokens * bs
            out_cache_loc_padded = torch.nn.functional.pad(
                out_cache_loc,
                pad=(0, num_tokens - len(out_cache_loc)),
                mode="constant",
                value=0,
            )
            temp_metadata = self.init_forward_metadata_target_verify(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc_padded,
                use_prefill_cuda_graph=True,
            )
        elif bucket == _GraphBucket.DRAFT_EXTEND:
            num_tokens_per_bs = self.draft_extend_num_tokens_per_bs
            temp_metadata = self.init_forward_metadata_draft_extend(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu.tolist(),
                num_tokens_per_bs=num_tokens_per_bs,
                use_prefill_cuda_graph=True,
            )
        else:
            raise NotImplementedError

        self.replay_cuda_graph_metadata_from(
            bs=bs, temp_metadata=temp_metadata, bucket=bucket
        )

    def replay_cuda_graph_metadata_from(
        self,
        bs: int,
        temp_metadata: Union[
            DSV4Metadata,
            DSV4RawVerifyMetadata,
            DSV4RawDecodeMetadata,
        ],
        bucket: _GraphBucket,
    ) -> None:
        chosen_metadata = self.cuda_graph_metadata_of_bucket_and_bs[bucket][bs]
        chosen_metadata.copy_(temp_metadata)
        self.forward_metadata = chosen_metadata

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def on_after_cuda_graph_warmup(self):
        # PREP_IN_CUDA_GRAPH=True: warmup upgraded raw->full on the host;
        # restore raw so capture re-runs the upgrade inside the graph.
        current_raw = getattr(self, "_current_capture_raw", None)
        if current_raw is not None:
            self.forward_metadata = current_raw

    def store_cache(
        self, layer_id: int, swa_k: torch.Tensor, forward_batch: ForwardBatch
    ) -> None:
        raw_loc = forward_batch.out_cache_loc
        if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
            self.token_to_kv_pool.set_swa_key_buffer_radix_fused(
                layer_id=layer_id,
                raw_loc=raw_loc,
                cache_k=swa_k,
            )
        else:
            swa_k_pack = quant_to_nope_fp8_rope_bf16_pack_triton(swa_k)
            self.token_to_kv_pool.set_swa_key_buffer_radix(
                layer_id=layer_id,
                raw_loc=raw_loc,
                cache_nope_fp8_rope_bf16_pack=swa_k_pack,
            )

    def _maybe_upgrade_forward_metadata(self) -> None:
        # With SGLANG_PREP_IN_CUDA_GRAPH=1, init_forward_metadata_*
        # returns a Raw metadata that only carries a few tensors. The
        # full DSV4Metadata (including c4/c128 compress + core_attn +
        # indexer metadata) must be materialized before any caller that
        # touches those fields. For 1.6T the first two layers have
        # compress_ratio=128, so forward_core_compressor / forward_c4_indexer
        # can fire before attn_backend.forward(), and must trigger the
        # upgrade themselves.
        if isinstance(self.forward_metadata, DSV4RawVerifyMetadata):
            self.forward_metadata = self.make_forward_metadata_from_raw_verify(
                raw_metadata=self.forward_metadata,
            )
        elif isinstance(self.forward_metadata, DSV4RawDecodeMetadata):
            self.forward_metadata = self.make_forward_metadata_from_raw_decode(
                raw_metadata=self.forward_metadata,
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        compress_ratio: Literal[0, 4, 128],
        save_kv_cache: bool = True,
        attn_sink: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        self._maybe_upgrade_forward_metadata()

        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], q.shape[1], layer.v_head_dim)

        assert k is v, "DeepseekV4 shares k and v"
        swa_k = k

        layer_id = layer.layer_id
        metadata = self.forward_metadata
        core_attn_metadata = metadata.core_attn_metadata
        token_to_kv_pool = forward_batch.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        if isinstance(core_attn_metadata, DSV4AttnMetadata):
            if save_kv_cache:
                self.store_cache(layer_id, swa_k, forward_batch)
            swa_k_cache = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)

            extra_k_cache, extra_indices, extra_topk_lengths = None, None, None
            indexed_page_table = None
            if compress_ratio == 4:
                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                extra_indices = core_attn_metadata.c4_sparse_page_indices
                extra_topk_lengths = core_attn_metadata.c4_sparse_topk_lengths
                if forward_batch.hisparse_coordinator is None:
                    indexed_page_table = core_attn_metadata.page_table
            elif compress_ratio == 128:
                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                extra_indices = core_attn_metadata.c128_page_indices
                extra_topk_lengths = core_attn_metadata.c128_topk_lengths_clamp1

            swa_page_indices = core_attn_metadata.swa_page_indices
            swa_topk_lengths = core_attn_metadata.swa_topk_lengths

            if self.mtp_enabled:
                if swa_page_indices.shape[0] != q.shape[0]:
                    swa_page_indices = _pad_tensor_to_size(
                        swa_page_indices, q.shape[0], value=0
                    )

                if swa_topk_lengths.shape[0] != q.shape[0]:
                    swa_topk_lengths = _pad_tensor_to_size(
                        swa_topk_lengths, q.shape[0], value=1
                    )
                if extra_indices is not None and extra_indices.shape[0] != q.shape[0]:
                    extra_indices = _pad_tensor_to_size(
                        extra_indices, q.shape[0], value=-1
                    )
                if (
                    extra_topk_lengths is not None
                    and extra_topk_lengths.shape[0] != q.shape[0]
                ):
                    extra_topk_lengths = _pad_tensor_to_size(
                        extra_topk_lengths, q.shape[0], value=0
                    )
                if (
                    indexed_page_table is not None
                    and indexed_page_table.shape[0] != q.shape[0]
                ):
                    indexed_page_table = _pad_tensor_to_size(
                        indexed_page_table, q.shape[0], value=0
                    )

            if q.ndim == 4:
                assert q.shape[1] == 1, f"expected singleton MQA dim, got {q.shape=}"
                q = q.squeeze(1)
            assert q.ndim == 3, f"expected q rank 3, got {q.shape=}"
            assert swa_page_indices.ndim == 2, f"{swa_page_indices.shape=}"
            if extra_indices is not None:
                assert extra_indices.ndim == 2, f"{extra_indices.shape=}"

            assert attn_sink is not None

            assert (
                swa_page_indices.shape[-1] % 64 == 0
            ), f"{swa_page_indices.shape=}'s last dimension is not aligned to 64"
            if extra_indices is not None:
                assert (
                    extra_indices.shape[-1] % 64 == 0
                ), f"{extra_indices.shape=}'s last dimension is not aligned to 64"

            indexed_page_size = None
            if extra_k_cache is not None:
                indexed_page_size = token_to_kv_pool.page_size // compress_ratio

            selected_width = swa_page_indices.shape[1] + (
                extra_indices.shape[1] if extra_indices is not None else 0
            )

            from b12x.integration.mla import compressed_mla_decode_forward

            def _forward_compressed_slice(row_start: int, row_end: int) -> torch.Tensor:
                workspace = self.get_b12x_compressed_mla_workspace(
                    forward_batch=forward_batch,
                    q_rows=row_end - row_start,
                    selected_width=selected_width,
                )
                indexed_indices_slice = (
                    None
                    if extra_indices is None
                    else extra_indices[row_start:row_end]
                )
                indexed_lengths_slice = (
                    None
                    if extra_topk_lengths is None
                    else extra_topk_lengths[row_start:row_end]
                )
                indexed_page_table_slice = (
                    None
                    if indexed_page_table is None
                    else indexed_page_table[row_start:row_end]
                )
                return compressed_mla_decode_forward(
                    q_all=q[row_start:row_end],
                    swa_k_cache=swa_k_cache,
                    swa_indices=swa_page_indices[row_start:row_end],
                    swa_topk_lengths=swa_topk_lengths[row_start:row_end],
                    swa_page_size=token_to_kv_pool.swa_page_size,
                    indexed_k_cache=extra_k_cache,
                    indexed_indices=indexed_indices_slice,
                    indexed_topk_lengths=indexed_lengths_slice,
                    indexed_page_size=indexed_page_size,
                    indexed_page_table=indexed_page_table_slice,
                    attn_sink=attn_sink,
                    workspace=workspace,
                    sm_scale=self.softmax_scale,
                    expected_num_q_heads=self.num_q_heads,
                )

            q_rows = q.shape[0]
            if self._b12x_uses_chunked_prefill_workspace(
                forward_batch.forward_mode
            ):
                compressed_q_capacity = self._b12x_compressed_prefill_q_capacity()
                if q_rows > compressed_q_capacity:
                    if (
                        get_is_capture_mode()
                        or (
                            self.device.type == "cuda"
                            and torch.cuda.is_current_stream_capturing()
                        )
                    ):
                        raise RuntimeError(
                            "b12x DeepSeek V4 compressed MLA prefill slicing cannot "
                            "allocate its output while CUDA graph capture is active: "
                            f"q_rows={q_rows}, "
                            f"compressed_q_capacity={compressed_q_capacity}"
                        )
                    output = q.new_empty(q_rows, q.shape[1], layer.v_head_dim)
                    for row_start in range(0, q_rows, compressed_q_capacity):
                        row_end = min(row_start + compressed_q_capacity, q_rows)
                        output[row_start:row_end].copy_(
                            _forward_compressed_slice(row_start, row_end)
                        )
                    return output

            return _forward_compressed_slice(0, q_rows)

        raise NotImplementedError("ragged attention")

    def expand_prefill_casually(
        self,
        num_tokens: int,
        seq_lens: List[int],
        extend_seq_lens: List[int],
        req_pool_indices: torch.Tensor,
        padded_num_tokens: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_lens_casual = torch.empty(num_tokens, **self.cuda_int32_kwargs)
        idx_to_req_repeated = torch.empty(num_tokens, **self.cuda_int32_kwargs)
        offset = 0
        for i, (kv_len, qo_len) in enumerate(zip(seq_lens, extend_seq_lens)):
            out = seq_lens_casual[offset : offset + qo_len]
            offset += qo_len
            torch.arange(kv_len - qo_len + 1, kv_len + 1, out=out)
            idx_to_req_repeated[offset - qo_len : offset].fill_(i)

        assert offset == num_tokens
        req_pool_indices_repeated = req_pool_indices[idx_to_req_repeated]

        if padded_num_tokens is not None and padded_num_tokens > num_tokens:
            pad_size = padded_num_tokens - num_tokens
            seq_lens_casual = torch.nn.functional.pad(
                seq_lens_casual,
                (0, pad_size),
                value=1,
            )
            req_pool_indices_repeated = torch.nn.functional.pad(
                req_pool_indices_repeated,
                (0, pad_size),
                value=req_pool_indices_repeated[-1].item(),
            )

        return seq_lens_casual, req_pool_indices_repeated

    def expand_extend_with_same_length(
        self,
        bs: int,
        qo_len: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ):
        seq_lens_casual = seq_lens[:, None] + torch.arange(
            -qo_len + 1, 1, **self.cuda_int32_kwargs
        )
        seq_lens_casual = seq_lens_casual.flatten()
        idx_to_req_repeated = torch.arange(
            bs, **self.cuda_int32_kwargs
        ).repeat_interleave(qo_len)
        req_pool_indices_repeated = req_pool_indices[idx_to_req_repeated]
        return seq_lens_casual, req_pool_indices_repeated

    def make_core_attn_metadata(
        self,
        req_to_token: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        max_seq_len: int,
        out_loc: torch.Tensor,
        need_compress: bool = True,
        is_prefill: bool = False,
    ) -> DSV4AttnMetadata:
        assert self.swa_page_size == SWA_WINDOW

        swa_page_indices = self.get_swa_page_indices(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
        )

        swa_page_indices = _pad_last_dim(
            swa_page_indices, multiples_of=PAGE_INDEX_ALIGNED_SIZE
        )

        raw_positions = seq_lens_casual - 1
        swa_topk_lengths = torch.clamp(seq_lens_casual, max=SWA_WINDOW).to(
            torch.int32
        )
        swa_topk_lengths = swa_topk_lengths.contiguous()

        page_table = req_to_token[
            req_pool_indices_repeated, : max_seq_len : self.page_size
        ]
        page_table = (page_table // self.page_size).to(torch.int32).contiguous()

        core_attn_metadata = DSV4AttnMetadata(
            page_size=self.page_size,
            raw_out_loc=out_loc,
            seq_lens_casual=seq_lens_casual,
            cuda_int32_kwargs=self.cuda_int32_kwargs,
            positions_casual=raw_positions,
            page_table=page_table,
            swa_page_indices=swa_page_indices,
            swa_topk_lengths=swa_topk_lengths,
            c4_sparse_topk=self.c4_topk,
            c128_index_capacity=(
                self._b12x_c128_index_width_capacity()
                if need_compress and self._b12x_uses_c128_attention()
                else None
            ),
        )

        if need_compress:
            core_attn_metadata.init_compression_metadata()
            core_attn_metadata.init_sparse_mla_related()
        else:
            core_attn_metadata.c4_sparse_topk_lengths = None
            core_attn_metadata.c4_sparse_page_indices = None
        return core_attn_metadata

    def get_swa_page_indices(
        self,
        seq_lens_casual: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
    ) -> torch.Tensor:
        pos_causal = seq_lens_casual - 1
        num_qo_tokens = seq_lens_casual.size(0)
        offsets = pos_causal.unsqueeze(1) - torch.arange(
            SWA_WINDOW, **self.cuda_int32_kwargs
        ).unsqueeze(0)
        invalid_offset_mask = offsets < 0
        offsets.masked_fill_(invalid_offset_mask, 0)
        raw_indices = self.req_to_token[req_pool_indices_repeated[:, None], offsets]
        assert raw_indices.shape == (num_qo_tokens, SWA_WINDOW)
        raw_indices.masked_fill_(invalid_offset_mask, -1)
        swa_indices = self.token_to_kv_pool.translate_loc_from_full_to_swa(raw_indices)
        return swa_indices


class DeepseekV4MultiStepBackend(DeepseekV4AttnBackend):
    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner)
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends: List[DeepseekV4AttnBackend] = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                DeepseekV4AttnBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                )
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    def on_after_cuda_graph_warmup(self):
        for backend in self.attn_backends:
            backend.on_after_cuda_graph_warmup()

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        if self.speculative_num_steps == 1:
            return

        self.attn_backends[0]._replay_forward_batch = forward_batch
        self.attn_backends[0].init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            seq_lens_sum=forward_batch.seq_lens_sum,
            encoder_lens=None,
            forward_mode=ForwardMode.DECODE,
            spec_info=forward_batch.spec_info,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )
        self.attn_backends[0]._replay_forward_batch = None
        temp_metadata = self.attn_backends[0].forward_metadata

        for i in range(1, self.speculative_num_steps - 1):
            self.attn_backends[i].replay_cuda_graph_metadata_from(
                bs=bs,
                temp_metadata=temp_metadata,
                bucket=_GraphBucket.DECODE_OR_IDLE,
            )


def _pad_tensor_to_size(tensor: torch.Tensor, size: int, *, value: int = 0):
    if value == 0:
        return torch.cat(
            [tensor, tensor.new_zeros(size - tensor.shape[0], *tensor.shape[1:])],
            dim=0,
        )
    else:
        return torch.cat(
            [
                tensor,
                tensor.new_full((size - tensor.shape[0], *tensor.shape[1:]), value),
            ],
            dim=0,
        )
