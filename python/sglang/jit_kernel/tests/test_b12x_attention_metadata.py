from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.b12x_backend import (
    B12xAttnBackend,
    B12xForwardMetadata,
    _b12x_get_config_attr,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class DummySwaPool:
    def __init__(self, offset: int):
        self.offset = offset

    def translate_loc_from_full_to_swa(self, locs: torch.Tensor) -> torch.Tensor:
        return locs + self.offset


def make_backend(*, bs: int, swa_offset: int = 4096):
    backend = object.__new__(B12xAttnBackend)
    backend.page_size = 64
    backend.device = torch.device("cpu")
    backend.max_pages_per_req = 4
    backend.graph_page_offsets = torch.arange(0, 256, 64, dtype=torch.int64)
    backend.req_to_token = torch.zeros((1, 256), dtype=torch.int32)
    backend.req_to_token[0, 0] = 0
    backend.req_to_token[0, 64] = 0
    backend.req_to_token[0, 128] = 128
    backend.req_to_token[0, 192] = 192
    backend.swa_kv_pool = DummySwaPool(swa_offset)
    backend.use_sliding_window_kv_pool = True
    backend.cuda_graph_row_indices = torch.arange(bs, dtype=torch.long)
    backend.cuda_graph_workspaces = {}
    backend.server_args = SimpleNamespace(disable_overlap_schedule=True)
    return backend


def make_metadata(bs: int):
    return B12xForwardMetadata(
        cu_seqlens_q=torch.arange(0, bs + 1, dtype=torch.int32),
        cache_seqlens=torch.tensor([65] + [1] * (bs - 1), dtype=torch.int32),
        page_table=torch.zeros((bs, 4), dtype=torch.int32),
        swa_page_table=torch.zeros((bs, 4), dtype=torch.int32),
        mode="decode",
        use_cuda_graph=True,
    )


class FakePagedWorkspace:
    @staticmethod
    def eager_extend_work_items_capacity(
        *,
        max_total_q: int,
        num_q_heads: int,
        num_kv_heads: int,
    ) -> int:
        group_size = num_q_heads // num_kv_heads
        return max((max_total_q * group_size + 15) // 16, 1)


def test_verify_graph_capacity_covers_graph_budget_chunk_search(monkeypatch):
    backend = object.__new__(B12xAttnBackend)
    backend.workspace_cls = FakePagedWorkspace
    backend.page_size = 64
    backend.device = torch.device("cuda")
    backend.max_pages_per_req = 16384
    backend.q_dtype = torch.bfloat16
    backend.kv_cache_dtype = torch.float16

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=188),
    )

    work_items, partial_rows = backend._paged_verify_graph_capacities(
        batch_capacity=8,
        total_q_capacity=32,
        head_dim_qk=192,
        head_dim_vo=128,
        num_q_heads=32,
        num_kv_heads=2,
        window_left=-1,
        num_cache_pages=4,
    )

    assert work_items >= 188
    assert partial_rows >= 160


def test_b12x_moe_config_aliases_cover_minimax_m2():
    cfg = SimpleNamespace(
        hidden_size=3072,
        intermediate_size=1536,
        num_experts_per_tok=8,
        num_local_experts=256,
    )

    assert _b12x_get_config_attr(
        cfg, ("n_routed_experts", "num_experts", "num_local_experts")
    ) == 256
    assert _b12x_get_config_attr(
        cfg, ("moe_intermediate_size", "intermediate_size")
    ) == 1536


def test_decode_graph_metadata_refreshes_current_full_and_swa_page_at_boundary():
    bs = 8
    backend = make_backend(bs=bs)
    metadata = make_metadata(bs)
    req_pool_indices = torch.zeros(bs, dtype=torch.int64)

    backend._build_page_tables_into(
        req_pool_indices,
        metadata.cache_seqlens,
        metadata.page_table,
        metadata.swa_page_table,
        bs,
    )
    assert metadata.page_table[0, 1].item() == 0
    assert metadata.swa_page_table[0, 1].item() == 64

    forward_batch = SimpleNamespace(
        out_cache_loc=torch.tensor([192] + [0] * (bs - 1), dtype=torch.int64),
        out_cache_loc_swa=torch.tensor([4288] + [0] * (bs - 1), dtype=torch.int64),
    )
    backend._sanitize_decode_graph_padding_metadata(metadata, forward_batch)

    assert metadata.page_table[0, 1].item() == 3
    assert metadata.swa_page_table[0, 1].item() == 67
    assert metadata.cache_seqlens.tolist() == [65] + [1] * (bs - 1)
    assert metadata.page_table[1:].eq(0).all()
    assert metadata.swa_page_table[1:].eq(0).all()


def test_decode_graph_metadata_can_translate_current_swa_page_from_full_loc():
    bs = 8
    backend = make_backend(bs=bs)
    metadata = make_metadata(bs)
    req_pool_indices = torch.zeros(bs, dtype=torch.int64)

    backend._build_page_tables_into(
        req_pool_indices,
        metadata.cache_seqlens,
        metadata.page_table,
        metadata.swa_page_table,
        bs,
    )

    forward_batch = SimpleNamespace(
        out_cache_loc=torch.tensor([192] + [0] * (bs - 1), dtype=torch.int64),
    )
    backend._sanitize_decode_graph_padding_metadata(metadata, forward_batch)

    assert metadata.page_table[0, 1].item() == 3
    assert metadata.swa_page_table[0, 1].item() == 67


def test_decode_graph_replay_metadata_uses_padded_cache_loc_buffer_at_boundary():
    bs = 8
    backend = make_backend(bs=bs)
    backend.cuda_graph_cu_seqlens_q = torch.zeros(bs + 1, dtype=torch.int32)
    backend.cuda_graph_cache_seqlens = torch.zeros(bs, dtype=torch.int32)
    backend.cuda_graph_page_table = torch.zeros((bs, 4), dtype=torch.int32)
    backend.cuda_graph_swa_page_table = torch.zeros((bs, 4), dtype=torch.int32)
    backend._prepare_cuda_graph_workspaces = lambda *args, **kwargs: None

    backend.init_forward_metadata_replay_cuda_graph_with_cache_loc(
        bs=bs,
        req_pool_indices=torch.zeros(bs, dtype=torch.int64),
        seq_lens=torch.tensor([65] + [1] * (bs - 1), dtype=torch.int32),
        seq_lens_sum=65 + (bs - 1),
        encoder_lens=None,
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
        seq_lens_cpu=None,
        out_cache_loc=torch.tensor([192] + [0] * (bs - 1), dtype=torch.int64),
    )

    metadata = backend.forward_metadata
    assert metadata.page_table[0, 1].item() == 3
    assert metadata.swa_page_table[0, 1].item() == 67
    assert metadata.cache_seqlens.tolist() == [65] + [1] * (bs - 1)
    assert metadata.page_table[1:].eq(0).all()
    assert metadata.swa_page_table[1:].eq(0).all()
