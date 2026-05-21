from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.b12x_backend import (
    B12xAttnBackend,
    B12xForwardMetadata,
    _b12x_get_config_attr,
)
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
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
        active_total_q=bs,
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


class RecordingDecodeWorkspace:
    def __init__(self):
        self.prepare_kwargs = None
        self.run_kwargs = None

    def prepare(self, *args, **kwargs):
        self.prepare_kwargs = kwargs

    def run(self, q, k_cache, v_cache, **kwargs):
        self.run_kwargs = kwargs
        return kwargs["output"].zero_(), torch.empty(0, dtype=q.dtype, device=q.device)


class RecordingGraphWorkspace:
    def __init__(self):
        self.request_indices = torch.empty(8, dtype=torch.int32)
        self.qo_tile_indices = torch.empty(8, dtype=torch.int32)
        self.kv_tile_indices = torch.empty(8, dtype=torch.int32)
        self.merge_indptr = torch.empty(3, dtype=torch.int32)
        self.o_indptr = torch.empty(3, dtype=torch.int32)
        self.kv_chunk_size_ptr = torch.empty(1, dtype=torch.int32)
        self.kv_window_start_tokens = torch.empty(2, dtype=torch.int32)
        self.total_num_rows_ptr = torch.empty(1, dtype=torch.int32)
        self.block_valid_mask = torch.empty(8, dtype=torch.int32)
        self._decode_graph_chunk_pages_lut = torch.ones(4, dtype=torch.int32)
        self._decode_graph_max_chunks_per_req = 4
        self._use_regular_decode_graph_replay = False
        self._decode_graph_metadata_captured_in_graph = False
        self.total_q_capacity = 2
        self.update_calls = 0
        self.fused_update_calls = 0
        self.fused_req_to_token = None
        self.fused_req_pool_indices = None
        self.bound_page_table = None
        self.cache_seqlens = None
        self.cu_seqlens_q = None

    def bind_cuda_graph_runtime_metadata(
        self,
        *,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ):
        self.bound_page_table = page_table
        self.cache_seqlens = cache_seqlens
        self.cu_seqlens_q = cu_seqlens_q

    def update_decode_graph_replay_metadata_from_runtime_cache_seqlens(self):
        self.update_calls += 1

    def update_decode_graph_replay_metadata(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ):
        self.fused_update_calls += 1
        self.fused_req_to_token = req_to_token
        self.fused_req_pool_indices = req_pool_indices


class RecordingReplayBackend:
    def __init__(self):
        self.calls = []

    def init_forward_metadata_replay_cuda_graph_with_cache_loc(self, **kwargs):
        self.calls.append(("with_cache_loc", kwargs))

    def init_forward_metadata_replay_cuda_graph(self, **kwargs):
        self.calls.append(("cpu", kwargs))

    def init_forward_metadata_replay_cuda_graph_no_cpu(self, **kwargs):
        self.calls.append(("no_cpu", kwargs))


class RecordingReplayBackendWithoutCacheLoc:
    def __init__(self):
        self.calls = []

    def init_forward_metadata_replay_cuda_graph(self, **kwargs):
        self.calls.append(("cpu", kwargs))

    def init_forward_metadata_replay_cuda_graph_no_cpu(self, **kwargs):
        self.calls.append(("no_cpu", kwargs))


def make_layer(layer_id: int):
    return SimpleNamespace(
        layer_id=layer_id,
        tp_q_head_num=24,
        tp_k_head_num=4,
        tp_v_head_num=4,
        qk_head_dim=128,
        v_head_dim=128,
        sliding_window_size=-1,
    )


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


def test_decode_graph_capture_boundaries_are_dynamo_disabled():
    from b12x.attention.paged.workspace import PagedAttentionWorkspace

    assert getattr(
        B12xAttnBackend._stage_decode_graph_preprocess_for_layer,
        "_torchdynamo_disable",
        False,
    )
    assert getattr(
        B12xAttnBackend._decode_graph_run_prepare_metadata,
        "_torchdynamo_disable",
        False,
    )
    assert getattr(
        B12xAttnBackend._mark_decode_graph_metadata_captured,
        "_torchdynamo_disable",
        False,
    )
    assert getattr(PagedAttentionWorkspace.prepare, "_torchdynamo_disable", False)


def test_forward_decode_disables_captured_metadata_refresh_for_cuda_graph():
    bs = 2
    backend = object.__new__(B12xAttnBackend)
    backend.forward_metadata = make_metadata(bs)
    backend.use_sliding_window_kv_pool = False
    backend.head_dim_qk = 2
    backend.head_dim_vo = 3
    backend.swa_head_dim_qk = 2
    backend.swa_head_dim_vo = 3
    backend.kv_cache_dtype = torch.bfloat16
    backend.fp8_descale_cache = {}
    backend.cuda_graph_row_indices = torch.arange(bs, dtype=torch.long)

    workspace = RecordingDecodeWorkspace()
    backend._get_workspace = lambda md, total_q, layer, has_sinks: workspace
    backend._get_paged_kv_buffers = lambda token_to_kv_pool, layer_id: (
        torch.empty((1, 1, 1, 2)),
        torch.empty((1, 1, 1, 3)),
    )
    backend._get_descale_tensors = lambda layer, batch_size: (None, None)

    layer = SimpleNamespace(
        tp_q_head_num=1,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=2,
        v_head_dim=3,
        layer_id=0,
        sliding_window_size=-1,
    )
    forward_batch = SimpleNamespace(token_to_kv_pool=None, out_cache_loc=None)
    out = backend.forward_decode(
        torch.randn(bs, 2),
        None,
        None,
        layer,
        forward_batch,
        save_kv_cache=False,
    )

    assert out.shape == (bs, 3)
    assert workspace.prepare_kwargs["active_total_q"] == bs
    assert workspace.run_kwargs["prepare_decode_graph_metadata"] is False


def test_decode_cuda_graph_replay_updates_equivalent_metadata_once():
    backend = object.__new__(B12xAttnBackend)
    backend.attention_layers = [make_layer(0), make_layer(1)]
    backend.has_attention_sinks = True
    backend.use_sliding_window_kv_pool = False
    backend.head_dim_qk = 128
    backend.head_dim_vo = 128
    backend.swa_head_dim_qk = 128
    backend.swa_head_dim_vo = 128
    backend.cuda_graph_workspaces = {}
    backend.cuda_graph_decode_metadata_sources = {}
    backend._num_cache_pages_for_layer = lambda layer: 16

    workspaces = {}

    def get_or_create_workspace(workspace_key, **kwargs):
        return workspaces.setdefault(workspace_key, RecordingGraphWorkspace())

    backend._get_or_create_graph_workspace = get_or_create_workspace

    cache_seqlens = torch.tensor([65, 129], dtype=torch.int32)
    page_table = torch.zeros((2, 4), dtype=torch.int32)
    cu_seqlens_q = torch.arange(0, 3, dtype=torch.int32)
    backend.cuda_graph_cache_seqlens = cache_seqlens
    backend.cuda_graph_page_table = page_table
    backend.cuda_graph_cu_seqlens_q = cu_seqlens_q

    backend._prepare_cuda_graph_workspaces(
        ("decode", 2),
        mode="decode",
        bs=2,
        total_q_capacity=2,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        swa_page_table=None,
        cu_seqlens_q=cu_seqlens_q,
    )

    assert len(workspaces) == 4
    assert sum(workspace.update_calls for workspace in workspaces.values()) == 1

    def fail_if_layer_walk_repeats(*args, **kwargs):
        pytest.fail("decode graph replay prep should use the cached workspace plan")

    backend._get_or_create_graph_workspace = fail_if_layer_walk_repeats
    backend._prepare_cuda_graph_workspaces(
        ("decode", 2),
        mode="decode",
        bs=2,
        total_q_capacity=2,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        swa_page_table=None,
        cu_seqlens_q=cu_seqlens_q,
    )

    assert sum(workspace.update_calls for workspace in workspaces.values()) == 2

    source = next(iter(backend.cuda_graph_decode_metadata_sources.values()))
    for workspace in workspaces.values():
        assert workspace.request_indices is source.request_indices
        assert workspace.kv_tile_indices is source.kv_tile_indices
        assert workspace.merge_indptr is source.merge_indptr


def test_decode_cuda_graph_replay_can_delegate_page_table_update_to_workspace():
    backend = object.__new__(B12xAttnBackend)
    backend.attention_layers = [make_layer(0), make_layer(1)]
    backend.has_attention_sinks = True
    backend.use_sliding_window_kv_pool = False
    backend.head_dim_qk = 128
    backend.head_dim_vo = 128
    backend.swa_head_dim_qk = 128
    backend.swa_head_dim_vo = 128
    backend.cuda_graph_workspaces = {}
    backend.cuda_graph_decode_metadata_sources = {}
    backend.req_to_token = torch.zeros((4, 256), dtype=torch.int32)
    backend._num_cache_pages_for_layer = lambda layer: 16

    workspaces = {}

    def get_or_create_workspace(workspace_key, **kwargs):
        return workspaces.setdefault(workspace_key, RecordingGraphWorkspace())

    backend._get_or_create_graph_workspace = get_or_create_workspace

    cache_seqlens = torch.tensor([65, 129], dtype=torch.int32)
    page_table = torch.zeros((2, 4), dtype=torch.int32)
    cu_seqlens_q = torch.arange(0, 3, dtype=torch.int32)
    req_pool_indices = torch.tensor([1, 2], dtype=torch.int64)
    backend.cuda_graph_cache_seqlens = cache_seqlens
    backend.cuda_graph_page_table = page_table
    backend.cuda_graph_cu_seqlens_q = cu_seqlens_q

    backend._prepare_cuda_graph_workspaces(
        ("decode", 2),
        mode="decode",
        bs=2,
        total_q_capacity=2,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        swa_page_table=None,
        cu_seqlens_q=cu_seqlens_q,
        req_pool_indices=req_pool_indices,
        decode_page_table_prebuilt=False,
    )

    assert len(workspaces) == 4
    assert sum(workspace.update_calls for workspace in workspaces.values()) == 0
    assert sum(workspace.fused_update_calls for workspace in workspaces.values()) == 1

    source = next(iter(backend.cuda_graph_decode_metadata_sources.values()))
    assert source.fused_req_to_token is backend.req_to_token
    assert source.fused_req_pool_indices is req_pool_indices
    for workspace in workspaces.values():
        assert workspace.request_indices is source.request_indices
        assert workspace.kv_tile_indices is source.kv_tile_indices
        assert workspace.merge_indptr is source.merge_indptr


def test_decode_graph_preprocess_can_be_deferred_to_captured_forward():
    backend = object.__new__(B12xAttnBackend)
    backend.attention_layers = [make_layer(0), make_layer(1)]
    backend.has_attention_sinks = True
    backend.use_sliding_window_kv_pool = False
    backend.head_dim_qk = 128
    backend.head_dim_vo = 128
    backend.swa_head_dim_qk = 128
    backend.swa_head_dim_vo = 128
    backend.cuda_graph_workspaces = {}
    backend.cuda_graph_decode_metadata_sources = {}
    backend._num_cache_pages_for_layer = lambda layer: 16

    workspaces = {}

    def get_or_create_workspace(workspace_key, **kwargs):
        return workspaces.setdefault(workspace_key, RecordingGraphWorkspace())

    backend._get_or_create_graph_workspace = get_or_create_workspace

    cache_seqlens = torch.tensor([65, 129], dtype=torch.int32)
    page_table = torch.zeros((2, 4), dtype=torch.int32)
    cu_seqlens_q = torch.arange(0, 3, dtype=torch.int32)
    backend.cuda_graph_cache_seqlens = cache_seqlens
    backend.cuda_graph_page_table = page_table
    backend.cuda_graph_cu_seqlens_q = cu_seqlens_q

    backend._prepare_cuda_graph_workspaces(
        ("decode", 2),
        mode="decode",
        bs=2,
        total_q_capacity=2,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        swa_page_table=None,
        cu_seqlens_q=cu_seqlens_q,
        decode_preprocess_in_graph=True,
        reset_decode_preprocess_capture=True,
    )

    assert len(workspaces) == 4
    assert sum(workspace.update_calls for workspace in workspaces.values()) == 0
    source = next(iter(backend.cuda_graph_decode_metadata_sources.values()))
    source._decode_graph_metadata_captured_in_graph = True

    backend._prepare_cuda_graph_workspaces(
        ("decode", 2),
        mode="decode",
        bs=2,
        total_q_capacity=2,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        swa_page_table=None,
        cu_seqlens_q=cu_seqlens_q,
        decode_preprocess_in_graph=True,
    )

    assert sum(workspace.update_calls for workspace in workspaces.values()) == 0


def test_decode_graph_sink_workspace_can_capture_shared_metadata(monkeypatch):
    backend = object.__new__(B12xAttnBackend)
    backend.device = torch.device("cuda")
    backend.attention_layers = [make_layer(0)]
    backend.has_attention_sinks = True
    backend.use_sliding_window_kv_pool = False
    backend.head_dim_qk = 128
    backend.head_dim_vo = 128
    backend.swa_head_dim_qk = 128
    backend.swa_head_dim_vo = 128
    backend.cuda_graph_workspaces = {}
    backend.cuda_graph_decode_metadata_sources = {}
    backend._num_cache_pages_for_layer = lambda layer: 16

    workspaces = {}

    def get_or_create_workspace(workspace_key, **kwargs):
        return workspaces.setdefault(workspace_key, RecordingGraphWorkspace())

    backend._get_or_create_graph_workspace = get_or_create_workspace

    cache_seqlens = torch.tensor([65, 129], dtype=torch.int32)
    page_table = torch.zeros((2, 4), dtype=torch.int32)
    cu_seqlens_q = torch.arange(0, 3, dtype=torch.int32)
    backend.cuda_graph_cache_seqlens = cache_seqlens
    backend.cuda_graph_page_table = page_table
    backend.cuda_graph_cu_seqlens_q = cu_seqlens_q

    graph_key = ("decode", 2)
    layer = backend.attention_layers[0]
    backend._prepare_cuda_graph_workspaces(
        graph_key,
        mode="decode",
        bs=2,
        total_q_capacity=2,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        swa_page_table=None,
        cu_seqlens_q=cu_seqlens_q,
        decode_preprocess_in_graph=True,
        reset_decode_preprocess_capture=True,
    )

    source = next(iter(backend.cuda_graph_decode_metadata_sources.values()))
    sink_workspace = workspaces[backend._cuda_graph_workspace_key(graph_key, layer, True)]
    metadata = B12xForwardMetadata(
        cu_seqlens_q=cu_seqlens_q,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        swa_page_table=None,
        mode="decode",
        use_cuda_graph=True,
        active_total_q=2,
        graph_key=graph_key,
        decode_preprocess_in_graph=True,
    )

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    prepared = backend._decode_graph_run_prepare_metadata(
        metadata,
        layer,
        sink_workspace,
        window_left=-1,
    )

    assert prepared is True
    backend._mark_decode_graph_metadata_captured(
        metadata,
        layer,
        sink_workspace,
        window_left=-1,
        prepared=prepared,
    )
    assert source._decode_graph_metadata_captured_in_graph is True


def test_b12x_moe_config_aliases_cover_minimax_m2():
    cfg = SimpleNamespace(
        hidden_size=3072,
        intermediate_size=1536,
        num_experts_per_tok=8,
        num_local_experts=256,
    )

    assert (
        _b12x_get_config_attr(
            cfg, ("n_routed_experts", "num_experts", "num_local_experts")
        )
        == 256
    )
    assert (
        _b12x_get_config_attr(cfg, ("moe_intermediate_size", "intermediate_size"))
        == 1536
    )


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


def test_tbo_replay_forwards_cache_loc_metadata_to_children():
    primary = RecordingReplayBackend()
    child_left = RecordingReplayBackend()
    child_right = RecordingReplayBackend()
    backend = TboAttnBackend(primary=primary, children=[child_left, child_right])

    out_cache_loc = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
    backend.init_forward_metadata_replay_cuda_graph_with_cache_loc(
        bs=4,
        req_pool_indices=torch.arange(4, dtype=torch.int64),
        seq_lens=torch.tensor([3, 4, 5, 6], dtype=torch.int32),
        seq_lens_sum=18,
        encoder_lens=None,
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
        seq_lens_cpu=None,
        out_cache_loc=out_cache_loc,
    )

    assert primary.calls[0][0] == "with_cache_loc"
    assert primary.calls[0][1]["out_cache_loc"] is out_cache_loc
    assert child_left.calls[0][0] == "with_cache_loc"
    assert child_left.calls[0][1]["out_cache_loc"].tolist() == [10, 11]
    assert child_left.calls[0][1]["seq_lens_sum"] == 7
    assert child_right.calls[0][0] == "with_cache_loc"
    assert child_right.calls[0][1]["out_cache_loc"].tolist() == [12, 13]
    assert child_right.calls[0][1]["seq_lens_sum"] == 11


def test_tbo_replay_cache_loc_hook_falls_back_for_plain_children():
    primary = RecordingReplayBackend()
    child_left = RecordingReplayBackendWithoutCacheLoc()
    child_right = RecordingReplayBackendWithoutCacheLoc()
    backend = TboAttnBackend(primary=primary, children=[child_left, child_right])

    backend.init_forward_metadata_replay_cuda_graph_with_cache_loc(
        bs=4,
        req_pool_indices=torch.arange(4, dtype=torch.int64),
        seq_lens=torch.tensor([3, 4, 5, 6], dtype=torch.int32),
        seq_lens_sum=18,
        encoder_lens=None,
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
        seq_lens_cpu=None,
        out_cache_loc=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
    )

    assert primary.calls[0][0] == "with_cache_loc"
    assert child_left.calls[0][0] == "no_cpu"
    assert "out_cache_loc" not in child_left.calls[0][1]
    assert child_left.calls[0][1]["seq_lens_sum"] == 7
    assert child_right.calls[0][0] == "no_cpu"
    assert child_right.calls[0][1]["seq_lens_sum"] == 11


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


def test_decode_graph_replay_prebuilds_page_table_without_swa():
    bs = 2
    backend = make_backend(bs=bs)
    backend.use_sliding_window_kv_pool = False
    backend.cuda_graph_cu_seqlens_q = torch.zeros(bs + 1, dtype=torch.int32)
    backend.cuda_graph_cache_seqlens = torch.zeros(bs, dtype=torch.int32)
    backend.cuda_graph_page_table = torch.zeros((bs, 4), dtype=torch.int32)
    backend.cuda_graph_swa_page_table = None
    prepared = {}
    backend._prepare_cuda_graph_workspaces = lambda *args, **kwargs: prepared.update(
        kwargs
    )

    backend.init_forward_metadata_replay_cuda_graph_no_cpu(
        bs=bs,
        req_pool_indices=torch.zeros(bs, dtype=torch.int64),
        seq_lens=torch.tensor([65, 1], dtype=torch.int32),
        seq_lens_sum=66,
        encoder_lens=None,
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
    )

    metadata = backend.forward_metadata
    assert prepared["decode_page_table_prebuilt"] is True
    assert metadata.swa_page_table is None
    assert metadata.page_table[0].tolist() == [0, 0, 2, 3]
