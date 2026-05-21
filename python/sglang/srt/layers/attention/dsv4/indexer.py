from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.jit_kernel.deepseek_v4 import fused_rope
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.layers.attention.dsv4.compressor import Compressor
from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation
from sglang.srt.layers.attention.nsa.triton_kernel import act_quant
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer
from sglang.srt.utils import add_prefix, is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.srt.layers.attention.dsv4.compressor import (
        CompressorBackendMixin,
    )
    from sglang.srt.layers.quantization import QuantizationConfig
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


if is_hip():
    FP8_DTYPE = torch.float8_e4m3fnuz
    FP8_MAX = torch.finfo(FP8_DTYPE).max
else:
    FP8_DTYPE = torch.float8_e4m3fn
    FP8_MAX = torch.finfo(FP8_DTYPE).max


def fp8_paged_mqa_logits_torch(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    _ = metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]

    assert head_dim == 128, "torch reference impl hardcodes DSV4 indexer head_dim=128"
    assert block_size == 64, "torch reference impl hardcodes block_size=64 cache layout"
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    logits = page_table.new_empty((batch_size, max_seq_len), dtype=torch.float32)
    for i in range(batch_size):
        q = q_fp8[i, 0]
        q = q.to(torch.float32)
        q_scale = weight[i]
        seq_len = int(seq_lens[i].item())
        assert seq_len <= max_seq_len
        num_pages = (seq_len + block_size - 1) // block_size
        padded_seq_len = num_pages * block_size
        pages = page_table[i, :num_pages]
        kvcache_fp8 = kvcache_fp8.view(-1, block_size * (head_dim + 4))
        kvcache = kvcache_fp8[pages]
        SCALE_OFFSET = block_size * head_dim
        kvcache_value = kvcache[..., :SCALE_OFFSET].view(dtype=FP8_DTYPE)
        kvcache_scale = kvcache[..., SCALE_OFFSET:].view(dtype=torch.float32)
        kvcache_value = kvcache_value.to(torch.float32)
        kvcache_scale = kvcache_scale.contiguous()
        kvcache_value = kvcache_value.view(padded_seq_len, head_dim)
        kvcache_scale = kvcache_scale.view(padded_seq_len)
        score = F.linear(kvcache_value, q)
        score = F.relu(score)
        score *= q_scale[None, :]
        score = score.sum(dim=1)
        score *= kvcache_scale
        logits[i, :seq_len] = score[:seq_len]

    return logits


@triton.jit
def _fused_scale_kernel(
    weight_ptr,
    q_scale_ptr,
    out_ptr,
    numel,
    out_scale,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    w = tl.load(weight_ptr + offs, mask=mask)
    qs = tl.load(q_scale_ptr + offs, mask=mask)

    acc = w.to(tl.float32) * out_scale * qs.to(tl.float32)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def fused_scale(
    weight: torch.Tensor,
    out_scale: float,
    q_scale: torch.Tensor,
) -> torch.Tensor:
    assert weight.is_contiguous() and q_scale.is_contiguous()
    B, H = weight.shape
    numel = B * H
    out_dtype = torch.promote_types(weight.dtype, q_scale.dtype)
    out = torch.empty((B, H, 1), device=weight.device, dtype=out_dtype)
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _fused_scale_kernel[grid](
        weight,
        q_scale,
        out,
        numel,
        out_scale,
        BLOCK=BLOCK,
    )
    return out


class C4IndexerBackendMixin:
    def __init__(self):
        super().__init__()
        self.debug_use_external_c4_sparse_indices: bool = False

    def _forward_prepare_multi_stream(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(self, CompressorBackendMixin)

        assert alt_streams is not None
        assert len(alt_streams) >= 2
        current_stream = torch.cuda.current_stream()
        stream_q = alt_streams[0]
        stream_weights = alt_streams[1]

        stream_q.wait_stream(current_stream)
        stream_weights.wait_stream(current_stream)

        self.forward_indexer_compressor(
            x=x,
            forward_batch=forward_batch,
            layer_id=c4_indexer.layer_id,
            compressor=c4_indexer.compressor,
        )
        c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=c4_indexer.layer_id,
        )

        with torch.cuda.stream(stream_q):
            if q_lora_ready is not None:
                stream_q.wait_event(q_lora_ready)
            q = c4_indexer.compute_q(q_lora, positions=positions)
            q_fp8, q_scale = act_quant(q)
            q_scale_ready = stream_q.record_event()

        with torch.cuda.stream(stream_weights):
            weights = c4_indexer.compute_weights(x, skip_scale=True)
            stream_weights.wait_event(q_scale_ready)
            weights = fused_scale(weights, c4_indexer.weight_scale, q_scale)

        current_stream.wait_stream(stream_q)
        current_stream.wait_stream(stream_weights)

        return q_fp8, weights, c4_indexer_kv_cache

    def _forward_prepare_normal(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(self, CompressorBackendMixin)

        q = c4_indexer.compute_q(q_lora, positions=positions)
        q_fp8, q_scale = act_quant(q)
        weights = c4_indexer.compute_weights(x, skip_scale=True)
        weights = fused_scale(weights, c4_indexer.weight_scale, q_scale)
        self.forward_indexer_compressor(
            x=x,
            forward_batch=forward_batch,
            layer_id=c4_indexer.layer_id,
            compressor=c4_indexer.compressor,
        )
        c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=c4_indexer.layer_id,
        )
        return q_fp8, weights, c4_indexer_kv_cache

    def forward_c4_indexer(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        forward_batch: ForwardBatch,
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        enable_multi_stream: bool = False,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        # PREP_IN_CG lazy upgrade: this runs from MQALayer._forward_prepare,
        # before attn_backend.forward() would trigger the upgrade.
        self._maybe_upgrade_forward_metadata()
        token_to_kv_pool = forward_batch.token_to_kv_pool

        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
            assert isinstance(self, CompressorBackendMixin)

        metadata = self.forward_metadata
        indexer_metadata = metadata.indexer_metadata
        core_metadata = metadata.core_metadata

        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DSV4AttnMetadata,
        )

        assert isinstance(core_metadata, DSV4AttnMetadata)
        assert isinstance(indexer_metadata, PagedIndexerMetadata)

        if enable_multi_stream:
            q_fp8, weights, c4_indexer_kv_cache = self._forward_prepare_multi_stream(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=core_metadata.positions,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
                alt_streams=alt_streams,
                q_lora_ready=q_lora_ready,
            )
        else:
            assert q_lora_ready is None
            q_fp8, weights, c4_indexer_kv_cache = self._forward_prepare_normal(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=core_metadata.positions,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
            )

        assert len(q_fp8.shape) == 3
        assert len(c4_indexer_kv_cache.shape) == 2

        assert indexer_metadata.page_table is core_metadata.page_table
        if self.debug_use_external_c4_sparse_indices:
            return

        indexer_capturer = get_global_indexer_capturer()
        capture_enabled = indexer_capturer is not None

        hisparse_coordinator = forward_batch.hisparse_coordinator
        hisparse_decode = (
            hisparse_coordinator is not None and forward_batch.forward_mode.is_decode()
        )

        selected_raw_indices = core_metadata.c4_sparse_page_indices
        if hisparse_decode:
            selected_raw_indices = hisparse_coordinator.raw_indices_buffer[
                : core_metadata.c4_sparse_page_indices.size(0)
            ]

        from b12x.integration.paged_mqa_indexer import (
            paged_mqa_index_decode_dense_topk_fp8,
            paged_mqa_index_decode_supertile_topk_fp8,
        )

        use_dense_decode = forward_batch.forward_mode.is_decode_or_idle()
        workspace = self.get_b12x_indexer_paged_workspace(
            forward_batch=forward_batch,
            q_rows=q_fp8.shape[0],
            page_table_width=indexer_metadata.page_table.shape[1],
            logits_mode="dense" if use_dense_decode else "tiled",
        )
        if use_dense_decode:
            paged_mqa_index_decode_dense_topk_fp8(
                q_fp8=q_fp8,
                weights=weights,
                index_k_cache=c4_indexer_kv_cache,
                metadata=indexer_metadata.b12x_metadata,
                page_size=indexer_metadata.c4_page_size,
                topk=core_metadata.c4_sparse_page_indices.shape[1],
                expected_num_q_heads=indexer_metadata.expected_num_q_heads,
                workspace=workspace,
                out_indices=selected_raw_indices,
            )
        else:
            workspace_q_rows = int(
                getattr(workspace, "max_paged_q_rows", q_fp8.shape[0])
            )
            supertile_k = self._b12x_c4_indexer_supertile_tokens_capacity(
                page_table_width=self._b12x_indexer_page_table_width_capacity(),
                q_rows=workspace_q_rows,
            )
            paged_mqa_index_decode_supertile_topk_fp8(
                q_fp8=q_fp8,
                weights=weights,
                index_k_cache=c4_indexer_kv_cache,
                metadata=indexer_metadata.b12x_metadata,
                page_size=indexer_metadata.c4_page_size,
                topk=core_metadata.c4_sparse_page_indices.shape[1],
                expected_num_q_heads=indexer_metadata.expected_num_q_heads,
                workspace=workspace,
                out_indices=selected_raw_indices,
                supertile_k=supertile_k,
            )
        if hisparse_coordinator is not None:
            if hisparse_decode:
                compress_layer_id = token_to_kv_pool.layer_mapping[
                    c4_indexer.layer_id
                ].compress_layer_id
                core_metadata.c4_sparse_page_indices = (
                    hisparse_coordinator.swap_in_selected_pages(
                        req_pool_indices=forward_batch.req_pool_indices,
                        compressed_seq_lens=indexer_metadata.c4_seq_lens,
                        top_k_result=selected_raw_indices,
                        layer_id=compress_layer_id,
                    )
                )
            else:
                core_metadata.c4_sparse_page_indices = (
                    token_to_kv_pool.c4_kv_pool.translate_loc_to_hisparse_device(
                        core_metadata.c4_sparse_page_indices
                    )
                )

        if capture_enabled:
            compress_layer_id = token_to_kv_pool.layer_mapping[
                c4_indexer.layer_id
            ].compress_layer_id
            indexer_capturer.capture(compress_layer_id, selected_raw_indices)


class C4Indexer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        freqs_cis: torch.Tensor,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        # The C4 selector is replicated across attention TP ranks, matching the
        # GLM/NSA indexer contract. Attention heads are sharded separately.
        self.n_local_heads = self.n_heads
        self.local_head_start = 0
        self.local_head_end = self.n_heads
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("wq_b", prefix),
        )
        self.weights_proj = ReplicatedLinear(
            self.dim,
            self.n_heads,
            bias=False,
            quant_config=None,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.compressor = Compressor(
            config,
            self.layer_id,
            True,
            freqs_cis,
            compress_ratio=4,
            head_dim=self.head_dim,
            rotate=True,
            prefix=add_prefix("compressor", prefix),
        )
        self.freqs_cis = freqs_cis
        self.weight_scale: float = self.softmax_scale * self.n_heads**-0.5
        self.alt_streams = alt_streams

    def compute_q(self, q_lora: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        q, _ = self.wq_b(q_lora)
        q = q.view(-1, self.n_heads, self.head_dim)
        fused_rope(
            q[..., -self.rope_head_dim :],
            None,
            self.freqs_cis,
            positions=positions,
        )
        q = rotate_activation(q)
        q = q.contiguous()
        return q

    def compute_weights(self, x: torch.Tensor, skip_scale=False) -> torch.Tensor:
        out, _ = self.weights_proj(x)
        out = out.contiguous()
        if not skip_scale:
            out = out * self.weight_scale
        return out

    def forward(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        forward_batch: ForwardBatch,
        enable_multi_stream: bool = False,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> None:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.attn_backend, DeepseekV4AttnBackend)
        return forward_batch.attn_backend.forward_c4_indexer(
            x=x,
            q_lora=q_lora,
            forward_batch=forward_batch,
            c4_indexer=self,
            alt_streams=self.alt_streams,
            enable_multi_stream=enable_multi_stream,
            q_lora_ready=q_lora_ready,
        )
