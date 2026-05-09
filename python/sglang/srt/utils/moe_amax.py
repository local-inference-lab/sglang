# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import atexit
import logging
import os
import threading
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

_TRACKER: Optional["MoEAmaxTracker"] = None
_AMAX_PROJ_NAMES = ("gate_proj", "up_proj")


def maybe_flush_moe_amax_tracker() -> None:
    if _TRACKER is not None:
        _TRACKER.on_graph_replay()


def prepare_moe_amax_for_graph_capture() -> None:
    if _TRACKER is not None:
        _TRACKER.prepare_for_graph_capture()


def finish_moe_amax_graph_capture() -> None:
    if _TRACKER is not None:
        _TRACKER.finish_graph_capture()


def get_moe_amax_tracker() -> Optional["MoEAmaxTracker"]:
    global _TRACKER

    output_path = os.environ.get("SGLANG_MOE_AMAX_PATH") or os.environ.get(
        "SGLANG_MTP_AMAX_PATH"
    )
    if not output_path:
        return None

    if _TRACKER is None:
        interval = int(
            os.environ.get("SGLANG_MOE_AMAX_INTERVAL")
            or os.environ.get("SGLANG_MTP_AMAX_INTERVAL", "100")
        )
        _TRACKER = MoEAmaxTracker(output_path=output_path, write_interval=interval)
    return _TRACKER


def _get_rank_world_size() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def _rank_path(output_path: str, rank: int, world_size: int) -> str:
    if "{rank}" in output_path or "{world_size}" in output_path:
        return output_path.format(rank=rank, world_size=world_size)
    if world_size <= 1:
        return output_path

    root, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".safetensors"
    return f"{root}.rank{rank}{ext}"


class MoEAmaxTracker:
    def __init__(self, output_path: str, write_interval: int):
        self.rank, self.world_size = _get_rank_world_size()
        self.output_path = _rank_path(output_path, self.rank, self.world_size)
        self.write_interval = max(write_interval, 1)
        self._amax: Dict[str, torch.Tensor] = {}
        self._gpu_amax: Dict[Tuple[str, torch.device], torch.Tensor] = {}
        self._num_updates = 0
        self._lock = threading.Lock()

        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        if os.path.exists(self.output_path):
            self._amax.update(
                {
                    name: value.detach().cpu().to(torch.float32).reshape(())
                    for name, value in load_file(self.output_path).items()
                }
            )
            logger.info(
                "Loaded existing MoE amax checkpoint with %d tensors from %s",
                len(self._amax),
                self.output_path,
            )
        atexit.register(self.flush)
        logger.info(
            "MoE amax tracking enabled: rank=%d/%d output_path=%s write_interval=%d",
            self.rank,
            self.world_size,
            self.output_path,
            self.write_interval,
        )

    @torch.no_grad()
    def update_moe_inputs(
        self,
        *,
        module_prefix: str,
        hidden_states: torch.Tensor,
        topk_ids: Optional[torch.Tensor],
        num_routed_experts: int,
        suppress_host_update: bool = False,
    ) -> None:
        if (
            topk_ids is None
            or hidden_states.numel() == 0
            or topk_ids.numel() == 0
            or num_routed_experts <= 0
        ):
            return

        topk_ids = topk_ids.detach()
        if topk_ids.dim() == 1:
            topk_ids = topk_ids[:, None]
        if topk_ids.shape[0] != hidden_states.shape[0]:
            return

        gpu_amax = self._get_gpu_amax(
            module_prefix, num_routed_experts, hidden_states
        )
        token_amax = hidden_states.detach().abs().amax(dim=-1)
        expanded_amax = token_amax[:, None].expand_as(topk_ids).reshape(-1)
        flat_ids = topk_ids.reshape(-1).to(torch.long)
        valid = (flat_ids >= 0) & (flat_ids < num_routed_experts)
        flat_ids = torch.where(valid, flat_ids, torch.zeros_like(flat_ids))
        expanded_amax = torch.where(
            valid,
            expanded_amax.to(torch.float32),
            torch.zeros_like(expanded_amax, dtype=torch.float32),
        )
        per_expert = torch.zeros(
            num_routed_experts, dtype=torch.float32, device=hidden_states.device
        )
        per_expert.scatter_reduce_(
            0, flat_ids, expanded_amax, reduce="amax", include_self=True
        )
        torch.maximum(gpu_amax, per_expert, out=gpu_amax)

        if suppress_host_update or self._is_cuda_graph_capturing(hidden_states):
            return

        with self._lock:
            self._num_updates += 1
            if self._num_updates % self.write_interval == 0:
                self._flush_locked()

    def on_graph_replay(self) -> None:
        if not self._gpu_amax:
            return
        with self._lock:
            self._num_updates += 1
            if self._num_updates % self.write_interval == 0:
                self._flush_locked()

    def prepare_for_graph_capture(self) -> None:
        with self._lock:
            self._sync_gpu_to_cpu_locked()

    def finish_graph_capture(self) -> None:
        with self._lock:
            for (module_prefix, _), gpu_amax in self._gpu_amax.items():
                self._copy_cpu_module_to_gpu_locked(module_prefix, gpu_amax)

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        self._sync_gpu_to_cpu_locked()
        if not self._amax:
            return
        tmp_path = f"{self.output_path}.tmp"
        save_file(
            {name: value.clone() for name, value in self._amax.items()},
            tmp_path,
            metadata={"format": "pt"},
        )
        os.replace(tmp_path, self.output_path)
        logger.info(
            "Wrote MoE amax checkpoint: rank=%d/%d updates=%d tensors=%d path=%s",
            self.rank,
            self.world_size,
            self._num_updates,
            len(self._amax),
            self.output_path,
        )

    def _get_gpu_amax(
        self,
        module_prefix: str,
        num_routed_experts: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        key = (module_prefix, hidden_states.device)
        gpu_amax = self._gpu_amax.get(key)
        if gpu_amax is None or gpu_amax.numel() != num_routed_experts:
            gpu_amax = torch.zeros(
                num_routed_experts,
                dtype=torch.float32,
                device=hidden_states.device,
            )
            self._copy_cpu_module_to_gpu_locked(module_prefix, gpu_amax)
            self._gpu_amax[key] = gpu_amax
        return gpu_amax

    def _copy_cpu_module_to_gpu_locked(
        self, module_prefix: str, gpu_amax: torch.Tensor
    ) -> None:
        cpu_values = torch.zeros(gpu_amax.numel(), dtype=torch.float32)
        for expert_id in range(gpu_amax.numel()):
            values = []
            for proj_name in _AMAX_PROJ_NAMES:
                value = self._amax.get(
                    self._amax_name(module_prefix, expert_id, proj_name)
                )
                if value is not None:
                    values.append(value.reshape(()))
            if values:
                cpu_values[expert_id] = torch.stack(values).max()
        gpu_amax.copy_(cpu_values.to(device=gpu_amax.device))

    def _sync_gpu_to_cpu_locked(self) -> None:
        for (module_prefix, _), gpu_amax in self._gpu_amax.items():
            if gpu_amax.is_cuda:
                torch.cuda.synchronize(gpu_amax.device)
            cpu_amax = gpu_amax.detach().cpu()
            for expert_id, value in enumerate(cpu_amax):
                value = value.to(torch.float32).reshape(())
                for proj_name in _AMAX_PROJ_NAMES:
                    name = self._amax_name(module_prefix, expert_id, proj_name)
                    current = self._amax.get(name)
                    self._amax[name] = (
                        value if current is None else torch.maximum(current, value)
                    )

    @staticmethod
    def _amax_name(module_prefix: str, expert_id: int, proj_name: str) -> str:
        return f"{module_prefix}.experts.{expert_id}.{proj_name}.input_quantizer"

    @staticmethod
    def _is_cuda_graph_capturing(tensor: torch.Tensor) -> bool:
        return tensor.is_cuda and torch.cuda.is_current_stream_capturing()
