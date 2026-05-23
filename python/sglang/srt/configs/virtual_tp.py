from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

from sglang.srt.configs.update_config import resolve_head_dim, update_config

logger = logging.getLogger(__name__)

VIRTUAL_TP_SHARDING_OFF = "off"
VIRTUAL_TP_SHARDING_B12X_PADDED = "b12x-padded"
VIRTUAL_TP_SHARDING_CHOICES = (
    VIRTUAL_TP_SHARDING_OFF,
    VIRTUAL_TP_SHARDING_B12X_PADDED,
)

B12X_MOE_LOCAL_N_ALIGNMENT = 128

_B12X_VIRTUAL_MODEL_TYPES = {
    "deepseek_v4",
    "minimax_m2",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_5_text",
    "qwen3_5_moe_text",
}
_B12X_VIRTUAL_ARCHITECTURES = {
    "DeepseekV4ForCausalLM",
    "DeepseekV4ForCausalLMNextN",
    "MiniMaxM2ForCausalLM",
    "Qwen3_5ForCausalLM",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
}
_DEEPSEEK_V4_MODEL_TYPES = {"deepseek_v4"}
_DEEPSEEK_V4_ARCHITECTURES = {
    "DeepseekV4ForCausalLM",
    "DeepseekV4ForCausalLMNextN",
}


@dataclass(frozen=True)
class VirtualAxis:
    name: str
    original_size: int
    virtual_size: int
    tp_size: int
    alignment: int = 1

    @property
    def local_size(self) -> int:
        return self.virtual_size // self.tp_size

    @property
    def is_padded(self) -> bool:
        return self.virtual_size != self.original_size

    def valid_range(self, rank: int) -> tuple[int, int]:
        start = min(rank * self.local_size, self.original_size)
        end = min(start + self.local_size, self.original_size)
        return start, end

    def valid_size(self, rank: int) -> int:
        start, end = self.valid_range(rank)
        return end - start


@dataclass
class VirtualShardPlan:
    policy: str
    attention_tp_size: int
    moe_tp_size: int
    moe_local_n_alignment: int = B12X_MOE_LOCAL_N_ALIGNMENT
    axes: Dict[str, VirtualAxis] = field(default_factory=dict)

    @property
    def is_padded(self) -> bool:
        return any(axis.is_padded for axis in self.axes.values())

    def add_axis(
        self,
        name: str,
        original_size: int,
        virtual_size: int,
        tp_size: int,
        alignment: int = 1,
    ) -> None:
        self.axes[name] = VirtualAxis(
            name=name,
            original_size=original_size,
            virtual_size=virtual_size,
            tp_size=tp_size,
            alignment=alignment,
        )

    def axis(self, name: str) -> Optional[VirtualAxis]:
        return self.axes.get(name)


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _iter_unique_configs(model_config: Any) -> Iterable[Any]:
    seen: set[int] = set()
    candidates = [
        model_config,
        getattr(model_config, "hf_config", None),
        getattr(model_config, "hf_text_config", None),
    ]
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is not None:
        candidates.append(getattr(hf_config, "text_config", None))
    for config in candidates:
        if config is None or id(config) in seen:
            continue
        seen.add(id(config))
        yield config


def _get_model_types_and_architectures(model_config: Any) -> tuple[set[str], set[str]]:
    model_types: set[str] = set()
    architectures: set[str] = set()
    for config in _iter_unique_configs(model_config):
        model_type = getattr(config, "model_type", None)
        if model_type is not None:
            model_types.add(str(model_type))
        config_archs = getattr(config, "architectures", None)
        if config_archs is not None:
            architectures.update(str(arch) for arch in config_archs)
    return model_types, architectures


def _update_existing_configs(model_config: Any, attr_name: str, new_value: Any) -> None:
    for config in _iter_unique_configs(model_config):
        if config is model_config or hasattr(config, attr_name):
            update_config(config, attr_name, new_value)


def _update_all_configs(model_config: Any, attr_name: str, new_value: Any) -> None:
    for config in _iter_unique_configs(model_config):
        update_config(config, attr_name, new_value)


def _set_virtual_plan(model_config: Any, plan: VirtualShardPlan) -> None:
    for config in _iter_unique_configs(model_config):
        update_config(config, "sglang_virtual_shard_plan", plan)


def _model_is_supported(model_config: Any) -> bool:
    model_types, architectures = _get_model_types_and_architectures(model_config)
    return bool(
        model_types & _B12X_VIRTUAL_MODEL_TYPES
        or architectures & _B12X_VIRTUAL_ARCHITECTURES
    )


def _model_is_deepseek_v4(model_config: Any) -> bool:
    model_types, architectures = _get_model_types_and_architectures(model_config)
    return bool(
        model_types & _DEEPSEEK_V4_MODEL_TYPES
        or architectures & _DEEPSEEK_V4_ARCHITECTURES
    )


def _pad_attention_heads(model_config: Any, plan: VirtualShardPlan) -> None:
    total_q_heads = model_config.num_attention_heads
    total_kv_heads = model_config.get_total_num_kv_heads()
    if total_q_heads % total_kv_heads != 0:
        raise ValueError(
            "b12x virtual TP padding requires query heads to be an integer "
            f"multiple of KV heads, got q={total_q_heads}, kv={total_kv_heads}."
        )

    head_dim = resolve_head_dim(model_config, total_q_heads, True)
    if head_dim is not None:
        _update_all_configs(model_config, "head_dim", head_dim)
    if hasattr(model_config.hf_config, "qk_nope_head_dim") and hasattr(
        model_config.hf_config, "qk_rope_head_dim"
    ):
        update_config(
            model_config.hf_config,
            "qk_head_dim",
            model_config.hf_config.qk_nope_head_dim
            + model_config.hf_config.qk_rope_head_dim,
        )

    for config in _iter_unique_configs(model_config):
        update_config(config, "original_num_attention_heads", total_q_heads)
        update_config(config, "original_total_num_kv_heads", total_kv_heads)
        update_config(config, "original_num_key_value_heads", total_kv_heads)

    query_heads_per_kv = total_q_heads // total_kv_heads
    virtual_kv_heads = _round_up(total_kv_heads, plan.attention_tp_size)
    virtual_q_heads = virtual_kv_heads * query_heads_per_kv

    plan.add_axis(
        "attention_q_heads",
        total_q_heads,
        virtual_q_heads,
        plan.attention_tp_size,
    )
    plan.add_axis(
        "attention_kv_heads",
        total_kv_heads,
        virtual_kv_heads,
        plan.attention_tp_size,
    )

    if virtual_q_heads != total_q_heads or virtual_kv_heads != total_kv_heads:
        _update_existing_configs(model_config, "num_attention_heads", virtual_q_heads)
        _update_existing_configs(model_config, "num_key_value_heads", virtual_kv_heads)


def _pad_deepseek_v4_attention(model_config: Any, plan: VirtualShardPlan) -> None:
    total_q_heads = model_config.num_attention_heads
    total_kv_heads = model_config.get_total_num_kv_heads()

    group_config = None
    for config in _iter_unique_configs(model_config):
        if hasattr(config, "o_groups"):
            group_config = config
            break
    if group_config is None:
        raise ValueError("DeepSeek V4 virtual TP padding requires config.o_groups.")

    total_o_groups = int(group_config.o_groups)
    if total_kv_heads != 1:
        raise ValueError(
            "DeepSeek V4 virtual TP padding expects one KV head, got "
            f"{total_kv_heads}."
        )
    if total_q_heads % total_o_groups != 0:
        raise ValueError(
            "DeepSeek V4 virtual TP padding requires query heads to be an "
            f"integer multiple of o_groups, got q={total_q_heads}, "
            f"o_groups={total_o_groups}."
        )

    heads_per_group = total_q_heads // total_o_groups
    virtual_o_groups = _round_up(total_o_groups, plan.attention_tp_size)
    virtual_q_heads = virtual_o_groups * heads_per_group

    plan.add_axis(
        "attention_q_heads",
        total_q_heads,
        virtual_q_heads,
        plan.attention_tp_size,
    )
    plan.add_axis(
        "dsv4_o_groups",
        total_o_groups,
        virtual_o_groups,
        plan.attention_tp_size,
    )

    for config in _iter_unique_configs(model_config):
        update_config(config, "original_num_attention_heads", total_q_heads)
        update_config(config, "original_total_num_kv_heads", total_kv_heads)
        update_config(config, "original_num_key_value_heads", total_kv_heads)
        if hasattr(config, "o_groups"):
            update_config(config, "original_o_groups", total_o_groups)

    if virtual_q_heads != total_q_heads:
        _update_existing_configs(model_config, "num_attention_heads", virtual_q_heads)
    if virtual_o_groups != total_o_groups:
        _update_existing_configs(model_config, "o_groups", virtual_o_groups)


def _pad_linear_attention_heads(model_config: Any, plan: VirtualShardPlan) -> None:
    linear_config = None
    for config in _iter_unique_configs(model_config):
        if hasattr(config, "linear_num_key_heads") and hasattr(
            config, "linear_num_value_heads"
        ):
            linear_config = config
            break
    if linear_config is None:
        return

    total_key_heads = linear_config.linear_num_key_heads
    total_value_heads = linear_config.linear_num_value_heads
    if total_value_heads % total_key_heads != 0:
        raise ValueError(
            "b12x virtual TP padding requires linear value heads to be an "
            f"integer multiple of key heads, got key={total_key_heads}, "
            f"value={total_value_heads}."
        )

    virtual_key_heads = _round_up(total_key_heads, plan.attention_tp_size)
    virtual_value_heads = virtual_key_heads * total_value_heads // total_key_heads

    plan.add_axis(
        "linear_key_heads",
        total_key_heads,
        virtual_key_heads,
        plan.attention_tp_size,
    )
    plan.add_axis(
        "linear_value_heads",
        total_value_heads,
        virtual_value_heads,
        plan.attention_tp_size,
    )

    for config in _iter_unique_configs(model_config):
        if hasattr(config, "linear_num_key_heads"):
            update_config(config, "original_linear_num_key_heads", total_key_heads)
            update_config(config, "linear_num_key_heads", virtual_key_heads)
        if hasattr(config, "linear_num_value_heads"):
            update_config(config, "original_linear_num_value_heads", total_value_heads)
            update_config(config, "linear_num_value_heads", virtual_value_heads)


def _pad_intermediate_attrs(model_config: Any, plan: VirtualShardPlan) -> None:
    alignment = plan.moe_local_n_alignment
    for attr_name in (
        "moe_intermediate_size",
        "intermediate_size",
        "intermediate_size_mlp",
        "shared_expert_intermediate_size",
    ):
        original_size = None
        for config in _iter_unique_configs(model_config):
            if hasattr(config, attr_name):
                original_size = getattr(config, attr_name)
                break
        if original_size is None:
            continue

        virtual_local_size = _round_up(
            _ceil_div(original_size, plan.moe_tp_size), alignment
        )
        virtual_size = virtual_local_size * plan.moe_tp_size
        plan.add_axis(
            attr_name,
            original_size,
            virtual_size,
            plan.moe_tp_size,
            alignment,
        )

        for config in _iter_unique_configs(model_config):
            if hasattr(config, attr_name):
                update_config(config, f"original_{attr_name}", original_size)
                update_config(config, attr_name, virtual_size)


def adjust_config_with_b12x_virtual_tp_sharding(
    model_config: Any,
    _load_config: Any,
    attention_tp_size: int,
    moe_tp_size: int,
    *,
    uses_b12x_attention: bool,
    uses_b12x_moe: bool,
    moe_local_n_alignment: int = B12X_MOE_LOCAL_N_ALIGNMENT,
) -> Any:
    if attention_tp_size < 1 or moe_tp_size < 1:
        raise ValueError(
            "b12x virtual TP padding requires positive attention and MoE TP sizes, "
            f"got attention_tp_size={attention_tp_size}, moe_tp_size={moe_tp_size}."
        )
    if moe_local_n_alignment < 1:
        raise ValueError(
            "b12x virtual TP padding requires a positive MoE local-N alignment, "
            f"got moe_local_n_alignment={moe_local_n_alignment}."
        )
    if not uses_b12x_attention and not uses_b12x_moe:
        raise ValueError(
            "--virtual-tp-sharding=b12x-padded is only supported when a b12x "
            "attention-like or MoE backend is selected."
        )
    if not _model_is_supported(model_config):
        model_type = getattr(
            getattr(model_config, "hf_config", None), "model_type", None
        )
        architectures = getattr(
            getattr(model_config, "hf_config", None), "architectures", None
        )
        raise ValueError(
            "--virtual-tp-sharding=b12x-padded currently supports DeepSeek V4, "
            f"Qwen3.5, and MiniMax M2 only, got model_type={model_type!r}, "
            f"architectures={architectures!r}."
        )

    plan = VirtualShardPlan(
        policy=VIRTUAL_TP_SHARDING_B12X_PADDED,
        attention_tp_size=attention_tp_size,
        moe_tp_size=moe_tp_size,
        moe_local_n_alignment=moe_local_n_alignment,
    )

    if uses_b12x_attention:
        if _model_is_deepseek_v4(model_config):
            _pad_deepseek_v4_attention(model_config, plan)
        else:
            _pad_attention_heads(model_config, plan)
            _pad_linear_attention_heads(model_config, plan)
    if uses_b12x_moe:
        _pad_intermediate_attrs(model_config, plan)

    _set_virtual_plan(model_config, plan)
    if plan.is_padded:
        logger.info(
            "Enabled b12x virtual TP padding with axes: %s",
            {
                name: {
                    "original": axis.original_size,
                    "virtual": axis.virtual_size,
                    "tp": axis.tp_size,
                    "local": axis.local_size,
                    "alignment": axis.alignment,
                }
                for name, axis in plan.axes.items()
            },
        )
    return model_config
