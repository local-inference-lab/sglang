from types import SimpleNamespace

import pytest
from sglang.srt.configs.virtual_tp import (
    VIRTUAL_TP_SHARDING_B12X_PADDED,
    adjust_config_with_b12x_virtual_tp_sharding,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class FakeModelConfig(SimpleNamespace):
    def get_total_num_kv_heads(self):
        return self.num_key_value_heads


def _make_qwen35_config():
    text_config = SimpleNamespace(
        model_type="qwen3_5_text",
        hidden_size=3584,
        num_attention_heads=28,
        num_key_value_heads=4,
        linear_num_key_heads=7,
        linear_num_value_heads=14,
        moe_intermediate_size=513,
    )
    hf_config = SimpleNamespace(
        model_type="qwen3_5",
        architectures=["Qwen3_5ForCausalLM"],
        text_config=text_config,
    )
    return FakeModelConfig(
        hidden_size=3584,
        num_attention_heads=28,
        num_key_value_heads=4,
        hf_config=hf_config,
        hf_text_config=text_config,
    )


def _make_deepseek_v4_config():
    hf_config = SimpleNamespace(
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        num_attention_heads=128,
        num_key_value_heads=1,
        o_groups=16,
        moe_intermediate_size=3072,
        intermediate_size=18432,
    )
    return FakeModelConfig(
        num_attention_heads=128,
        num_key_value_heads=1,
        o_groups=16,
        hf_config=hf_config,
        hf_text_config=None,
    )


def _make_deepseek_v4_flash_config():
    hf_config = SimpleNamespace(
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        num_attention_heads=64,
        num_key_value_heads=1,
        o_groups=8,
        moe_intermediate_size=2048,
        intermediate_size=18432,
    )
    return FakeModelConfig(
        num_attention_heads=64,
        num_key_value_heads=1,
        o_groups=8,
        hf_config=hf_config,
        hf_text_config=None,
    )


def test_b12x_virtual_tp_pads_attention_linear_and_moe_axes():
    model_config = _make_qwen35_config()

    adjust_config_with_b12x_virtual_tp_sharding(
        model_config,
        None,
        attention_tp_size=5,
        moe_tp_size=5,
        uses_b12x_attention=True,
        uses_b12x_moe=True,
    )

    assert model_config.num_attention_heads == 35
    assert model_config.num_key_value_heads == 5
    assert model_config.hf_text_config.head_dim == 128
    assert model_config.hf_text_config.linear_num_key_heads == 10
    assert model_config.hf_text_config.linear_num_value_heads == 20
    assert model_config.hf_text_config.moe_intermediate_size == 640

    plan = model_config.sglang_virtual_shard_plan
    assert plan.policy == VIRTUAL_TP_SHARDING_B12X_PADDED
    assert plan.axis("attention_kv_heads").valid_size(4) == 0
    assert plan.axis("linear_key_heads").valid_size(3) == 1
    assert plan.axis("moe_intermediate_size").local_size == 128

    assert model_config.hf_text_config.original_num_attention_heads == 28
    assert model_config.hf_text_config.original_total_num_kv_heads == 4
    assert model_config.hf_text_config.original_linear_num_key_heads == 7
    assert model_config.hf_text_config.original_moe_intermediate_size == 513


def test_b12x_virtual_tp_preserves_deepseek_v4_heads_per_group():
    model_config = _make_deepseek_v4_config()

    adjust_config_with_b12x_virtual_tp_sharding(
        model_config,
        None,
        attention_tp_size=3,
        moe_tp_size=3,
        uses_b12x_attention=True,
        uses_b12x_moe=True,
    )

    assert model_config.num_attention_heads == 144
    assert model_config.o_groups == 18
    assert model_config.num_key_value_heads == 1
    assert model_config.num_attention_heads // model_config.o_groups == 8
    assert model_config.hf_config.num_attention_heads == 144
    assert model_config.hf_config.o_groups == 18
    assert model_config.hf_config.moe_intermediate_size == 3072

    plan = model_config.sglang_virtual_shard_plan
    assert plan.axis("attention_q_heads").local_size == 48
    assert plan.axis("attention_q_heads").valid_size(2) == 32
    assert plan.axis("dsv4_o_groups").local_size == 6
    assert plan.axis("dsv4_o_groups").valid_size(2) == 4
    assert plan.axis("attention_kv_heads") is None
    assert model_config.hf_config.original_num_attention_heads == 128
    assert model_config.hf_config.original_o_groups == 16


def test_b12x_virtual_tp_pads_deepseek_v4_flash_tp3_shape():
    model_config = _make_deepseek_v4_flash_config()

    adjust_config_with_b12x_virtual_tp_sharding(
        model_config,
        None,
        attention_tp_size=3,
        moe_tp_size=3,
        uses_b12x_attention=True,
        uses_b12x_moe=True,
    )

    assert model_config.num_attention_heads == 72
    assert model_config.o_groups == 9
    assert model_config.num_key_value_heads == 1
    assert model_config.hf_config.moe_intermediate_size == 2304
    assert model_config.hf_config.intermediate_size == 18432

    plan = model_config.sglang_virtual_shard_plan
    assert plan.axis("attention_q_heads").local_size == 24
    assert plan.axis("attention_q_heads").valid_size(2) == 16
    assert plan.axis("dsv4_o_groups").local_size == 3
    assert plan.axis("dsv4_o_groups").valid_size(2) == 2
    assert plan.axis("moe_intermediate_size").local_size == 768
    assert plan.axis("intermediate_size").local_size == 6144


@pytest.mark.parametrize(
    (
        "tp_size",
        "expected_heads",
        "expected_o_groups",
        "expected_moe",
        "expected_dense",
        "last_rank",
    ),
    [
        (9, 144, 18, 3456, 18432, 8),
        (10, 160, 20, 3840, 19200, 9),
    ],
)
def test_b12x_virtual_tp_pads_deepseek_v4_pro_tp9_tp10_default_alignment(
    tp_size,
    expected_heads,
    expected_o_groups,
    expected_moe,
    expected_dense,
    last_rank,
):
    model_config = _make_deepseek_v4_config()

    adjust_config_with_b12x_virtual_tp_sharding(
        model_config,
        None,
        attention_tp_size=tp_size,
        moe_tp_size=tp_size,
        uses_b12x_attention=True,
        uses_b12x_moe=True,
    )

    assert model_config.num_attention_heads == expected_heads
    assert model_config.o_groups == expected_o_groups
    assert model_config.hf_config.moe_intermediate_size == expected_moe
    assert model_config.hf_config.intermediate_size == expected_dense
    assert model_config.num_attention_heads // model_config.o_groups == 8

    plan = model_config.sglang_virtual_shard_plan
    assert plan.moe_local_n_alignment == 128
    assert plan.axis("attention_q_heads").local_size == 16
    assert plan.axis("attention_q_heads").valid_size(last_rank) == 0
    assert plan.axis("dsv4_o_groups").local_size == 2
    assert plan.axis("dsv4_o_groups").valid_size(last_rank) == 0
    assert plan.axis("moe_intermediate_size").alignment == 128
    assert plan.axis("moe_intermediate_size").local_size == 384
    assert plan.axis("intermediate_size").local_size == expected_dense // tp_size


@pytest.mark.parametrize(
    (
        "tp_size",
        "expected_moe",
        "expected_dense",
        "expected_moe_local",
        "expected_dense_local",
    ),
    [
        (9, 3168, 18432, 352, 2048),
        (10, 3200, 18560, 320, 1856),
    ],
)
def test_b12x_virtual_tp_pads_deepseek_v4_pro_with_custom_moe_alignment(
    tp_size,
    expected_moe,
    expected_dense,
    expected_moe_local,
    expected_dense_local,
):
    model_config = _make_deepseek_v4_config()

    adjust_config_with_b12x_virtual_tp_sharding(
        model_config,
        None,
        attention_tp_size=tp_size,
        moe_tp_size=tp_size,
        uses_b12x_attention=True,
        uses_b12x_moe=True,
        moe_local_n_alignment=16,
    )

    assert model_config.hf_config.moe_intermediate_size == expected_moe
    assert model_config.hf_config.intermediate_size == expected_dense

    plan = model_config.sglang_virtual_shard_plan
    assert plan.moe_local_n_alignment == 16
    assert plan.axis("moe_intermediate_size").alignment == 16
    assert plan.axis("moe_intermediate_size").local_size == expected_moe_local
    assert plan.axis("intermediate_size").local_size == expected_dense_local


def test_b12x_virtual_tp_rejects_unsupported_models():
    text_config = SimpleNamespace(model_type="llama", num_attention_heads=32)
    model_config = FakeModelConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        hf_config=text_config,
        hf_text_config=text_config,
    )

    with pytest.raises(ValueError, match="DeepSeek V4, Qwen3.5, and MiniMax M2"):
        adjust_config_with_b12x_virtual_tp_sharding(
            model_config,
            None,
            attention_tp_size=5,
            moe_tp_size=5,
            uses_b12x_attention=True,
            uses_b12x_moe=False,
        )
