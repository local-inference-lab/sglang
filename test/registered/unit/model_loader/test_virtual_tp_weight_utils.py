import sglang.srt.layers.parameter as parameter
import sglang.srt.model_loader.weight_utils as weight_utils
import torch
from sglang.srt.layers.parameter import ModelWeightParameter
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def test_sharded_weight_loader_zero_pads_out_of_bounds_tail(monkeypatch):
    monkeypatch.setattr(weight_utils, "get_attention_tp_rank", lambda: 2)

    param = torch.nn.Parameter(torch.empty(3))
    loaded_weight = torch.arange(5, dtype=torch.float32)

    loader = weight_utils.sharded_weight_loader(0)
    loader(param, loaded_weight)

    torch.testing.assert_close(param, torch.zeros(3))


def test_sharded_weight_loader_copies_partial_tail_and_zero_pads(monkeypatch):
    monkeypatch.setattr(weight_utils, "get_attention_tp_rank", lambda: 1)

    param = torch.nn.Parameter(torch.empty(3))
    loaded_weight = torch.arange(5, dtype=torch.float32)

    loader = weight_utils.sharded_weight_loader(0)
    loader(param, loaded_weight)

    torch.testing.assert_close(param, torch.tensor([3.0, 4.0, 0.0]))


def test_column_parameter_loader_zero_pads_out_of_bounds_tail(monkeypatch):
    monkeypatch.setattr(parameter, "_is_cpu", False)
    param = ModelWeightParameter(
        data=torch.empty(3, 4),
        input_dim=1,
        output_dim=1,
        weight_loader=lambda *args, **kwargs: None,
    )
    loaded = torch.arange(30, dtype=torch.float32).reshape(3, 10)

    param.load_column_parallel_weight(loaded, tp_rank=2)

    expected = torch.zeros(3, 4)
    expected[:, :2] = loaded[:, 8:10]
    torch.testing.assert_close(param.data, expected)


def test_column_parameter_loader_zero_pads_full_out_of_bounds_shard(monkeypatch):
    monkeypatch.setattr(parameter, "_is_cpu", False)
    param = ModelWeightParameter(
        data=torch.empty(3, 4),
        input_dim=1,
        output_dim=1,
        weight_loader=lambda *args, **kwargs: None,
    )
    loaded = torch.arange(24, dtype=torch.float32).reshape(3, 8)

    param.load_column_parallel_weight(loaded, tp_rank=2)

    torch.testing.assert_close(param.data, torch.zeros(3, 4))


def test_qkv_parameter_loader_zero_pads_out_of_bounds_tail(monkeypatch):
    monkeypatch.setattr(parameter, "_is_cpu", False)
    param = ModelWeightParameter(
        data=torch.empty(3, 4),
        input_dim=1,
        output_dim=1,
        weight_loader=lambda *args, **kwargs: None,
    )
    loaded = torch.arange(30, dtype=torch.float32).reshape(3, 10)

    param.load_qkv_weight(
        loaded,
        tp_rank=2,
        shard_id="q",
        shard_offset=0,
        shard_size=4,
        num_heads=1,
    )

    expected = torch.zeros(3, 4)
    expected[:, :2] = loaded[:, 8:10]
    torch.testing.assert_close(param.data, expected)


def test_qkv_parameter_loader_zero_pads_full_out_of_bounds_shard(monkeypatch):
    monkeypatch.setattr(parameter, "_is_cpu", False)
    param = ModelWeightParameter(
        data=torch.empty(3, 4),
        input_dim=1,
        output_dim=1,
        weight_loader=lambda *args, **kwargs: None,
    )
    loaded = torch.arange(24, dtype=torch.float32).reshape(3, 8)

    param.load_qkv_weight(
        loaded,
        tp_rank=3,
        shard_id="q",
        shard_offset=0,
        shard_size=4,
        num_heads=1,
    )

    torch.testing.assert_close(param.data, torch.zeros(3, 4))
