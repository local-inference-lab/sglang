import pytest
import torch
from sglang.srt.models.deepseek_v4 import MQALayer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


def test_attn_sink_loader_zero_pads_virtual_tail():
    param = torch.nn.Parameter(torch.empty(72, dtype=torch.float32))
    loaded = torch.arange(64, dtype=torch.float32)

    MQALayer._load_attn_sink(param, loaded)

    torch.testing.assert_close(param[:64], loaded)
    torch.testing.assert_close(param[64:], torch.zeros(8))


@pytest.mark.parametrize("virtual_heads", [144, 160])
def test_attn_sink_loader_zero_pads_deepseek_v4_pro_virtual_tail(virtual_heads):
    param = torch.nn.Parameter(torch.empty(virtual_heads, dtype=torch.float32))
    loaded = torch.arange(128, dtype=torch.float32)

    MQALayer._load_attn_sink(param, loaded)

    torch.testing.assert_close(param[:128], loaded)
    torch.testing.assert_close(param[128:], torch.zeros(virtual_heads - 128))
