import importlib.util
import sys

import pytest
import torch

from sglang.srt.layers.quantization.modelopt_quant import _b12x_fp4_gemm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-kernel-unit-1-gpu-b200")

try:
    from flashinfer import fp4_quantize
except Exception:
    fp4_quantize = None

try:
    from b12x.gemm.dense import dense_gemm
    from b12x.quant.expert_fp4 import _as_grouped_scale_view
except Exception:
    dense_gemm = None
    _as_grouped_scale_view = None


def _b12x_sm120_supported() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (12, 0)
        and fp4_quantize is not None
        and dense_gemm is not None
        and _as_grouped_scale_view is not None
        and importlib.util.find_spec("b12x") is not None
    )


def _make_global_scale(x: torch.Tensor) -> torch.Tensor:
    max_abs = torch.amax(x.abs()).clamp_min_(1e-6)
    return (torch.finfo(torch.float8_e4m3fn).max * 6.0 / max_abs).to(torch.float32)


pytestmark = pytest.mark.skipif(
    not _b12x_sm120_supported(),
    reason="b12x FP4 GEMM coverage requires SM120, flashinfer, and b12x",
)


def _legacy_dense_expected(
    packed_input: torch.Tensor,
    packed_weight: torch.Tensor,
    input_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    m_orig = packed_input.shape[0]
    k = packed_input.shape[1] * 2
    n_orig = packed_weight.shape[0]

    def _pad128(x, dim):
        size = x.shape[dim]
        pad_size = ((size + 127) // 128) * 128 - size
        if pad_size == 0:
            return x
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_size
        return torch.cat(
            [x, torch.zeros(pad_shape, dtype=x.dtype, device=x.device)], dim=dim
        )

    m = ((m_orig + 127) // 128) * 128
    n = ((n_orig + 127) // 128) * 128

    a_padded = _pad128(packed_input, 0).unsqueeze(2)
    b_padded = _pad128(packed_weight, 0).unsqueeze(2)

    def _sf_to_6d(sf, rows_padded, cols):
        sf_u8 = sf.contiguous().view(torch.uint8)
        cols_sf_padded = sf_u8.numel() // rows_padded
        return _as_grouped_scale_view(
            sf_u8.reshape(1, rows_padded, cols_sf_padded), rows_padded, cols
        )

    out = dense_gemm(
        (a_padded.view(torch.float4_e2m1fn_x2), _sf_to_6d(input_sf, m, k)),
        (b_padded.view(torch.float4_e2m1fn_x2), _sf_to_6d(weight_sf, n, k)),
        alpha=alpha.view(1),
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype="bfloat16" if out_dtype == torch.bfloat16 else "float16",
        sf_vec_size=16,
    )
    return out[:m_orig, :n_orig, 0]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [1, 2, 8])
def test_b12x_fp4_gemm_matches_legacy_dense_path(
    dtype: torch.dtype,
    m: int,
) -> None:
    torch.manual_seed(20260418 + m)

    n = 256
    k = 128
    activation = torch.randn((m, k), device="cuda", dtype=dtype) / 4
    weight = torch.randn((n, k), device="cuda", dtype=dtype) / 4

    input_global_scale = _make_global_scale(activation)
    weight_global_scale = _make_global_scale(weight)
    alpha = (1.0 / (input_global_scale * weight_global_scale)).reshape(1)

    packed_input, input_sf = fp4_quantize(activation, input_global_scale)
    packed_weight, weight_sf = fp4_quantize(weight, weight_global_scale)

    expected = _legacy_dense_expected(
        packed_input,
        packed_weight,
        input_sf,
        weight_sf,
        alpha,
        dtype,
    )

    actual = _b12x_fp4_gemm(
        packed_input,
        packed_weight,
        input_sf,
        weight_sf,
        alpha,
        dtype,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
