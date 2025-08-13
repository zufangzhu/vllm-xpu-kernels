# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.ops.onednn_gemm_op import QuantDtype, WeightOnlyQuantizedLinear
from tests.utils import _cast_from_fp8, _cast_to_fp8


@pytest.mark.parametrize("M", [4])
@pytest.mark.parametrize("N", [5])
@pytest.mark.parametrize("K", [6])
@pytest.mark.parametrize("with_bias", [False])
@pytest.mark.parametrize("fp8_dtype", ["fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@torch.inference_mode()
def test_fp8_gemm_w8a16(M, N, K, with_bias, dtype, fp8_dtype):
    device = "xpu"
    torch.set_default_device(device)

    input = torch.randn([1, 8, 2], dtype=dtype,
                        device=torch.device("xpu")) / 10.0
    weight_cpu = torch.rand([3, 2], dtype=dtype) / 10.0
    weight = weight_cpu.to(device)

    scale_wei = torch.ones(1) * 4
    scale_wei_inv = torch.tensor([0.25]).xpu()
    weight_fp8_cpu = _cast_to_fp8(weight_cpu, scale_wei, fp8_dtype)
    weight_dequantized_cpu = _cast_from_fp8(weight_fp8_cpu, scale_wei, dtype)

    gemm_ref = torch.nn.Linear(2, 3, bias=with_bias).xpu().to(dtype)
    gemm_ref.weight.data = weight_dequantized_cpu.to(device)
    output_ref = gemm_ref(input)

    fp8_linear = WeightOnlyQuantizedLinear.from_weight(
        weight_fp8_cpu.to(device),
        scale_wei_inv,
        torch.Tensor(),
        in_features=weight.shape[1],
        out_features=weight.shape[0],
        dtype=(QuantDtype.FP8_E5M2
               if fp8_dtype == torch.float8_e5m2 else QuantDtype.FP8_E4M3FN),
    )

    output_fp8 = fp8_linear(input,
                            gemm_ref.bias.data.clone() if with_bias else None)
    torch.testing.assert_close(output_fp8, output_ref, atol=1e-2, rtol=1e-2)
