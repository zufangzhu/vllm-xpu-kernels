# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.ops.mx_utils import (FP4_EBITS, FP4_MBITS, _floatx_unpacked_to_f32,
                                from_blocked_format, to_mxfp, unpack_uint4)
from tests.register_ops import fp4_gemm

MX_MNK_FACTORS = [
    (1, 32, 32),
    (1, 64, 32),
    (32, 32, 32),
    (32, 64, 32),
]


def _convert_to_mxfp4_with_hp_ref(t):
    # Convert a tensor to mxfp8, returning:
    #   t_hp : reconstructed bf16 version of t_lp
    #   t_lp : fp4_e2m1x2 tensor
    #   t_scale: fp8_e8m0 block-wise scaling factors (non-swizzled)
    t_scale, t_lp = to_mxfp(t, format="mxfp4")
    t_hp = from_blocked_format(
        _floatx_unpacked_to_f32(unpack_uint4(t_lp), FP4_EBITS, FP4_MBITS),
        t_scale,
        blocksize=32,
    )

    return t_hp, t_lp, t_scale


@pytest.mark.parametrize("mnk_factors", MX_MNK_FACTORS)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_mxfp4_gemm(mnk_factors, out_dtype):
    m, n, k = mnk_factors
    input_dtype = out_dtype
    if out_dtype is torch.float16:
        input_dtype = torch.float
    inputs = torch.randn((m, k), dtype=input_dtype).xpu() * 0.01
    weights = torch.randn((n, k), dtype=input_dtype).xpu() * 0.01
    inputs_hp, inputs_lp, inputs_scale = _convert_to_mxfp4_with_hp_ref(inputs)
    weights_hp, weights_lp, weights_scale = _convert_to_mxfp4_with_hp_ref(
        weights)
    output = fp4_gemm(
        inputs_lp,
        weights_lp.transpose(0, 1),
        inputs_scale,
        weights_scale,
        out_dtype,
        torch.Tensor(),
    )

    output_ref = torch.matmul(inputs_hp, weights_hp.t())
    torch.testing.assert_close(output.to(torch.float),
                               output_ref.to(torch.float),
                               atol=5e-2,
                               rtol=5e-2)
