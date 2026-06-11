# SPDX-License-Identifier: Apache-2.0
import torch

from . import _C  # noqa: F401
from . import _xpu_C  # noqa: F401

finfo = torch.finfo(torch.float8_e4m3fn)
FP8_E4M3_MIN = finfo.min
FP8_E4M3_MAX = finfo.max
    
_FP4_E2M1_LUT = torch.tensor(
    [
         0.0,  0.5,  1.0,  1.5,
         2.0,  3.0,  4.0,  6.0,
        -0.0, -0.5, -1.0, -1.5,
        -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)

_FP4_E2M1_LUT_XPU = None

def _get_lut(device):
    global _FP4_E2M1_LUT_XPU
    if _FP4_E2M1_LUT_XPU is None:
        _FP4_E2M1_LUT_XPU = _FP4_E2M1_LUT.to(device)
    return _FP4_E2M1_LUT_XPU

def _fp4_e2m1fn_x2_to_float_lut(
    packed: torch.Tensor,
) -> torch.Tensor:
    """
    packed:
        float4_e2m1fn_x2 tensor
        shape [..., N]

    return:
        fp32 tensor
        shape [..., N*2]
    """

    lut = _get_lut(packed.device)

    u8 = packed.view(torch.uint8)

    lo = u8 & 0xF
    hi = u8 >> 4

    lo_fp = lut[lo.long()]
    hi_fp = lut[hi.long()]

    out = torch.empty(
        *u8.shape[:-1],
        u8.shape[-1] * 2,
        device=u8.device,
        dtype=torch.float32,
    )

    out[..., 0::2] = lo_fp
    out[..., 1::2] = hi_fp

    return out

def dequant_mxfp4(x_lp, x_scale):
    act_ori_shape = x_lp.shape
    x = _fp4_e2m1fn_x2_to_float_lut(x_lp).reshape(-1, 32) * (x_scale.reshape(
            -1, 1).to(torch.float32))
    return x.reshape(act_ori_shape[:-1] + (act_ori_shape[-1] * 2, ))

def dequant_mxfp8(x_lp, x_scale):
    act_ori_shape = x_lp.shape
    x = x_lp.to(torch.float32).reshape(-1, 32) * (x_scale.reshape(-1, 1).to(torch.float32))
    return x.reshape(act_ori_shape)

def quant_mxfp_act_xpu(x, recipe):
    assert recipe in ("mxfp8", "mxfp4")
    if recipe == "mxfp8":
        return _quant_mxfp8_act_xpu(x)
    else:
        return _quant_mxfp4_act_xpu(x)

def _quant_mxfp8_act_xpu(x):
    MXFP8_BLOCK_SIZE = 32
    assert x.shape[-1] % MXFP8_BLOCK_SIZE == 0

    eps = 1e-10
    x_q = torch.empty_like(x, device=x.device, dtype=torch.float8_e4m3fn)
    shape = x.shape[:-1] + (x.shape[-1] // MXFP8_BLOCK_SIZE,)
    x_s = torch.empty(shape, device=x.device, dtype=torch.float32)
    torch.ops._C.per_token_group_fp8_quant(
        x,
        x_q,
        x_s,
        MXFP8_BLOCK_SIZE,
        eps,
        FP8_E4M3_MIN,
        FP8_E4M3_MAX,
        True,
        False,
        False,  # dummy_is_scale_transposed, dummy_is_tma_aligned
    )
    x_s = x_s.to(torch.float8_e8m0fnu)
    return x_q, x_s

def _quant_mxfp4_act_xpu(x):
    MXFP4_BLOCK_SIZE = 32
    eps = 1e-10
    M, N = x.shape
    # Packed FP4 output: two nibbles per byte
    x_q = torch.empty(M, N // 2, device=x.device, dtype=torch.uint8)
    x_s = torch.empty(M, N // MXFP4_BLOCK_SIZE, device=x.device, dtype=torch.float32)

    torch.ops._C.per_token_group_quant_mxfp4(x, x_q, x_s, MXFP4_BLOCK_SIZE, eps)

    x_q = x_q.view(torch.float4_e2m1fn_x2)
    x_s = x_s.to(dtype=torch.float8_e8m0fnu, memory_format=torch.preserve_format)
    return x_q, x_s

def qdq_fp8_act(x):
    x_fp = x.to(torch.float32)
    scale = (x_fp.abs().max() / FP8_E4M3_MAX).clamp(
            min=torch.finfo(torch.float32).eps
        )
    return (x_fp / scale).clamp(
            -FP8_E4M3_MAX, FP8_E4M3_MAX
        ).to(torch.float8_e4m3fn).to(torch.float32) * scale

def dequant_fp8_block_wei(x_lp, x_scale):
    orig_shape = x_lp.shape
    M, K = orig_shape
    x_lp = x_lp.view(M // 128, 128, K // 128, 128)
    x_scale = x_scale.unsqueeze(1).unsqueeze(-1)
    x_hp = x_lp.to(torch.float32)
    x_hp = x_hp * x_scale
    return x_hp.reshape(orig_shape).to(torch.float32)

def dequant_fp8_block_act(x_lp, x_scale):
    orig_shape = x_lp.shape
    x_lp = x_lp.reshape(x_lp.shape[0], x_lp.shape[-1] // 128, 128)
    x_hp = x_lp.to(torch.float32)
    x_hp = x_hp * x_scale.unsqueeze(-1)
    return x_hp.reshape(orig_shape).to(torch.float32)

def quant_fp8_block_act(x: torch.Tensor):
    x_q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)    
    shape = x.shape[:-1] + (x.shape[-1] // 128,)
    x_s = torch.empty(shape, device=x.device, dtype=torch.float32)
    torch.ops._C.per_token_group_fp8_quant(
            x,
            x_q,
            x_s,
            128,
            1e-10,
            FP8_E4M3_MIN,
            FP8_E4M3_MAX,
            False,
            False,
            False,
        )
    return x_q, x_s