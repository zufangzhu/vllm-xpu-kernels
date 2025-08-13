# SPDX-License-Identifier: Apache-2.0
from enum import IntEnum
from typing import Optional

import torch
import torch.nn as nn

DTYPE_BITS_MAPPING = {
    "nf4": 4,
    "fp4_e2m1_bnb": 4,
    "fp4_e2m1": 4,
    "int4_fullrange": 4,
    "int4_clip": 4,
    "int8": 8,
    "fp8_e5m2": 8,
    "fp8_e4m3fn": 8,
}


class QuantDtype(IntEnum):
    FP8_E5M2 = 0
    FP8_E4M3FN = 1


class GPTQShuffle(nn.Module):

    def __init__(self, bits=4, blocksize=128):
        super(GPTQShuffle, self).__init__()
        self.bits = bits
        self.blocksize = blocksize

    def convert_idx(self, g_idx, k):
        ret_idx = torch.zeros(k, dtype=int).to(g_idx.device)
        groups = k // self.blocksize
        remainder = k % self.blocksize
        g_idx_2 = g_idx * self.blocksize
        if remainder > 0:
            g_idx_2[g_idx == groups] += torch.arange(remainder).to(
                g_idx.device)
        arange_tensor = torch.arange(self.blocksize).to(g_idx.device)
        for i in range(groups):
            g_idx_2[g_idx == i] += arange_tensor
        ret_idx[g_idx_2] = torch.arange(k).to(g_idx.device)
        return ret_idx.to(torch.int32)

    def unpack(self, qweight_int32):
        s32_bits = 32

        assert self.bits == 4
        # Int32 can store 8 * 4bits data. This is the offset for each data.
        wf = (torch.tensor(list(range(0, s32_bits, self.bits)),
                           dtype=torch.int32).unsqueeze(0).to(
                               qweight_int32.device))
        weight = torch.bitwise_right_shift(
            torch.unsqueeze(qweight_int32, 1).expand(-1, 32 // self.bits, -1),
            wf.unsqueeze(-1),
        ).to(torch.int16 if self.bits == 8 else torch.int8)
        torch.bitwise_and(weight, (2**self.bits) - 1, out=weight)

        return weight

    def pack(self, qweight_int8):
        i = 0
        row = 0
        qweight_int32_shape = (
            qweight_int8.shape[0] // 32 * self.bits,
            qweight_int8.shape[1],
        )
        qweight_int32 = torch.zeros(qweight_int32_shape,
                                    dtype=torch.int32,
                                    device=qweight_int8.device)

        while row < qweight_int32.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight_int32[row] |= qweight_int8[j].to(
                        torch.int32) << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        return qweight_int32

    def forward(self, qweight_int32, g_idx):
        k = qweight_int32.shape[0] * 8
        g_idx4kernel = self.convert_idx(g_idx, k).to(qweight_int32.device)
        qweight_int8 = self.unpack(qweight_int32)
        qweight_int8 = qweight_int8.reshape(-1, qweight_int8.shape[-1])
        qweight_int8_shuffled = qweight_int8[g_idx4kernel, :]
        qweight_int32_shuffled = self.pack(qweight_int8_shuffled)
        return qweight_int32_shuffled, g_idx4kernel


def convert_dtype_str2torch(str_dtype):
    if str_dtype == "int8":
        return torch.int8
    elif str_dtype == "fp32" or str_dtype == "auto":
        return torch.float
    elif str_dtype == "fp16":
        return torch.float16
    elif str_dtype == "bf16":
        return torch.bfloat16
    elif "fp8" in str_dtype:
        return torch.float32
    else:
        AssertionError(False), "Unsupport str dtype {} to torch dtype".format(
            str_dtype)


class ParamsLowBits(torch.nn.Parameter):

    def __new__(
        cls,
        data=None,
        requires_grad=True,
        quant_state=None,
        blocksize=32,
        compress_statistics=True,
        quant_dtype=None,
        scale_dtype="fp32",
        double_quant_scale_dtype=None,
        compression_dtype=None,
    ):
        if data is None:
            data = torch.empty(0)

        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.blocksize = blocksize
        self.compress_statistics = compress_statistics
        self.quant_dtype = quant_dtype
        self.scale_dtype = scale_dtype
        self.double_quant_scale_dtype = double_quant_scale_dtype
        self.quant_state = quant_state
        self.data = data
        self.compression_dtype = compression_dtype
        return self


class WeightOnlyQuantizedLinearImpl(nn.Module):
    __constants__ = ["in_features", "out_features", "blocksize"]
    in_features: int
    out_features: int
    blocksize: int

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            compute_dtype="fp32",
            compress_statistics=True,
            weight_dtype="int4_fullrange",
            scale_dtype="fp16",
            blocksize: int = 64,
            scheme="sym",
            double_quant_scale_dtype=None,
            compression_dtype=torch.int32,
            compression_dim=1,
            device=None,
            use_optimum_format=False,
            quant_method=0,  # QuantMethod(GPTQ_GEMM)
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert compute_dtype in [
            "fp32",
            "fp16",
        ], "compute_dtype must be 'fp32', 'fp16'."
        assert scale_dtype in [
            "fp16",
            "fp32",
        ], "scale_dtype only support 'fp32', 'fp16'. now."
        self.scale_dtype = scale_dtype
        self.double_quant_scale_dtype = double_quant_scale_dtype
        self.compute_dtype = compute_dtype
        self.compress_statistics = compress_statistics
        self.blocksize = (blocksize
                          if blocksize != -1 and blocksize < self.in_features
                          else self.in_features)
        self.scheme = scheme
        self.weight_dtype = weight_dtype
        self.device = device
        # `compression_dim` indicates in which dimension to be compressed in data.
        self.compression_dim = compression_dim
        self.weight_transposed = False
        assert compression_dtype in [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        ], "Only support torch.int8|16|32|64 as compressed dtype."
        dtype_bits_mapping = {
            torch.int8: 8,
            torch.int16: 16,
            torch.int32: 32,
            torch.int64: 64,
            torch.float8_e5m2: 8,
            torch.float8_e4m3fn: 8,
        }
        self.bits = DTYPE_BITS_MAPPING[weight_dtype]
        self.compress_bits = dtype_bits_mapping[compression_dtype]
        self.n_pack = self.compress_bits // self.bits
        self.compression_dtype = compression_dtype
        # K is input channel, N is output channel
        assert compression_dim in [
            0, 1
        ], ("Only support 0 or 1 as compression dimension, " +
            "0 is output channel, 1 is input channel.")

        # `use_optimum_format` is for GPTQ model, if it is True, it's weight is k x n,
        # so it needn't to transpose in optimized operator.
        self.use_optimum_format = use_optimum_format

        self.register_parameter("qweight", None)
        self.register_parameter("bias", None)
        self.register_buffer("g_idx", None)
        self.force_xetla = False

    def transpose_onednn_woq_format(self):
        # The oneDNN int4 GEMM has the following requirements:
        # - Weights need to be contiguous along the 'k' dimension, but the shape should remain (k, n/8).
        # - Scales remains unchanged.
        # - Zero-point is a scalar value of 8 in symmetric (symm) scenarios, allowing oneDNN to broadcast it.
        # - Zero-point remains unchanged in asymmetric (asymm) scenarios.
        reshaped_tensor = self.qweight.transpose(0, 1).contiguous().transpose(
            0, 1)
        self.qweight.as_strided_(reshaped_tensor.shape,
                                 reshaped_tensor.stride())
        self.qweight.copy_(reshaped_tensor)
        self.scales.data = self.scales.contiguous().to(torch.float16)
        if self.bias is not None:
            self.bias.data = self.bias.contiguous().to(torch.float16)
        if (self.scheme == "sym" and self.quant_method == 0
                and self.weight_dtype == "int4_fullrange"):
            self.qzeros = torch.Tensor([8]).to(torch.int8).to("xpu")
        elif self.quant_method == 0 and self.weight_dtype == "int4_fullrange":
            self.qzeros += 0x11111111

    @classmethod
    def from_weight(
        cls,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        in_feature: int,
        out_feature: int,
        bias: Optional[torch.Tensor] = None,
        group_size: int = -1,
        g_idx: Optional[torch.Tensor] = None,
        dtype=0,
        **kwargs,
    ):
        r"""Create a weight-only quantized module from weight

        Args:
            qweight (Tensor): tensor in int32 dtype and contains actually int4 data
            scales (Tensor): scales for qweight
            zero_points (Tensor): zero points for qweight
            in_feature (int): size of each input sample
            out_feature (int): size of each output sample
                Default value is ``None``.
            bias (Tensor or None): bias for linear
            group_size: Group size for weight quantization
            g_idx: Indices of groups for each input channel of weight. Generated by
                GPTQ with act-order.
            dtype (QuantDtype): quantization data type

        """
        from intel_extension_for_pytorch.llm.quantization.utils import (
            QuantDtype)

        assert (
            dtype == QuantDtype.INT4 or dtype == QuantDtype.FP8_E5M2
            or dtype == QuantDtype.FP8_E4M3FN
        ), "IPEX only support INT4 FP8_E5M2 FP8_E4M3FN as quantization data type for now."
        compression_dim = None
        compression_dtype = None
        scheme = "sym"
        compression_dim = 1
        compression_dtype = torch.int32
        if dtype == QuantDtype.FP8_E5M2:
            compression_dtype = torch.float8_e5m2
        elif dtype == QuantDtype.FP8_E4M3FN:
            compression_dtype = torch.float8_e4m3fn
        cls_inst = WeightOnlyQuantizedLinearImpl(
            in_features=in_feature,
            out_features=out_feature,
            bias=True if bias is not None else False,
            compute_dtype="fp32",
            compress_statistics=True,
            weight_dtype=("fp8_e5m2"
                          if dtype == QuantDtype.FP8_E5M2 else "fp8_e4m3fn"),
            scale_dtype="fp16" if scales.dtype == torch.float16 else "fp32",
            blocksize=group_size,
            scheme=scheme,
            double_quant_scale_dtype=None,
            compression_dtype=compression_dtype,
            compression_dim=compression_dim,
            device="xpu",
            use_optimum_format=False,
        )

        if g_idx is not None:
            shuffler = GPTQShuffle(bits=4, blocksize=group_size)
            qweight_new, g_idx_new = shuffler(qweight, g_idx)
            qweight.data.copy_(qweight_new)
            g_idx.data.copy_(g_idx_new)
            qweight_new = None
            g_idx_new = None
            del qweight_new, g_idx_new

        cls_inst.set_weights_bias(qweight, bias)
        cls_inst.set_scales_zps_gidx(scales, zero_points, g_idx)

        if dtype == QuantDtype.FP8_E5M2 or dtype == QuantDtype.FP8_E4M3FN:
            if qweight.shape[0] != in_feature:
                cls_inst.weight_transposed = True

        return cls_inst

    def forward(self,
                input: torch.Tensor,
                bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.compute_dtype == "fp16":
            input = input.to(convert_dtype_str2torch(self.compute_dtype))
        torch.ops.torch_ipex.fp8_gemm_w8a16(
            input,
            self.qweight,
            self.weight_transposed,
            self.scales,
            bias,
        )

    def set_weights_bias(self, weight_data, bias=None, update_g_idx=True):
        qweight = ParamsLowBits(
            data=weight_data,
            requires_grad=False,
            quant_state={"scheme": self.scheme},
            blocksize=self.blocksize,
            compress_statistics=self.compress_statistics,
            quant_dtype=self.weight_dtype,
            scale_dtype=self.scale_dtype,
            double_quant_scale_dtype=self.double_quant_scale_dtype,
            compression_dtype=self.compression_dtype,
        )
        self.qweight = qweight
        if bias is not None:
            self.bias = torch.nn.Parameter(bias.contiguous().to(torch.float16),
                                           requires_grad=False)
        if hasattr(self, "g_idx") and self.g_idx is not None and update_g_idx:
            # The prerequisite for this to work is that set_scales_zps_gidx is called first.
            assert self.qweight.data.dtype == torch.int32
            shuf_weight = GPTQShuffle(self.bits, self.blocksize)
            self.qweight.data, self.g_idx = shuf_weight(
                self.qweight.data, self.g_idx)

    def set_scales_zps_gidx(self, scales, qzeros=None, g_idx=None):
        self.register_buffer("scales", scales)
        self.register_buffer("qzeros", qzeros)
        unuse_g_idx = torch.tensor(
            [i // self.blocksize for i in range(self.in_features)],
            dtype=torch.int32,
            device=scales.device,
        )
        if g_idx is not None and not torch.equal(g_idx, unuse_g_idx):
            self.g_idx = g_idx
        self.qzeros = torch.Tensor([8]).to(torch.int8).to("xpu")

    def extra_repr(self) -> str:
        tmp_str = (
            "in_features={}, out_features={}, bits={}, blocksize={}, bias={}".
            format(
                self.in_features,
                self.out_features,
                self.bits,
                self.blocksize,
                self.bias is not None,
            ))
        if self.use_optimum_format:
            tmp_str += ", use_optimum_format=True"
        return tmp_str


class WeightOnlyQuantizedLinear(nn.Module):
    r"""
    A weight-only quantized (WOQ) linear module with floating point tensor as inputs and outputs.
    Weight is dequantized at runtime for computation.
    """

    def __init__(self, woq_linear_impl):
        super().__init__()
        self.woq_linear_impl = woq_linear_impl

    module_mapping = {
        "cpu":
        "intel_extension_for_pytorch.nn.modules.weight_only_quantization",
        "xpu": "intel_extension_for_pytorch.nn.utils._quantize_convert",
    }

    @classmethod
    def from_weight(
        cls,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        in_features: int,
        out_features: int,
        bias: Optional[torch.Tensor] = None,
        group_size: int = -1,
        g_idx: Optional[torch.Tensor] = None,
        dtype: QuantDtype = QuantDtype.FP8_E4M3FN,
        **kwargs,
    ):
        r"""Create a weight-only quantized module from weight

        Args:
            qweight (Tensor): tensor in int32 dtype and contains actually int4 data
            scales (Tensor): scales for qweight
            zero_points (Tensor): zero points for qweight. FP8 does not need zero points.
            in_features (int): size of each input sample
            out_features (int): size of each output sample
                Default value is ``None``.
            bias (Tensor or None): bias for linear
            group_size: Group size for weight quantization
            g_idx: Indices of groups for each input channel of weight. Generated by
                GPTQ with act-order.
            dtype (QuantDtype): quantization data type

        """
        device_type = qweight.device.type
        assert device_type in {"cpu", "xpu"}, "Device type not supported."
        woq_linear_impl = WeightOnlyQuantizedLinearImpl.from_weight(
            qweight,
            scales,
            zero_points,
            in_features,
            out_features,
            bias,
            group_size,
            g_idx,
            dtype,
            **kwargs,
        )
        return cls(woq_linear_impl)

    def forward(self, x, bias: Optional[torch.Tensor] = None, **kwargs):
        if bias is not None and self.woq_linear_impl.qweight.device.type == "xpu":
            return self.woq_linear_impl(x, bias, **kwargs)
        else:
            return self.woq_linear_impl(x, **kwargs)
