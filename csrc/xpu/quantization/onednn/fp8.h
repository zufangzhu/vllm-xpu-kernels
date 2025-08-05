#pragma once

#include "onednn_utils.h"
// #include <ATen/ATen.h>
// #include <ATen/record_function.h>

// #include <oneDNN/Runtime.h>
// #include <runtime/Utils.h>
#include <torch/all.h>
// #include "Attr.h"
// #include "Utils.h"
#include <c10/xpu/XPUStream.h>
#include <oneapi/dnnl/dnnl.hpp>
#include "runtime.h"
// using namespace dnnl;
// using namespace torch_ipex::xpu::dpcpp;

// namespace vllm {
namespace oneDNN {

static inline void dnnl_matmul_w8a16_fp8(
    torch::Tensor& result, const torch::Tensor& mat1, const torch::Tensor& mat2,
    bool trans_b, const std::optional<torch::Tensor>& bias,
    const torch::Tensor& m2_sc, const int64_t group_size = 0) {
  TORCH_CHECK(mat2.scalar_type() == at::ScalarType::Float8_e5m2 ||
                  mat2.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "weight must be f8_e5m2 or f8_e4m3fn for fp8 matmul");
  auto src_sz = mat1.sizes();
  auto o_sz = result.sizes();

  const int m = std::reduce(src_sz.begin(), src_sz.end() - 1, 1,
                            std::multiplies<int64_t>());
  const int n = o_sz.back();  // presume channel last format
  const int k = *(src_sz.end() - 1);

  // get joint dtypes
  joint_dtypes_t jd;
  auto in_dtype = mat1.scalar_type();
  auto wei_dtype = mat2.scalar_type();
  if (in_dtype == at::ScalarType::Half) {
    jd = wei_dtype == at::ScalarType::Float8_e5m2 ? joint_dtypes_t::f16_f8_e5m2
                                                  : joint_dtypes_t::f16_f8_e4m3;
  } else if (in_dtype == at::ScalarType::BFloat16) {
    jd = wei_dtype == at::ScalarType::Float8_e5m2
             ? joint_dtypes_t::bf16_f8_e5m2
             : joint_dtypes_t::bf16_f8_e4m3;
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported data type for fp8 matmul: ", mat1.scalar_type());
  }

  // get bias type
  bias_type_t b_type;
  if (bias.has_value() && bias.value().defined()) {
    auto& b = bias.value();
    const auto nuelm = b.numel();
    if (nuelm == 1) {
      b_type = bias_type_t::scalar;
    } else if (nuelm == m * n) {
      b_type = bias_type_t::mn;
    } else if (b.size(b.dim() - 1) == n && nuelm == n) {
      b_type = bias_type_t::n;
    } else if (b.size(b.dim() - 1) == 1 && nuelm == m) {
      b_type = bias_type_t::m;
    } else if (nuelm == 0) {
      b_type = bias_type_t::none;
    } else {
      TORCH_CHECK(0, "unsupported bias dim in matmul ...", b.sizes());
    }
  } else {
    b_type = bias_type_t::none;
  }

  trans_type_t tt = trans_type_t::nn;
  if (trans_b) {
    // transpose mat2
    tt = trans_type_t::nt;
  }

  // get lda ldb and ldc
  auto mat1_strides = mat1.strides();
  int64_t leading_dim = -1;
  if (mat1.dim() == 2) {
    leading_dim = 0;
  } else if (mat1.dim() == 3) {
    leading_dim = mat1_strides[0] < mat1_strides[1] ? 0 : 1;
  } else {
    TORCH_CHECK(false,
                "Unsupported input dimension for fp8 matmul: ", mat1.dim());
  }
  int64_t lda = mat1_strides[leading_dim];
  int64_t ldb = mat2.strides()[mat2.dim() - 1] == 1
                    ? mat2.strides()[mat2.dim() - 2]
                    : mat2.strides()[mat2.dim() - 1];
  int64_t ldc = result.strides()[leading_dim];

  auto f_attr = [&](dnnl::primitive_attr& pattr) {
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  };

  int arg_off = 0;

  // ************************************************************
  // get device, engine, stream
  const int dev_id = c10::xpu::getCurrentXPUStream().device_index();
  at::Device curDevice = at::Device(at::kXPU, dev_id);
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  auto& matmul_ext = matmul_primitive_create_and_cache(
      jd, tt, b_type, m, n, k, lda, ldb, ldc, dev_id, f_attr, group_size);

  matmul_ext.set_attribute(arg_off++, DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
                           m2_sc.data_ptr(), [&]() {
                             return dpcpp_onednn_memory(get_onednn_md(m2_sc),
                                                        engine,
                                                        m2_sc.data_ptr());
                           });

  std::vector<std::pair<int, void*>> arg_handles;
  arg_handles.reserve(8);

  arg_handles.emplace_back(DNNL_ARG_SRC, mat1.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_WEIGHTS, mat2.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_DST, result.data_ptr());
  if (b_type != bias_type_t::none) {
    arg_handles.emplace_back(DNNL_ARG_BIAS, bias.value().data_ptr());
  }

  int scratchpad_size = matmul_ext.get_scratchpad_size();
  torch::Tensor scratchpad_tensor = at::empty(  // TODO(zzf)
      {scratchpad_size}, mat1.options().dtype(at::kByte), c10::nullopt);
  arg_handles.emplace_back(DNNL_ARG_SCRATCHPAD, scratchpad_tensor.data_ptr());

  auto strm = GpuStreamManager::Instance().get_stream();
  DPCPP_ONEDNN_EXEC_WITH_ARGHANDLES(matmul_ext, strm, engine, arg_handles,
                                    arg_off);
}
}  // namespace oneDNN
// }  // namespace vllm
