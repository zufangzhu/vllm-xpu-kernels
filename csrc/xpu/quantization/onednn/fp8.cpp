#include <vector>
#include "fp8.h"

torch::Tensor fp8_gemm_w8a16(const torch::Tensor& A, const torch::Tensor& B,
                             bool trans_B,
                             const c10::optional<torch::Tensor>& B_scale_,
                             const c10::optional<torch::Tensor>& bias_) {
  std::vector<int64_t> result_shape;
  if (A.dim() == 2) {
    if (trans_B) {
      result_shape = {A.size(0), B.size(0)};
    } else {
      result_shape = {A.size(0), B.size(1)};
    }
    // src{m, k}, wei{k, n}, bias{n}, dst{m, n}
  } else if (A.dim() == 3) {
    if (B.dim() == 2) {
      if (trans_B) {
        result_shape = {A.size(0), A.size(1), B.size(0)};
      } else {
        result_shape = {A.size(0), A.size(1), B.size(1)};
      }
      // src{b, m, k}, wei{k, n}, bias{n}, dst{b, m, n}
    } else {
      TORCH_CHECK(false, "fp8_gemm only support 2D weight\n");
    }
  } else {
    TORCH_CHECK(false, "linear only support for 2D and 3D tensors!\n");
  }

  // deal with input shape [m, b, k] stride [k, m * k, 1]
  auto k = A.size(A.dim() - 1);
  auto n = result_shape.back();
  auto res_stride = A.strides().vec();
  for (int i = 0; i < res_stride.size() - 1; i++) {
    res_stride[i] = res_stride[i] / k * n;
  }

  torch::Tensor result =
      at::empty_strided(result_shape, res_stride, A.options());

  // check if nt format
  bool is_nt = true;
  if (trans_B) {
    is_nt = B.strides()[B.dim() - 1] == 1;
  } else {
    is_nt = B.strides()[B.dim() - 2] == 1;
  }

  torch::Tensor B_scale = B_scale_.has_value()
                              ? B_scale_.value()
                              : at::ones({1}, B.options().dtype(A.dtype()));

  oneDNN::dnnl_matmul_w8a16_fp8(result, A, B, is_nt, bias_, B_scale);

  return result;
}
