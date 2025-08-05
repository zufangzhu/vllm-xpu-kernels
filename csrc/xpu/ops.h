#pragma once

#include <torch/all.h>

void rms_norm(torch::Tensor &out, torch::Tensor &input, torch::Tensor &weight,
              double epsilon);

void fused_add_rms_norm(torch::Tensor &input, torch::Tensor &residual,
                        torch::Tensor &weight, double epsilon);

torch::Tensor fp8_gemm_w8a16(const torch::Tensor& A, const torch::Tensor& B,
                             bool trans_B,
                             const c10::optional<torch::Tensor>& B_scale_,
                             const c10::optional<torch::Tensor>& bias_);