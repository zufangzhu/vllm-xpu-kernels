#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/xpu/XPUContext.h>
#include <sycl/sycl.hpp>

#include "dispatch_utils.h"
#include "ops.h"
#include "utils.h"

#include "quantization/int8/int8_quant.h"

namespace vllm {
template <typename scalar_t, typename int8_type>
class dynamic_per_token_scaled_int8_quant_kernel {
 private:
  int8_type* out;
  float* scale;
  scalar_t const* input;
  const int hidden_size;

 public:
  dynamic_per_token_scaled_int8_quant_kernel(
      int8_type* out,
      float* scale,
      scalar_t const* input,
      const int hidden_size)
      : out(out), scale(scale), input(input), hidden_size(hidden_size) {}

  void operator()() const {
    // Implement the kernel logic here
  }
};
}  // namespace vllm

void dynamic_per_token_scaled_int8_quant(
    torch::Tensor& out,
    const torch::Tensor& input,
    torch::Tensor& scales,
    torch::Tensor& zp,
    bool use_sym_quant) {
  TORCH_CHECK(
      input.is_contiguous(),
      "dynamic_per_token_quant only supports contiguous input tensor");
  // init out tensor, scales, zp
  int64_t hidden_size = input.size(-1);
  int64_t m = input.numel() / hidden_size;

  sycl::range<1> grid(m);
  sycl::range<1> block(std::min(hidden_size, 1024));

  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  at::DeviceGuard device_guard(curDevice);

  auto& queue = vllm::xpu::vllmGetQueue();
  VLLM_DISPATCH_QUANT_TYPES(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "dynamic_per_token_scaled_int8_quant",
      [=]() {
        queue.submit([&](sycl::handler& cgh) {
          if (use_sym_quant) {
            auto kernel = DynamicPerTokenQuantActFunctor<scalar_t, int8_t>{
                input.data_ptr<scalar_t>(),
                out.data_ptr<int8_t>(),
                scales.data_ptr<scalar_t>(),
                zps.data_ptr<int32_t>(),
                hidden_size,
                use_sym_quant};
          } else {
            auto kernel = DynamicPerTokenQuantActFunctor<scalar_t, uint8_t>{
                input.data_ptr<scalar_t>(),
                out.data_ptr<uint8_t>(),
                scales.data_ptr<scalar_t>(),
                zps.data_ptr<int32_t>(),
                hidden_size,
                use_sym_quant};
          }
          cgh.parallel_for(sycl::nd_range<1>(grid * block, block), kernel);
        });
      });
}

// std::tuple<Tensor, Tensor, Tensor>
// dynamic_per_tensor_quant(const Tensor& input, bool use_sym_quant) {
//   TORCH_CHECK(
//       input.is_contiguous(),
//       "dynamic_per_tensor_quant only supports contiguous input tensor");
//   // init out tensor, scales, zp
//   int64_t hidden_size = input.size(-1);
//   int64_t num_elements = input.numel();
//   int64_t num_tokens = num_elements / hidden_size;
//   auto out_dtype = use_sym_quant ? at::kChar : at::kByte;
//   auto out = at::empty_like(input, out_dtype);
//   // scales use same dtype as input, zp use int32
//   // OPTIMIZE ME! currently use a torch op to find the min/max
//   auto [input_min, input_max] = input.aminmax();
//   auto scale = at::empty({1}, input.options());
//   auto zp = at::empty({1}, input.options().dtype(at::kInt));

//   switch (input.scalar_type()) {
//     case at::kFloat: {
//       GetPerTensorScaleZPFunctor<float> compute_scale_zp(
//           input_min.data_ptr<float>(),
//           input_max.data_ptr<float>(),
//           scale.data_ptr<float>(),
//           zp.data_ptr<int32_t>(),
//           use_sym_quant);
//       dpcppGetCurrentQueue().single_task<decltype(compute_scale_zp)>(
//           compute_scale_zp);
//       break;
//     }
//     case at::kHalf: {
//       GetPerTensorScaleZPFunctor<at::Half> compute_scale_zp(
//           input_min.data_ptr<at::Half>(),
//           input_max.data_ptr<at::Half>(),
//           scale.data_ptr<at::Half>(),
//           zp.data_ptr<int32_t>(),
//           use_sym_quant);
//       dpcppGetCurrentQueue().single_task<decltype(compute_scale_zp)>(
//           compute_scale_zp);
//       break;
//     }
//     default:
//       TORCH_CHECK(false, "Unsupported input type for
//       dynamic_per_tensor_quant");
//   }

//   auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
//   int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);

//   auto stream = at::xpu::getCurrentXPUStream().queue();
//   IPEX_DISPATCH_FLOATING_TYPES_AND2(
//       at::ScalarType::Half,
//       at::ScalarType::BFloat16,
//       input.scalar_type(),
//       "dynamic_per_tensor_quant",
//       [=]() {
//         auto cgf = DPCPP_Q_CGF(cgh) {
//           if (use_sym_quant) {
//             auto kernel = DynamicPerTensorQuantFunctor<scalar_t, int8_t>{
//                 input.data_ptr<scalar_t>(),
//                 out.data_ptr<int8_t>(),
//                 scale.data_ptr<scalar_t>(),
//                 zp.data_ptr<int32_t>(),
//                 num_elements,
//                 use_sym_quant};
//             cgh.parallel_for<decltype(kernel)>(
//                 sycl::nd_range<1>(
//                     sycl::range<1>(num_tokens * max_wg_size),
//                     sycl::range<1>(max_wg_size)),
//                 kernel);
//           } else {
//             auto kernel = DynamicPerTensorQuantFunctor<scalar_t, uint8_t>{
//                 input.data_ptr<scalar_t>(),
//                 out.data_ptr<uint8_t>(),
//                 scale.data_ptr<scalar_t>(),
//                 zp.data_ptr<int32_t>(),
//                 num_elements,
//                 use_sym_quant};
//             cgh.parallel_for<decltype(kernel)>(
//                 sycl::nd_range<1>(
//                     sycl::range<1>(num_tokens * max_wg_size),
//                     sycl::range<1>(max_wg_size)),
//                 kernel);
//           }
//         };
//         DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
//       });

//   return std::make_tuple<Tensor&&, Tensor&&, Tensor&&>(
//       std::move(out), std::move(scale), std::move(zp));
// }
