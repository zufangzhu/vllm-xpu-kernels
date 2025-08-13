// #include <string>
#include <sycl/sycl.hpp>
// #include <sycl/__spirv/spirv_ops.hpp>
// #include <ATen/ATen.h>
// #include <torch/extension.h>
#include "dispatch_utils.h"
#include "quantization/fp8/quant_utils.hpp"
#include "utils.h"

extern "C" {
#if defined(__SYCL_DEVICE_ONLY__)
SYCL_EXTERNAL sycl::half __builtin_IB_bf8tohf_1(char a) __attribute__((const));
SYCL_EXTERNAL sycl::half __builtin_IB_hf8tohf_1(char a) __attribute__((const));
#else
sycl::half __builtin_IB_bf8tohf_1(char a) __attribute__((const)) {
  return sycl::half(a);
};
sycl::half __builtin_IB_hf8tohf_1(char a) __attribute__((const)) {
  return sycl::half(a);
};
#endif
}

namespace vllm {
// /* Scaled and vectorized conversions, for data exchange between high and low
//    precision domains Convention of the scale in API, e.g: FP8_data =
//    Quantization( High_Precision_data / scale ) s.t. Quantize(HP / scale) =>
//    FP8
//      Dequant(FP8) * scale =>  HP
//  */
template <typename Tout, typename Tin>
__inline__ Tout scaled_vec_conversion(const Tin& x, const float scale,
                                      const Fp8KVCacheDataType& quant_type) {
  return x;
}

// fp8_e4m3 -> half

template <>
__inline__ uint16_t scaled_vec_conversion<uint16_t, uint8_t>(
    const uint8_t& a, const float scale, const Fp8KVCacheDataType& quant_type) {
  char a_char = static_cast<char>(a);
  if (quant_type == Fp8KVCacheDataType::kFp8E5M2) {
    return __builtin_IB_bf8tohf_1(a_char);
  }
  return __builtin_IB_hf8tohf_1(a_char);
}

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ Tout scaled_convert1(const Tin& x, const float scale) {
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    return scaled_vec_conversion<Tout, Tin>(x, scale,
                                            Fp8KVCacheDataType::kFp8E4M3);
  } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
    return scaled_vec_conversion<Tout, Tin>(x, scale,
                                            Fp8KVCacheDataType::kFp8E5M2);
  }
  assert(false);
}

void cast_fp8_to_fp16(const uint8_t* input, uint16_t* output, float scale,
                      const sycl::nd_item<1>& item) {
  size_t idx = item.get_global_id(0);
  output[idx] =
      scaled_convert1<uint16_t, uint8_t, Fp8KVCacheDataType::kFp8E4M3>(
          input[idx], scale);
}

void call_cast_fp8_to_fp16(torch::Tensor& input, torch::Tensor& output,
                           float scale) {
  int num_elems = input.numel();
  auto input_ptr = reinterpret_cast<uint8_t*>(input.data_ptr());
  auto output_ptr = reinterpret_cast<uint16_t*>(output.data_ptr());
  sycl::range<1> grid(num_elems);
  sycl::range<1> block(std::min(num_elems, 1024));
  auto& queue = vllm::xpu::vllmGetQueue();
  std::cout << "Submit Queue" << std::endl;
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(grid * block, block),
                     [=](sycl::nd_item<1> item_ct1) {
                       cast_fp8_to_fp16(input_ptr, output_ptr, scale, item_ct1);
                     });
  });
}
}  // namespace vllm

void cast_fp8_to_fp161(torch::Tensor& input, torch::Tensor& output,
                       double scale) {
  std::cout << "cast_fp8_to_fp161 called with scale" << std::endl;
  VLLM_DISPATCH_QUANT_TYPES(input.scalar_type(), "cast_fp8_to_fp161", [&] {
    vllm::call_cast_fp8_to_fp16(input, output, scale);
  });
}
