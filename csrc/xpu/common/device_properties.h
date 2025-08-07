#pragma once

#include <ATen/xpu/XPUContext.h>
#include <iostream>

#include "xpu/common/runtime.h"

namespace vllm {
namespace xpu {
static inline int64_t syclMaxWorkGroupSize(
    at::DeviceIndex dev_id = getDeviceIndexOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->max_work_group_size;
}

template <typename T>
uint32_t syclPrefVectorWidth(
    at::DeviceIndex dev_id = getDeviceIndexOfCurrentQueue()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  if (std::is_same<T, char>::value) {
    return dev_prop->preferred_vector_width_char;
  }
  if (std::is_same<T, short>::value) {
    return dev_prop->preferred_vector_width_short;
  }
  if (std::is_same<T, int>::value) {
    return dev_prop->preferred_vector_width_int;
  }
  if (std::is_same<T, int64_t>::value) {
    return dev_prop->preferred_vector_width_long;
  }
  if (std::is_same<T, float>::value) {
    return dev_prop->preferred_vector_width_float;
  }
  if (std::is_same<T, double>::value) {
    return dev_prop->preferred_vector_width_double;
  }
  if (std::is_same<T, sycl::half>::value) {
    return dev_prop->preferred_vector_width_half;
  }
  throw std::invalid_argument(
      "Invalid data type to fetch preferred vector width!");
}
// template <typename T>
// uint32_t syclPrefVectorWidth(
//     at::DeviceIndex dev_id = getDeviceIndexOfCurrentQueue()) {
//   (void)dev_id;  // Suppress unused variable warning

//   // Hot fix. This is the preferred vector width for GPUs up to LNL/BMG.
//   uint32_t vec_width = 16;

//   if (std::is_same<T, char>::value) {
//     return vec_width / sizeof(char);
//   }
//   if (std::is_same<T, short>::value) {
//     return vec_width / sizeof(short);
//   }
//   if (std::is_same<T, int>::value) {
//     return vec_width / sizeof(int);
//   }
//   if (std::is_same<T, int64_t>::value) {
//     return vec_width / sizeof(int64_t);
//   }
//   if (std::is_same<T, float>::value) {
//     return vec_width / sizeof(float);
//   }
//   if (std::is_same<T, double>::value) {
//     return vec_width / sizeof(double);
//   }
//   if (std::is_same<T, ::sycl::half>::value) {
//     return vec_width / sizeof(::sycl::half);
//   }
//   return 1;
// }
}  // namespace xpu
}  // namespace vllm