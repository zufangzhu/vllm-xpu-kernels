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

}  // namespace xpu
}  // namespace vllm