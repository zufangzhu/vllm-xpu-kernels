#pragma once

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

namespace vllm {
namespace xpu {

static inline at::DeviceIndex getDeviceIndexOfCurrentQueue() {
  return c10::xpu::getCurrentXPUStream().device_index();
}

static inline sycl::queue& getCurrentSYCLQueue() {
  return c10::xpu::getCurrentXPUStream().queue();
}

}  // namespace xpu
}  // namespace vllm
