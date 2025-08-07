#include <ATen/xpu/XPUContext.h>
#include <CL/sycl.hpp>
#include <assert.h>

#include "xpu/common/device_properties.h"

namespace vllm {
namespace xpu {
namespace memory {

// aligned vector generates vectorized load/store
template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

template <int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
struct DefaultVecOp {
  ScaOp scalar_op;

  inline void operator()(aligned_vector<OutT, VEC_SIZE>& dst,
                         const aligned_vector<InT, VEC_SIZE>& src) const {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      scalar_op(dst.val[i], src.val[i]);
    }
  }
};

template <typename scalar_t>
static inline int preferred_vector_width(
    at::DeviceIndex dev_id = getDeviceIndexOfCurrentQueue()) {
  size_t ret;
  int elem_size = sizeof(scalar_t);
  switch (elem_size) {
    case 1:
      static_assert(sizeof(char) == 1, "the char size is not 1 bytes");
      ret = vllm::xpu::syclPrefVectorWidth<char>(dev_id);
      break;
    case 2:
      static_assert(sizeof(short) == 2, "the short size is not 2 bytes");
      ret = vllm::xpu::syclPrefVectorWidth<short>(dev_id);
      break;
    case 4:
      static_assert(sizeof(int) == 4, "the long size is not 4 bytes");
      ret = vllm::xpu::syclPrefVectorWidth<int>(dev_id);
      break;
    case 8:
      static_assert(sizeof(int64_t) == 8, "the long size is not 8");
      ret = vllm::xpu::syclPrefVectorWidth<int64_t>(dev_id);
      break;
    default:
      // no vectorize
      ret = 1;
  }
  return ret;
}

template <typename scalar_t>
inline int can_vectorize_up_to(at::DeviceIndex dev_id, char* pointer) {
  int preferred_width = preferred_vector_width<scalar_t>(dev_id);
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec2_alignment =
      std::alignment_of<typename aligned_vector<scalar_t, 2>::type>::value;
  constexpr int vec4_alignment =
      std::alignment_of<typename aligned_vector<scalar_t, 4>::type>::value;
  constexpr int vec8_alignment =
      std::alignment_of<typename aligned_vector<scalar_t, 8>::type>::value;
  constexpr int vec16_alignment =
      std::alignment_of<typename aligned_vector<scalar_t, 16>::type>::value;

  if (address % vec16_alignment == 0) {
    return std::min<int>(preferred_width, 16);
  } else if (address % vec8_alignment == 0) {
    return std::min<int>(preferred_width, 8);
  } else if (address % vec4_alignment == 0) {
    return std::min<int>(preferred_width, 4);
  } else if (address % vec2_alignment == 0) {
    return std::min<int>(preferred_width, 2);
  }
  return 1;
}

template <int VEC_SIZE, typename InT, typename OutT, typename VecOp,
          typename ScaOp>
inline void vectorize_with_alignment1(
    const InT* in, OutT* out, int len, int tid, int stride,
    VecOp&& vec_op,       // vec_n_t<InT,16> -> vec_n_t<OutT,16>
    ScaOp&& scalar_op) {  // InT -> OutT
  static_assert(VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0,
                "VEC_SIZE must be a positive power-of-two");
  constexpr int WIDTH = VEC_SIZE * sizeof(InT);  // eg: 64 B
  uintptr_t addr = reinterpret_cast<uintptr_t>(in);

  // fast path when the whole region is already aligned
  // Note: currently the output is guaranteed to be same as the input, so we
  // don't check it here, comments here just for future reference.
  bool can_vec = ((addr & (WIDTH - 1)) == 0) && ((len & (VEC_SIZE - 1)) == 0);
  if (can_vec) {
    int num_vec = len / VEC_SIZE;

    using vin_t = aligned_vector<InT, VEC_SIZE>;
    using vout_t = aligned_vector<OutT, VEC_SIZE>;
    auto* v_in = reinterpret_cast<const vin_t*>(in);
    auto* v_out = reinterpret_cast<vout_t*>(out);

    for (int i = tid; i < num_vec; i += stride) {
      vout_t tmp;
      vec_op(tmp, v_in[i]);
      v_out[i] = tmp;
    }
    return;
  }

  int misalignment_offset = addr & (WIDTH - 1);       // addr % 64
  int alignment_bytes = WIDTH - misalignment_offset;  // 64 - (addr % 64)
  int prefix_elems = alignment_bytes & (WIDTH - 1);   // handle 64
  prefix_elems /= sizeof(InT);
  prefix_elems = std::min(prefix_elems, len);  // 0 ≤ prefix < 16

  // 1. prefill the when it is unsafe to vectorize
  for (int i = tid; i < prefix_elems; i += stride) {
    scalar_op(out[i], in[i]);
  }

  in += prefix_elems;
  out += prefix_elems;
  len -= prefix_elems;

  int num_vec = len / VEC_SIZE;
  using vin_t = aligned_vector<InT, VEC_SIZE>;
  using vout_t = aligned_vector<OutT, VEC_SIZE>;
  auto* v_in = reinterpret_cast<const vin_t*>(in);
  auto* v_out = reinterpret_cast<vout_t*>(out);

  // 2. vectorize the main part
  for (int i = tid; i < num_vec; i += stride) {
    vout_t tmp;
    vec_op(tmp, v_in[i]);
    v_out[i] = tmp;
  }

  // 3. handle the tail
  int tail_start = num_vec * VEC_SIZE;
  for (int i = tid + tail_start; i < len; i += stride) {
    scalar_op(out[i], in[i]);
  }
}

template <int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
inline void vectorize_with_alignment(const InT* in, OutT* out, int len, int tid,
                                     int stride, ScaOp&& scalar_op) {
  using Vec = DefaultVecOp<VEC_SIZE, InT, OutT, std::decay_t<ScaOp>>;
  vectorize_with_alignment1<VEC_SIZE>(in, out, len, tid, stride, Vec{scalar_op},
                                      std::forward<ScaOp>(scalar_op));
}

}  // namespace memory
}  // namespace xpu
}  // namespace vllm