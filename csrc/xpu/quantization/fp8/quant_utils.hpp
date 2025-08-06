#include <ATen/ScalarType.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e4m3fn.h>
#include <algorithm>
namespace vllm {

enum class Fp8KVCacheDataType {
  kAuto = 0,
  kFp8E4M3 = 1,
  kFp8E5M2 = 2,
};

namespace fp8 {

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ Tout scaled_convert(const Tin& x, const float scale) {
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    return static_cast<at::Float8_e4m3fn>(x * scale);
  } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
    return static_cast<at::Float8_e5m2>(x * scale);
  }

  assert(false);
  return {};  // Squash missing return statement warning
}

}  // namespace fp8

// Vectorization containers
template <typename scalar_t, size_t vec_size>
struct vec_n_t {
  scalar_t val[vec_size];
} __attribute__((aligned(vec_size * sizeof(scalar_t))));

template <int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
struct DefaultVecOp {
  ScaOp scalar_op;

  inline void operator()(vec_n_t<OutT, VEC_SIZE>& dst,
                         const vec_n_t<InT, VEC_SIZE>& src) const {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      scalar_op(dst.val[i], src.val[i]);
    }
  }
};

template <int VEC_SIZE, typename InT, typename OutT, typename VecOp,
          typename ScaOp>
inline void vectorize_with_alignment(
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

    using vin_t = vec_n_t<InT, VEC_SIZE>;
    using vout_t = vec_n_t<OutT, VEC_SIZE>;
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
  using vin_t = vec_n_t<InT, VEC_SIZE>;
  using vout_t = vec_n_t<OutT, VEC_SIZE>;
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
  vectorize_with_alignment<VEC_SIZE>(in, out, len, tid, stride, Vec{scalar_op},
                                     std::forward<ScaOp>(scalar_op));
}
}  // namespace vllm

// The following macro is used to dispatch the conversion function based on
// the data type of the key and value cache. The FN is a macro that calls a
// function with template<typename scalar_t, typename cache_t,
// Fp8KVCacheDataType kv_dt>.
#define DISPATCH_BY_KV_CACHE_DTYPE(SRC_DTYPE, KV_DTYPE, FN)                    \
  if (KV_DTYPE == "auto") {                                                    \
    if (SRC_DTYPE == at::ScalarType::Float) {                                  \
      FN(float, float, vllm::Fp8KVCacheDataType::kAuto);                       \
    } else if (SRC_DTYPE == at::ScalarType::Half) {                            \
      FN(at::Half, at::Half, vllm::Fp8KVCacheDataType::kAuto);                 \
    } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                        \
      FN(at::BFloat16, at::BFloat16, vllm::Fp8KVCacheDataType::kAuto);         \
    } else {                                                                   \
      TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE);   \
    }                                                                          \
  } else {                                                                     \
    if (KV_DTYPE == "fp8" || KV_DTYPE == "fp8_e4m3") {                         \
      if (SRC_DTYPE == at::ScalarType::Float) {                                \
        FN(float, at::Float8_e4m3fn, vllm::Fp8KVCacheDataType::kFp8E4M3);      \
      } else if (SRC_DTYPE == at::ScalarType::Half) {                          \
        FN(at::Half, at::Float8_e4m3fn, vllm::Fp8KVCacheDataType::kFp8E4M3);   \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                      \
        FN(at::BFloat16, at::Float8_e4m3fn,                                    \
           vllm::Fp8KVCacheDataType::kFp8E4M3);                                \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE); \
      }                                                                        \
    } else if (KV_DTYPE == "fp8_e5m2") {                                       \
      if (SRC_DTYPE == at::ScalarType::Float) {                                \
        FN(float, at::Float8_e5m2, vllm::Fp8KVCacheDataType::kFp8E5M2);        \
      } else if (SRC_DTYPE == at::ScalarType::Half) {                          \
        FN(at::Half, at::Float8_e5m2, vllm::Fp8KVCacheDataType::kFp8E5M2);     \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                      \
        FN(at::BFloat16, at::Float8_e5m2, vllm::Fp8KVCacheDataType::kFp8E5M2); \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE); \
      }                                                                        \
    } else {                                                                   \
      TORCH_CHECK(false, "Unsupported data type of kv cache: ", KV_DTYPE);     \
    }                                                                          \
  }