// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// compute_slot_mapping: XPU/SYCL replacement for the Triton kernel
// ``_compute_slot_mapping_kernel`` in
// ``vllm/v1/worker/block_table.py``.
//
// For each scheduled token it computes the physical KV-cache slot:
//
//   pos          = positions[i]
//   block_index  = pos / block_size
//   block_number = block_table[req, block_index]
//   slot         = block_number * block_size + pos % block_size
//
// and writes it into ``slot_mapping``. The tail range
// ``[num_tokens, max_num_tokens)`` is padded with ``PAD_ID`` so the buffer is
// safe to consume under CUDA/XPU-graph padding.
//
// This is the *no context-parallel* path (TOTAL_CP_WORLD_SIZE == 1 and
// BLOCKS_PER_KV_BLOCK == 1), which is what the CPU fallback also implements
// (see ``vllm/utils/cpu_triton_utils.py``). Context/decode parallelism and
// hybrid kernel blocks are intentionally not handled here.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <torch/all.h>
#include <ATen/DeviceGuard.h>
#include "utils.h"

namespace vllm {

// One work-group per request. Work-group id == req_idx.
class compute_slot_mapping_kernel {
 public:
  compute_slot_mapping_kernel(
      const int32_t* __restrict__ query_start_loc_,  // [num_reqs + 1], int32
      const int64_t* __restrict__ positions_,        // [num_tokens], int64
      const int32_t* __restrict__ block_table_,      // flat int32
      int64_t* __restrict__ slot_mapping_,           // [max_num_tokens], int64
      const int64_t block_table_stride_,
      const int64_t block_size_,
      const int64_t num_reqs_,
      const int64_t num_tokens_,
      const int64_t max_num_tokens_,
      const int64_t pad_id_)
      : query_start_loc(query_start_loc_),
        positions(positions_),
        block_table(block_table_),
        slot_mapping(slot_mapping_),
        block_table_stride(block_table_stride_),
        block_size(block_size_),
        num_reqs(num_reqs_),
        num_tokens(num_tokens_),
        max_num_tokens(max_num_tokens_),
        pad_id(pad_id_) {}

  void operator()(const sycl::nd_item<1>& item) const {
    const int64_t req_idx = item.get_group(0);
    const int64_t lid = item.get_local_id(0);
    const int64_t lrange = item.get_local_range(0);

    // Last work-group: pad the tail [num_tokens, max_num_tokens) with PAD_ID.
    if (req_idx == num_reqs) {
      for (int64_t i = num_tokens + lid; i < max_num_tokens; i += lrange) {
        slot_mapping[i] = pad_id;
      }
      return;
    }

    const int64_t start_idx = static_cast<int64_t>(query_start_loc[req_idx]);
    const int64_t end_idx = static_cast<int64_t>(query_start_loc[req_idx + 1]);
    const int64_t row_offset = req_idx * block_table_stride;

    for (int64_t i = start_idx + lid; i < end_idx; i += lrange) {
      const int64_t pos = positions[i];
      const int64_t block_index = pos / block_size;
      const int64_t block_number =
          static_cast<int64_t>(block_table[row_offset + block_index]);
      slot_mapping[i] = block_number * block_size + (pos % block_size);
    }
  }

 private:
  const int32_t* __restrict__ query_start_loc;
  const int64_t* __restrict__ positions;
  const int32_t* __restrict__ block_table;
  int64_t* __restrict__ slot_mapping;
  const int64_t block_table_stride;
  const int64_t block_size;
  const int64_t num_reqs;
  const int64_t num_tokens;
  const int64_t max_num_tokens;
  const int64_t pad_id;
};

}  // namespace vllm

// Host launcher. Signature mirrors the CPU op:
//   compute_slot_mapping(query_start_loc, positions, block_table,
//                        slot_mapping, block_size, pad_id)
// ``slot_mapping`` is written in place; its length (max_num_tokens) drives the
// tail padding, while ``positions.size(0)`` gives the number of real tokens.
void compute_slot_mapping(
    const torch::Tensor& query_start_loc,  // [num_reqs + 1], int32
    const torch::Tensor& positions,        // [num_tokens], int64
    const torch::Tensor& block_table,      // [max_num_reqs, stride], int32
    torch::Tensor& slot_mapping,           // [max_num_tokens], int64
    int64_t block_size,
    int64_t pad_id) {
  TORCH_CHECK(
      query_start_loc.dtype() == torch::kInt32,
      "query_start_loc must be int32");
  TORCH_CHECK(positions.dtype() == torch::kInt64, "positions must be int64");
  TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must be int32");
  TORCH_CHECK(
      slot_mapping.dtype() == torch::kInt64, "slot_mapping must be int64");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be 2D");
  TORCH_CHECK(query_start_loc.is_contiguous(), "query_start_loc must be contiguous");
  TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
  TORCH_CHECK(slot_mapping.is_contiguous(), "slot_mapping must be contiguous");

  const int64_t num_reqs = query_start_loc.size(0) - 1;
  const int64_t num_tokens = positions.size(0);
  const int64_t max_num_tokens = slot_mapping.size(0);
  const int64_t block_table_stride = block_table.stride(0);

  TORCH_CHECK(num_reqs >= 0, "query_start_loc must have at least 1 element");
  TORCH_CHECK(
      max_num_tokens >= num_tokens,
      "slot_mapping (",
      max_num_tokens,
      ") must be >= positions (",
      num_tokens,
      ")");

  if (max_num_tokens == 0) {
    return;
  }

  // num_reqs real groups + 1 padding group.
  const int64_t num_groups = num_reqs + 1;
  const int64_t local_size = 256;
  sycl::range<1> global(num_groups * local_size);
  sycl::range<1> local(local_size);

  at::DeviceGuard device_guard(positions.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  const int32_t* query_start_loc_ptr = query_start_loc.data_ptr<int32_t>();
  const int64_t* positions_ptr = positions.data_ptr<int64_t>();
  const int32_t* block_table_ptr = block_table.data_ptr<int32_t>();
  int64_t* slot_mapping_ptr = slot_mapping.data_ptr<int64_t>();

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(global, local),
        vllm::compute_slot_mapping_kernel(
            query_start_loc_ptr,
            positions_ptr,
            block_table_ptr,
            slot_mapping_ptr,
            block_table_stride,
            block_size,
            num_reqs,
            num_tokens,
            max_num_tokens,
            pad_id));
  });
}
