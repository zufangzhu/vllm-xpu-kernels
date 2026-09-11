# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the compute_slot_mapping SYCL kernel.

Validates the XPU kernel against a pure-PyTorch reference that mirrors the
no-context-parallel path of the Triton ``_compute_slot_mapping_kernel`` in
``vllm/v1/worker/block_table.py`` (TOTAL_CP_WORLD_SIZE == 1,
BLOCKS_PER_KV_BLOCK == 1).
"""

import pytest
import torch

try:
    import tests.register_ops as ops
except (ImportError, ModuleNotFoundError):
    import vllm_xpu_kernels._xpu_C  # noqa: F401
    import tests.register_ops as ops

PAD_ID = -1


def _ref_compute_slot_mapping(query_start_loc, positions, block_table,
                              max_num_tokens, block_size, pad_id):
    """Pure-PyTorch reference (no context parallelism)."""
    num_reqs = query_start_loc.shape[0] - 1
    num_tokens = positions.shape[0]
    slot_mapping = torch.full((max_num_tokens, ),
                              pad_id,
                              dtype=torch.int64,
                              device=positions.device)
    for req_idx in range(num_reqs):
        start = int(query_start_loc[req_idx].item())
        end = int(query_start_loc[req_idx + 1].item())
        for i in range(start, end):
            pos = int(positions[i].item())
            block_index = pos // block_size
            block_number = int(block_table[req_idx, block_index].item())
            slot_mapping[i] = block_number * block_size + (pos % block_size)
    return slot_mapping


def _make_case(seq_lens, block_size, max_num_blocks_per_req, max_num_tokens,
               device, seed=0):
    torch.manual_seed(seed)
    num_reqs = len(seq_lens)
    num_tokens = sum(seq_lens)

    query_start_loc = torch.zeros(num_reqs + 1,
                                  dtype=torch.int32,
                                  device=device)
    query_start_loc[1:] = torch.tensor(seq_lens,
                                        dtype=torch.int32).cumsum(0)

    # positions: contiguous prefill positions [0, seq_len) per request.
    positions = torch.cat([
        torch.arange(sl, dtype=torch.int64) for sl in seq_lens
    ]).to(device)

    # random but distinct-ish block ids per request row.
    block_table = torch.randint(0,
                                10000,
                                (num_reqs, max_num_blocks_per_req),
                                dtype=torch.int32,
                                device=device)

    slot_mapping = torch.empty(max_num_tokens,
                               dtype=torch.int64,
                               device=device)
    return query_start_loc, positions, block_table, slot_mapping, num_tokens


@pytest.mark.parametrize("device", ["xpu"])
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [1],
        [8],
        [3500],
        [128, 256, 64],
        [1, 1, 1, 1],
        [3500, 1500],
    ],
    ids=["1", "8", "3500", "multi", "decode4", "two_long"],
)
@pytest.mark.parametrize("pad", [0, 37, 512], ids=["nopad", "pad37", "pad512"])
def test_compute_slot_mapping(device, block_size, seq_lens, pad):
    num_tokens = sum(seq_lens)
    max_num_tokens = num_tokens + pad
    max_pos = max(seq_lens)
    max_num_blocks_per_req = max_pos // block_size + 2

    (query_start_loc, positions, block_table, slot_mapping,
     n) = _make_case(seq_lens, block_size, max_num_blocks_per_req,
                     max_num_tokens, device)

    ops.compute_slot_mapping(query_start_loc, positions, block_table,
                             slot_mapping, block_size, PAD_ID)

    ref = _ref_compute_slot_mapping(query_start_loc, positions, block_table,
                                    max_num_tokens, block_size, PAD_ID)

    torch.testing.assert_close(slot_mapping, ref)
