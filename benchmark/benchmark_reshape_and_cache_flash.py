# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import itertools
import random
import time
from typing import Optional

import torch
from tabulate import tabulate
from torch import Tensor

from tests import register_ops as vllm_ops
from tests.utils import (check_ipex_availability,
                         create_kv_caches_with_random_flash, parse_args)

HAS_IPEX = check_ipex_availability()

if HAS_IPEX:
    import intel_extension_for_pytorch as ipex


def reshape_and_cache_flash_vllm(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    k_scale: Optional[Tensor] = None,
    v_scale: Optional[Tensor] = None,
) -> None:
    """vLLM's fused kernel for reshaping and caching K/V tensors (flash layout)."""
    vllm_ops.reshape_and_cache_flash(key, value, key_cache, value_cache,
                                     slot_mapping, kv_cache_dtype, k_scale,
                                     v_scale)


def reshape_and_cache_flash_ipex(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    k_scale: Optional[Tensor] = None,
    v_scale: Optional[Tensor] = None,
) -> None:
    """IPEX native implementation using ipex.llm.modules.PagedAttention."""
    if not HAS_IPEX:
        raise RuntimeError("IPEX is not available")
    assert kv_cache_dtype == "auto", \
        "IPEX reshape_and_cache_flash uses 'auto' mode"

    ipex.llm.modules.PagedAttention.reshape_and_cache_flash(
        key, value, key_cache, value_cache, slot_mapping)


def calculate_diff(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    kv_cache_dtype: str = "auto",
    device: str = "xpu",
) -> None:
    """Compare vLLM and IPEX outputs for correctness."""
    if kv_cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            "fp8 kv-cache requires head_size to be a multiple of 16.")

    torch.manual_seed(42)
    torch.set_default_device(device)

    key = torch.randn(num_tokens, num_heads, head_size, dtype=dtype,
                      device=device)
    value = torch.randn_like(key)

    num_slots = block_size * num_blocks
    slot_mapping_lst = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long,
                                device=device)

    num_layers = 1
    key_caches_vllm, value_caches_vllm = create_kv_caches_with_random_flash(
        num_blocks, block_size, num_layers, num_heads, head_size,
        kv_cache_dtype, dtype, device=device)
    key_cache_vllm = key_caches_vllm[0].clone()
    value_cache_vllm = value_caches_vllm[0].clone()

    k_scale = (key.amax() / 64.0).to(torch.float32)
    v_scale = (value.amax() / 64.0).to(torch.float32)

    reshape_and_cache_flash_vllm(key, value, key_cache_vllm, value_cache_vllm,
                                 slot_mapping, kv_cache_dtype, k_scale, v_scale)

    if HAS_IPEX:
        try:
            key_caches_ipex, value_caches_ipex = \
                create_kv_caches_with_random_flash(
                    num_blocks, block_size, num_layers, num_heads, head_size,
                    kv_cache_dtype, dtype, device=device)
            key_cache_ipex = key_caches_ipex[0].clone()
            value_cache_ipex = value_caches_ipex[0].clone()

            reshape_and_cache_flash_ipex(key, value, key_cache_ipex,
                                         value_cache_ipex, slot_mapping,
                                         kv_cache_dtype, k_scale, v_scale)

            if torch.allclose(key_cache_vllm.float(), key_cache_ipex.float(),
                              atol=1e-2, rtol=1e-2):
                print("✅ IPEX key_cache matches vLLM")
            else:
                print("❌ IPEX key_cache differs from vLLM")

            if torch.allclose(value_cache_vllm.float(),
                              value_cache_ipex.float(), atol=1e-2, rtol=1e-2):
                print("✅ IPEX value_cache matches vLLM")
            else:
                print("❌ IPEX value_cache differs from vLLM")
        except Exception as e:
            print(f"❌ IPEX implementation failed: {e}")
    else:
        print("⚠️  IPEX not available, skipping correctness check")


@torch.inference_mode()
def run_benchmark(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    num_iters: int,
    device: str = "xpu",
) -> dict:
    """Return latencies (seconds) for vLLM and IPEX implementations."""

    if kv_cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            "fp8 kv-cache requires head_size to be a multiple of 16.")

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_device(device)

    key = torch.randn(num_tokens, num_heads, head_size, dtype=dtype,
                      device=device)
    value = torch.randn_like(key)

    num_slots = block_size * num_blocks
    if num_tokens > num_slots:
        raise ValueError(
            "num_tokens cannot exceed the total number of cache slots")
    slot_mapping_lst = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long,
                                device=device)

    num_layers = 1
    key_caches, value_caches = create_kv_caches_with_random_flash(
        num_blocks, block_size, num_layers, num_heads, head_size,
        kv_cache_dtype, dtype, device=device)
    key_cache, value_cache = key_caches[0], value_caches[0]

    k_scale = (key.amax() / 64.0).to(torch.float32)
    v_scale = (value.amax() / 64.0).to(torch.float32)

    def time_fn(fn, n_iters: int) -> float:
        torch.xpu.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            fn(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype,
               k_scale, v_scale)
        torch.xpu.synchronize()
        return (time.perf_counter() - start) / n_iters

    results = {}

    # vLLM benchmark
    time_fn(reshape_and_cache_flash_vllm, 3)  # warm up
    results["vllm"] = time_fn(reshape_and_cache_flash_vllm, num_iters)

    # IPEX benchmark
    if HAS_IPEX and kv_cache_dtype == "auto":
        try:
            time_fn(reshape_and_cache_flash_ipex, 3)  # warm up
            results["ipex"] = time_fn(reshape_and_cache_flash_ipex, num_iters)
        except Exception as e:
            print(f"⚠️  IPEX benchmark failed for config ({num_tokens}, "
                  f"{num_heads}, {head_size}, {block_size}, {num_blocks}): "
                  f"{e}")

    del key, value, key_cache, value_cache, slot_mapping
    torch.xpu.empty_cache()

    return results


if __name__ == "__main__":
    args = parse_args()

    # TODO: to refine the device detect in the future
    device = "xpu"

    print("Benchmark Configuration:")
    print(f"  Num Heads: {args.head_num_range}")
    print(f"  Head Size: {args.head_size}")
    print(f"  Block Size: {args.block_size}")
    print(f"  Num Blocks: {args.num_blocks}")
    print(f"  Data Type: {args.dtype}")
    print(f"  KV Cache Dtype: {args.kv_cache_dtype}")
    print(f"  Device: {device}")
    if HAS_IPEX:
        print(f"✅ IPEX {ipex.__version__} is available.")
    else:
        print("⚠️ IPEX not available. Only benchmarking vLLM.")

    num_token_range = [2**i for i in range(1, 12)]
    head_num_range = args.head_num_range
    head_size_range = [args.head_size]
    block_size_range = [args.block_size]
    num_blocks_range = [args.num_blocks]
    configs = list(
        itertools.product(num_token_range, head_num_range, head_size_range,
                          block_size_range, num_blocks_range))

    # Run correctness check
    calculate_diff(
        num_tokens=4,
        num_heads=args.head_num_range[0],
        head_size=args.head_size,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        device=device,
    )

    headers = [
        "num_tokens", "num_heads", "head_size", "block_size", "num_blocks",
        "dtype", "kv_cache_dtype", "vllm (us)"
    ]
    if HAS_IPEX:
        headers.append("ipex (us)")

    rows = []
    for num_tokens, num_heads, head_size, block_size, num_blocks in configs:
        results = run_benchmark(
            num_tokens=num_tokens,
            num_heads=num_heads,
            head_size=head_size,
            block_size=block_size,
            num_blocks=num_blocks,
            dtype=args.dtype,
            kv_cache_dtype=args.kv_cache_dtype,
            num_iters=100,
            device=device,
        )
        row = [
            num_tokens, num_heads, head_size, block_size, num_blocks,
            str(args.dtype), args.kv_cache_dtype,
            f"{results['vllm'] * 1e6:.3f}",
        ]
        if HAS_IPEX:
            row.append(f"{results['ipex'] * 1e6:.3f}"
                       if "ipex" in results else "N/A")
        rows.append(row)

    print(tabulate(rows, headers=headers))
