# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import itertools
import random
from typing import Optional

import torch
import triton
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


def get_benchmark(
    dtype: torch.dtype,
    kv_cache_dtype: str = "auto",
    device: str = "xpu",
):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=[
                "num_tokens", "num_heads", "head_size", "block_size",
                "num_blocks"
            ],
            x_vals=configs,
            line_arg="provider",
            line_vals=["vllm", "ipex"] if HAS_IPEX else ["vllm"],
            line_names=["vLLM", "IPEX"] if HAS_IPEX else ["vLLM"],
            styles=[("blue", "-"),
                    ("red", "-")] if HAS_IPEX else [("blue", "-")],
            ylabel="latency (us)",
            plot_name="reshape_and_cache_flash-benchmark",
            args={},
        ))
    @torch.inference_mode()
    def benchmark(num_tokens, num_heads, head_size, block_size, num_blocks,
                  provider):

        if kv_cache_dtype == "fp8" and head_size % 16:
            raise ValueError(
                "fp8 kv-cache requires head_size to be a multiple of 16.")

        torch.manual_seed(42)
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

        torch.xpu.synchronize()
        # Warm up
        for _ in range(5):
            if provider == "vllm":
                reshape_and_cache_flash_vllm(key, value, key_cache,
                                             value_cache, slot_mapping,
                                             kv_cache_dtype, k_scale, v_scale)
            elif provider == "ipex" and HAS_IPEX:
                reshape_and_cache_flash_ipex(key, value, key_cache,
                                             value_cache, slot_mapping,
                                             kv_cache_dtype, k_scale, v_scale)

        # Benchmark
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: {
                "vllm": reshape_and_cache_flash_vllm,
                "ipex": reshape_and_cache_flash_ipex,
            }[provider](
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
                v_scale,
            ),
            quantiles=quantiles,
        )
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


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

    benchmark = get_benchmark(
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        device=device,
    )
    benchmark.run(print_data=True, save_path=None)
