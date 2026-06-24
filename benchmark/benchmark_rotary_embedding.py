# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import torch
import triton

from tests import register_ops as vllm_ops
from tests.ops.rotary_embedding_op import RotaryEmbedding, apply_rotary_emb_torch
from tests.utils import check_ipex_availability, parse_args

HAS_IPEX = check_ipex_availability()

if HAS_IPEX:
    import intel_extension_for_pytorch as ipex


def _make_rotary_embedding(head_size: int,
                           rotary_dim: int,
                           max_position: int,
                           base: float,
                           is_neox_style: bool,
                           dtype: torch.dtype) -> RotaryEmbedding:
    rot = RotaryEmbedding(head_size, rotary_dim, max_position, base,
                          is_neox_style, dtype)
    return rot


def rotary_embedding_naive(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    rot: RotaryEmbedding,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch (native) rotary embedding implementation."""
    return rot.forward_native(positions, query.clone(), key.clone())


@torch.compile
def _rotary_emb_compile(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions_flat = positions.flatten()
    num_tokens = positions_flat.shape[0]
    cos_sin = cos_sin_cache.index_select(0, positions_flat)
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_shape = query.shape
    query = query.view(num_tokens, -1, head_size)
    query_rot = query[..., :rotary_dim]
    query_pass = query[..., rotary_dim:]
    query_rot = apply_rotary_emb_torch(query_rot, cos, sin, is_neox_style)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.view(num_tokens, -1, head_size)
    key_rot = key[..., :rotary_dim]
    key_pass = key[..., rotary_dim:]
    key_rot = apply_rotary_emb_torch(key_rot, cos, sin, is_neox_style)
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

    return query, key


def rotary_embedding_compile(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    rot: RotaryEmbedding,
) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.compile rotary embedding implementation."""
    cos_sin_cache = rot.cos_sin_cache.to(query.device, dtype=query.dtype)
    return _rotary_emb_compile(positions, query.clone(), key.clone(),
                               cos_sin_cache, rot.head_size, rot.rotary_dim,
                               rot.is_neox_style)


def rotary_embedding_vllm(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    rot: RotaryEmbedding,
) -> tuple[torch.Tensor, torch.Tensor]:
    """vllm-xpu-kernels rotary embedding implementation (in-place)."""
    q = query.clone()
    k = key.clone()
    cos_sin_cache = rot.cos_sin_cache.to(q.device, dtype=q.dtype)
    vllm_ops.rotary_embedding(positions, q, k, rot.head_size, cos_sin_cache,
                              rot.is_neox_style)
    return q, k


def rotary_embedding_ipex(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    rot: RotaryEmbedding,
) -> tuple[torch.Tensor, torch.Tensor]:
    """IPEX rotary embedding implementation."""
    if not HAS_IPEX:
        raise RuntimeError("IPEX is not available")

    q = query.clone()
    k = key.clone()
    cos_sin_cache = rot.cos_sin_cache.to(q.device, dtype=q.dtype)

    # IPEX apply_rotary_embedding operates in-place on query/key
    # Signature: apply_rotary_embedding(q, k, sin, cos, positions)
    # The cos_sin_cache layout is [max_pos, rotary_dim//2 * 2] where
    # the first half is cos and the second half is sin.
    cos_sin = cos_sin_cache.index_select(0, positions.flatten())
    cos, sin = cos_sin.chunk(2, dim=-1)

    # IPEX llm functional rotary embedding
    if hasattr(ipex.llm.functional, 'rotary_embedding'):
        # ipex.llm.functional.rotary_embedding(q, k, sin, cos, rotary_dim,
        #                                       num_concats)
        num_concats = 1 if rot.is_neox_style else 0
        ipex.llm.functional.rotary_embedding(
            q, k, sin, cos, rot.rotary_dim, num_concats
        )
    elif hasattr(ipex.llm.functional, 'apply_rotary_embedding'):
        ipex.llm.functional.apply_rotary_embedding(q, k, sin, cos, positions)
    else:
        # Fallback: use the native torch implementation
        positions_flat = positions.flatten()
        num_tokens = positions_flat.shape[0]
        cos_s = cos_sin_cache.index_select(0, positions_flat)
        cos_v, sin_v = cos_s.chunk(2, dim=-1)

        q_shape = q.shape
        q = q.view(num_tokens, -1, rot.head_size)
        q_rot = q[..., :rot.rotary_dim]
        q_pass = q[..., rot.rotary_dim:]
        q_rot = apply_rotary_emb_torch(q_rot, cos_v, sin_v, rot.is_neox_style)
        q = torch.cat((q_rot, q_pass), dim=-1).reshape(q_shape)

        k_shape = k.shape
        k = k.view(num_tokens, -1, rot.head_size)
        k_rot = k[..., :rot.rotary_dim]
        k_pass = k[..., rot.rotary_dim:]
        k_rot = apply_rotary_emb_torch(k_rot, cos_v, sin_v, rot.is_neox_style)
        k = torch.cat((k_rot, k_pass), dim=-1).reshape(k_shape)

    return q, k


def calculate_diff(batch_size: int,
                   seq_len: int,
                   num_heads: int,
                   head_size: int,
                   rotary_dim: int,
                   max_position: int,
                   is_neox_style: bool,
                   dtype: torch.dtype = torch.bfloat16) -> None:
    base = 10000
    rot = _make_rotary_embedding(head_size, rotary_dim, max_position, base,
                                 is_neox_style, dtype)
    rot = rot.to("xpu")

    positions = torch.randint(0, max_position, (batch_size, seq_len),
                              device="xpu")
    query = torch.randn(batch_size, seq_len, num_heads, head_size,
                        dtype=dtype, device="xpu")
    key = torch.randn_like(query)

    out_naive_q, out_naive_k = rotary_embedding_naive(positions, query, key,
                                                      rot)
    out_vllm_q, out_vllm_k = rotary_embedding_vllm(positions, query, key, rot)

    print(f"Naive  query[:1,:1,0,:4] = {out_naive_q[0, 0, 0, :4]}")
    print(f"vLLM   query[:1,:1,0,:4] = {out_vllm_q[0, 0, 0, :4]}")

    if torch.allclose(out_naive_q, out_vllm_q, atol=1e-2, rtol=1e-2) and \
       torch.allclose(out_naive_k, out_vllm_k, atol=1e-2, rtol=1e-2):
        print("✅ vLLM-XPU-Kernels implementation matches naive")
    else:
        print("❌ vLLM-XPU-Kernels implementation differs from naive")

    if HAS_IPEX:
        try:
            out_ipex_q, out_ipex_k = rotary_embedding_ipex(
                positions, query, key, rot)
            print(f"IPEX   query[:1,:1,0,:4] = {out_ipex_q[0, 0, 0, :4]}")
            if torch.allclose(out_naive_q, out_ipex_q, atol=1e-2, rtol=1e-2) \
               and torch.allclose(out_naive_k, out_ipex_k, atol=1e-2,
                                  rtol=1e-2):
                print("✅ IPEX implementation matches naive")
            else:
                print("❌ IPEX implementation differs from naive")
        except Exception as e:
            print(f"❌ IPEX implementation failed: {e}")


def get_benchmark(is_neox_style: bool, dtype: torch.dtype):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_heads", "batch_size", "seq_len"],
            x_vals=[tuple(_) for _ in configs],
            line_arg="provider",
            line_vals=["naive", "vllm", "t.compile", "ipex"]
            if HAS_IPEX else ["naive", "vllm", "t.compile"],
            line_names=["Naive", "vLLM-XPU-Kernels", "t.compile", "IPEX"]
            if HAS_IPEX else ["Naive", "vLLM-XPU-Kernels", "t.compile"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-"),
                    ("red", "-")] if HAS_IPEX else [("blue", "-"),
                                                    ("green", "-"),
                                                    ("orange", "-")],
            ylabel="us",
            plot_name=(
                f"rotary-embedding-perf-"
                f"{'neox' if is_neox_style else 'gptj'}-style"),
            args={},
        ))
    def benchmark(num_heads, batch_size, seq_len, provider):
        head_size = 128
        rotary_dim = head_size  # full rotary
        max_position = 4096
        base = 10000

        rot = _make_rotary_embedding(head_size, rotary_dim, max_position, base,
                                     is_neox_style, dtype).to("xpu")

        positions = torch.randint(0, max_position, (batch_size, seq_len),
                                  device="xpu")
        query = torch.randn(batch_size, seq_len, num_heads, head_size,
                            dtype=dtype, device="xpu")
        key = torch.randn_like(query)

        quantiles = [0.5, 0.2, 0.8]

        if provider == "naive":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rotary_embedding_naive(positions, query, key, rot),
                quantiles=quantiles,
            )
        elif provider == "t.compile":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rotary_embedding_compile(positions, query, key, rot),
                quantiles=quantiles,
            )
        elif provider == "ipex" and HAS_IPEX:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rotary_embedding_ipex(positions, query, key, rot),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rotary_embedding_vllm(positions, query, key, rot),
                quantiles=quantiles,
            )
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":

    args = parse_args()

    print("Final configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Head size: {args.head_size}")
    print(f"  Data type: {args.dtype}")

    batch_size_range = [2**i for i in range(0, 7, 2)]
    seq_length_range = [2**i for i in range(6, 10, 1)]
    head_num_range = args.head_num_range
    configs = list(
        itertools.product(head_num_range, batch_size_range, seq_length_range))

    if HAS_IPEX:
        print("✅ IPEX is available")
        print(f"IPEX version: {ipex.__version__}")
    else:
        print("⚠️  IPEX is not available, skipping IPEX benchmarks")

    # Run correctness test
    calculate_diff(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_heads=args.head_num_range[0],
        head_size=args.head_size,
        rotary_dim=args.head_size,
        max_position=4096,
        is_neox_style=True,
        dtype=args.dtype,
    )

    # Run benchmarks for both neox and gptj style
    for is_neox in [True, False]:
        benchmark = get_benchmark(is_neox, args.dtype)
        benchmark.run(print_data=True, save_path=args.save_path)
