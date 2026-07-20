# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import sys
import time

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
from tabulate import tabulate


# peak dense-matmul bf16 FLOPs/sec per chip, used only to compute MFU (fp32 MXU throughput isn't a well defined
# fraction of this across TPU generations, so MFU is only reported for bfloat16). Matched by substring since
# jax.devices()[0].device_kind spells generations inconsistently (e.g. "TPU v5 lite" for v5e, "TPU v5" for v5p).
_TPU_PEAK_BF16_FLOPS = [
    ("v6e", 918e12),
    ("v5e", 197e12),
    ("v5 lite", 197e12),
    ("v5p", 459e12),
    ("v5", 459e12),
    ("v4", 275e12),
    ("v3", 123e12),
]


def _get_peak_bf16_flops() -> float | None:
    if jax.default_backend() != "tpu":
        return None

    device_kind = jax.devices()[0].device_kind
    kind = device_kind.lower()

    for name, peak_flops in _TPU_PEAK_BF16_FLOPS:
        if name in kind:
            return peak_flops

    print(f"WARNING: unrecognized TPU device_kind {device_kind!r}, MFU cannot be computed", file=sys.stderr)
    return None


n = 100

dtypes = ["float32", "bfloat16"]
headers = ["kernel"] + dtypes

B = 4
S = 4096
N = 32
D = 128
causal = True

run_forward = False

# dense-attention FLOP count: qk^T (2*B*S^2*N*D) + attn @ v (2*B*S^2*N*D); causal masking computes roughly half
# the (row, col) pairs, so the achieved-FLOPs count is halved to match the actual work a causal kernel does
f = 4 * B * S**2 * N * D
if causal:
    f //= 2
flops = f if run_forward else 2.5 * f  # standard fwd:bwd ~= 1:2.5 ratio for flash-attention-style kernels

# fwd: read q, k, v, write o
bytes_forward_elements = 4 * B * S * N * D
# bwd: read q, k, v, o, do, write dq, dk, dv
bytes_backward_elements = 8 * B * S * N * D
bytes_elements = bytes_forward_elements if run_forward else bytes_backward_elements

peak_flops = _get_peak_bf16_flops()

mfu_row = ["flash_attention"]
bw_row = ["flash_attention"]

if jax.default_backend() != "tpu":
    mfu_row.extend(["NA"] * len(dtypes))
    bw_row.extend(["NA"] * len(dtypes))
else:
    for dtype in dtypes:
        jax_dtype = getattr(jnp, dtype)
        itemsize = jnp.dtype(jax_dtype).itemsize

        key_q, key_k, key_v, key_do = jax.random.split(jax.random.PRNGKey(0), 4)
        # jax.experimental.pallas.ops.tpu.flash_attention expects (batch, num_heads, seq_len, head_dim)
        q = jax.random.normal(key_q, (B, N, S, D), dtype=jnp.float32).astype(jax_dtype)
        k = jax.random.normal(key_k, (B, N, S, D), dtype=jnp.float32).astype(jax_dtype)
        v = jax.random.normal(key_v, (B, N, S, D), dtype=jnp.float32).astype(jax_dtype)

        fn = lambda q, k, v: flash_attention(q, k, v, causal=causal, sm_scale=1.0)

        if run_forward:
            forward = jax.jit(fn)
            run = lambda: forward(q, k, v)
        else:
            do = jax.random.normal(key_do, (B, N, S, D), dtype=jnp.float32).astype(jax_dtype)

            # compute the primal once and reuse its residuals so every timed iteration only re-runs the
            # backward (pullback), mirroring how torch.autograd.grad(..., retain_graph=True) is used elsewhere
            _, vjp_fn = jax.vjp(fn, q, k, v)
            backward = jax.jit(vjp_fn)
            run = lambda: backward(do)

        jax.block_until_ready(run())

        start = time.perf_counter()
        for _ in range(n):
            output = run()
        jax.block_until_ready(output)
        end = time.perf_counter()

        t = (end - start) / n

        if peak_flops is not None and dtype == "bfloat16":
            mfu_row.append(100 * flops / t / peak_flops)
        else:
            mfu_row.append("NA")

        bw_row.append(bytes_elements * itemsize / t / 1e12)


print("MFU (%)")
print(tabulate([mfu_row], headers=headers))
print()
print("Bandwidth (TB/s)")
print(tabulate([bw_row], headers=headers))
