# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import sys
import time

import jax
import jax.numpy as jnp
from tabulate import tabulate

from xma import KernelBackend, linear_attention_jax


# peak dense-matmul bf16 FLOPs/sec per chip, used only to compute MFU (fp32 MXU throughput isn't a well defined
# fraction of this across TPU generations, so MFU is only reported for bfloat16). Matched by substring since
# jax.devices()[0].device_kind spells generations inconsistently (e.g. "TPU v5 lite" for v5e, "TPU v5" for v5p).
_TPU_PEAK_BF16_FLOPS = [("TPU v6 lite", 918e12)]


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

kernels = [
    (KernelBackend.pallas, "pallas"),
    # (KernelBackend.jax, "jax"),
]
dtypes = ["float32", "bfloat16"]
headers = ["kernel"] + dtypes

B = 4
S = 4096
N = 32
K = 512
V = 512
BLOCK_SIZE_S = 512

run_forward = True

NUM_CHUNKS = -(-S // BLOCK_SIZE_S)  # ceil division

# reference ("jax") path is a strict per-token recurrence: q @ h (2KV) + state update k^T (x) v (2KV)
flops_forward_jax = B * N * S * 4 * K * V
# pallas path additionally does an intra-chunk causal matmul per chunk (qk: 2*BLOCK^2*K, qk@v: 2*BLOCK^2*V) on
# top of the same q @ h / state update terms above, real compute that's only negligible when BLOCK_SIZE_S << K, V
flops_forward_pallas = B * N * (NUM_CHUNKS * 2 * BLOCK_SIZE_S**2 * (K + V) + S * 4 * K * V)
# backward redoes the forward-checkpoint pass (same cost as forward) plus, per chunk: qk, d_masked_qk, dq's
# intra-chunk term, dk's intra-chunk term (4 * 2*BLOCK^2*K-or-V terms) and 4 BLOCK*K*V cross-chunk terms
flops_backward_pallas = flops_forward_pallas + B * N * (
    NUM_CHUNKS * 2 * BLOCK_SIZE_S**2 * (3 * K + 2 * V) + S * 8 * K * V
)
flops_backward_jax = 2 * flops_forward_jax  # standard backward ~= 2x forward heuristic for autodiff'd code

# forward: read q, k, v, write y, h_out
bytes_forward_elements = 3 * B * S * N * K + B * S * N * V + B * N * K * V
# backward: read q, k, v, dy, dh_out cotangent, write dq, dk, dv, dh0
bytes_backward_elements = 4 * B * S * N * K + B * N * K * V + 3 * B * S * N * K + B * N * K * V
bytes_elements = bytes_forward_elements if run_forward else bytes_backward_elements

peak_flops = _get_peak_bf16_flops()

mfu_table = []
bw_table = []

for kernel_backend, row_header in kernels:
    mfu_row = [row_header]
    bw_row = [row_header]

    if kernel_backend == KernelBackend.pallas:
        flops = flops_forward_pallas if run_forward else flops_backward_pallas
    else:
        flops = flops_forward_jax if run_forward else flops_backward_jax

    if kernel_backend == KernelBackend.pallas and not kernel_backend.verify_accelerator():
        mfu_row.extend(["NA"] * len(dtypes))
        bw_row.extend(["NA"] * len(dtypes))
        mfu_table.append(mfu_row)
        bw_table.append(bw_row)
        continue

    for dtype in dtypes:
        jax_dtype = getattr(jnp, dtype)
        itemsize = jnp.dtype(jax_dtype).itemsize

        key_q, key_k, key_v, key_dy, key_dht = jax.random.split(jax.random.PRNGKey(0), 5)
        q = jax.random.normal(key_q, (B, S, N, K), dtype=jnp.float32).astype(jax_dtype)
        k = jax.random.normal(key_k, (B, S, N, K), dtype=jnp.float32).astype(jax_dtype)
        v = jax.random.normal(key_v, (B, S, N, V), dtype=jnp.float32).astype(jax_dtype)

        fn = lambda q, k, v: linear_attention_jax(
            q, k, v, None, BLOCK_SIZE_S=BLOCK_SIZE_S, kernel_backend=kernel_backend
        )

        if run_forward:
            forward = jax.jit(fn)
            run = lambda: forward(q, k, v)
        else:
            dy = jax.random.normal(key_dy, (B, S, N, V), dtype=jnp.float32).astype(jax_dtype)
            dht = jax.random.normal(key_dht, (B, N, K, V), dtype=jnp.float32)

            # compute the primal once and reuse its residuals so every timed iteration only re-runs the
            # backward (pullback), mirroring how torch.autograd.grad(..., retain_graph=True) is used elsewhere
            (_, ht), vjp_fn = jax.vjp(fn, q, k, v)
            backward = jax.jit(vjp_fn)
            run = lambda: backward((dy, dht))

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

    mfu_table.append(mfu_row)
    bw_table.append(bw_row)


print("MFU (%)")
print(tabulate(mfu_table, headers=headers))
print()
print("Bandwidth (TB/s)")
print(tabulate(bw_table, headers=headers))
