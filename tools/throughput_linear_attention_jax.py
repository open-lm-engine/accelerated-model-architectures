# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import time

import jax
import jax.numpy as jnp
from tabulate import tabulate

from xma import KernelBackend, linear_attention_jax


# peak dense-matmul bf16 FLOPs/sec per chip, used only to compute MFU (fp32 MXU throughput isn't a well defined
# fraction of this across TPU generations, so MFU is only reported for bfloat16)
_TPU_PEAK_BF16_FLOPS = {
    "TPU v3": 123e12,
    "TPU v4": 275e12,
    "TPU v5e": 197e12,
    "TPU v5p": 459e12,
    "TPU v6e": 918e12,
}

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

# recurrent-form FLOPs: per token, q @ h (2KV) + state update k^T (x) v (2KV)
flops_forward = B * N * S * 4 * K * V
# backward's compute-dominant term (BLOCK_SIZE_S^2 intra-chunk term is negligible next to BLOCK_SIZE_S * K * V
# when K, V >> 1) is 10 * BLOCK_SIZE_S * K * V per chunk vs forward's 4 * BLOCK_SIZE_S * K * V, i.e. ~2.5x
flops = flops_forward if run_forward else 2.5 * flops_forward

# forward: read q, k, v, write y, h_out
bytes_forward_elements = 3 * B * S * N * K + B * S * N * V + B * N * K * V
# backward: read q, k, v, dy, dh_out cotangent, write dq, dk, dv, dh0
bytes_backward_elements = 4 * B * S * N * K + B * N * K * V + 3 * B * S * N * K + B * N * K * V
bytes_elements = bytes_forward_elements if run_forward else bytes_backward_elements

peak_flops = None
if jax.default_backend() == "tpu":
    peak_flops = _TPU_PEAK_BF16_FLOPS.get(jax.devices()[0].device_kind)

mfu_table = []
bw_table = []

for kernel_backend, row_header in kernels:
    mfu_row = [row_header]
    bw_row = [row_header]

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
