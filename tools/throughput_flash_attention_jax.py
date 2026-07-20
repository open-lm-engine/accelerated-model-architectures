# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import time

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as splash_mask
from tabulate import tabulate


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

peak_flops = 918e12

# splash attention's causal structure is a static per-head mask baked into the kernel at construction time
# (unlike flash_attention's `causal=` flag), and batching over B is done via jax.vmap rather than a leading
# batch dim the kernel understands natively -- built once, outside the timed loop
_splash_mask = splash_mask.MultiHeadMask([splash_mask.CausalMask(shape=(S, S)) for _ in range(N)])
_splash_fn = jax.vmap(splash_kernel.make_splash_mha_single_device(mask=_splash_mask))

kernels = [
    ("flash_attention", lambda q, k, v: flash_attention(q, k, v, causal=causal, sm_scale=1.0)),
    ("splash_attention", _splash_fn),
]

mfu_table = []
bw_table = []

for row_header, fn in kernels:
    mfu_row = [row_header]
    bw_row = [row_header]

    if jax.default_backend() != "tpu":
        mfu_row.extend(["NA"] * len(dtypes))
        bw_row.extend(["NA"] * len(dtypes))
        mfu_table.append(mfu_row)
        bw_table.append(bw_row)
        continue

    for dtype in dtypes:
        jax_dtype = getattr(jnp, dtype)
        itemsize = jnp.dtype(jax_dtype).itemsize

        key_q, key_k, key_v, key_do = jax.random.split(jax.random.PRNGKey(0), 4)
        # (batch, num_heads, seq_len, head_dim) for both kernels (splash_attention via the jax.vmap above)
        q = jax.random.normal(key_q, (B, N, S, D), dtype=jnp.float32).astype(jax_dtype)
        k = jax.random.normal(key_k, (B, N, S, D), dtype=jnp.float32).astype(jax_dtype)
        v = jax.random.normal(key_v, (B, N, S, D), dtype=jnp.float32).astype(jax_dtype)

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

        if dtype == "bfloat16":
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
