# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import time

import jax
import jax.numpy as jnp
from tabulate import tabulate

from xma import KernelBackend, swiglu_jax


n = 100

kernels = [
    (KernelBackend.pallas, "pallas"),
    (KernelBackend.jax, "jax"),
]
dtypes = ["float32", "bfloat16"]
headers = ["kernel"] + dtypes
table = []

B = 16 * 4096
H = 4096

run_forward = False

for kernel_backend, row_header in kernels:
    row = [row_header]

    if kernel_backend == KernelBackend.pallas and not kernel_backend.verify_accelerator():
        row.extend(["NA"] * len(dtypes))
        table.append(row)
        continue

    for dtype in dtypes:
        jax_dtype = getattr(jnp, dtype)
        itemsize = jnp.dtype(jax_dtype).itemsize

        key_gate, key_up, key_dy = jax.random.split(jax.random.PRNGKey(0), 3)
        gate = jax.random.normal(key_gate, (B, H), dtype=jnp.float32).astype(jax_dtype)
        up = jax.random.normal(key_up, (B, H), dtype=jnp.float32).astype(jax_dtype)

        fn = lambda gate, up: swiglu_jax(gate, up, kernel_backend=kernel_backend)

        if run_forward:
            forward = jax.jit(fn)
            run = lambda: forward(gate, up)
        else:
            dy = jax.random.normal(key_dy, (B, H), dtype=jnp.float32).astype(jax_dtype)

            # compute the primal once and reuse its residuals so every timed
            # iteration only re-runs the backward (pullback), mirroring how
            # torch.autograd.grad(..., retain_graph=True) is used elsewhere
            _, vjp_fn = jax.vjp(fn, gate, up)
            backward = jax.jit(vjp_fn)
            run = lambda: backward(dy)

        jax.block_until_ready(run())

        start = time.perf_counter()
        for _ in range(n):
            output = run()
        jax.block_until_ready(output)
        end = time.perf_counter()

        t = (end - start) / n
        row.append((3 if run_forward else 5) * B * H * itemsize / t / 1e12)

    table.append(row)


print(tabulate(table, headers=headers))
