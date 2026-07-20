# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import jax
import jax.numpy as jnp

from xma import KernelBackend
from xma.layers_jax.linear_attention.op import linear_attention_jax


LOG_DIR = "tmp"
NUM_ITERATIONS = 10

kernel_backend = KernelBackend.pallas
dtype = jnp.bfloat16

B = 4
S = 4096
N = 32
K = 128
V = 128
BLOCK_SIZE_S = 64

if kernel_backend == KernelBackend.pallas:
    assert jax.default_backend() == "tpu", "KernelBackend.pallas is only supported on TPU"

key_q, key_k, key_v = jax.random.split(jax.random.PRNGKey(0), 3)
q = jax.random.normal(key_q, (B, S, N, K), dtype=jnp.float32).astype(dtype)
k = jax.random.normal(key_k, (B, S, N, K), dtype=jnp.float32).astype(dtype)
v = jax.random.normal(key_v, (B, S, N, V), dtype=jnp.float32).astype(dtype)

fn = jax.jit(
    lambda q, k, v: linear_attention_jax(q, k, v, None, BLOCK_SIZE_S=BLOCK_SIZE_S, kernel_backend=kernel_backend)
)

# warm up so the trace captures steady-state execution, not the one-time jit compilation
jax.block_until_ready(fn(q, k, v))

with jax.profiler.trace(LOG_DIR, create_perfetto_trace=True):
    for _ in range(NUM_ITERATIONS):
        output = fn(q, k, v)
    jax.block_until_ready(output)

print(f"trace written to {LOG_DIR}")
print(f"find a *.trace.json.gz under {LOG_DIR}/plugins/profile/<timestamp>/")
print("view it by either:")
print("  - opening chrome://tracing in Chrome and loading the (gunzipped) .json file, or")
print("  - dragging the .trace.json.gz directly onto https://ui.perfetto.dev")
