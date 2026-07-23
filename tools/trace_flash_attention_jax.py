# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as splash_mask


LOG_DIR = "tmp"
NUM_ITERATIONS = 10

dtype = jnp.bfloat16

B = 4
S = 4096
N = 32
D = 128
causal = True

assert jax.default_backend() == "tpu", "this trace targets TPU pallas/mosaic kernels"

key_q, key_k, key_v, key_do = jax.random.split(jax.random.PRNGKey(0), 4)
q = jax.random.normal(key_q, (B, N, S, D), dtype=jnp.float32).astype(dtype)
k = jax.random.normal(key_k, (B, N, S, D), dtype=jnp.float32).astype(dtype)
v = jax.random.normal(key_v, (B, N, S, D), dtype=jnp.float32).astype(dtype)
do = jax.random.normal(key_do, (B, N, S, D), dtype=jnp.float32).astype(dtype)

# splash attention's causal structure is a static per-head mask baked into the kernel at construction time,
# and batching over B is done via jax.vmap (see tools/throughput_flash_attention_jax.py for details)
_splash_mask = splash_mask.MultiHeadMask([splash_mask.CausalMask(shape=(S, S)) for _ in range(N)])
_splash_fn = jax.vmap(splash_kernel.make_splash_mha_single_device(mask=_splash_mask))

kernels = {
    "flash_attention": lambda q, k, v: flash_attention(q, k, v, causal=causal, sm_scale=1.0),
    "splash_attention": _splash_fn,
}

backward_fns = {}
for name, fn in kernels.items():
    _, vjp_fn = jax.vjp(fn, q, k, v)
    backward_fns[name] = jax.jit(vjp_fn)

    # warm up so the trace captures steady-state execution, not the one-time jit compilation
    jax.block_until_ready(backward_fns[name](do))

with jax.profiler.trace(LOG_DIR, create_perfetto_trace=True):
    for name, backward in backward_fns.items():
        for _ in range(NUM_ITERATIONS):
            output = backward(do)
        jax.block_until_ready(output)

print(f"trace written to {LOG_DIR}")
print(f"find a *.trace.json.gz under {LOG_DIR}/plugins/profile/<timestamp>/")
print("view it by either:")
print("  - opening chrome://tracing in Chrome and loading the (gunzipped) .json file, or")
print("  - dragging the .trace.json.gz directly onto https://ui.perfetto.dev")
print()
print("flash_attention and splash_attention each ran back to back in this trace -- compare the gaps between")
print("dispatched kernel instances (dispatch/queueing overhead) against each kernel's own execution duration")
print("(actual TPU compute time) to see whether wall-clock time is dominated by the kernel or by overhead")
