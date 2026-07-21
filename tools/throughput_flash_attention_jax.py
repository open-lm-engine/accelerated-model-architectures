# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import re
import time

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as splash_mask
from tabulate import tabulate


def _print_compiled_info(jitted_fn, name: str, *args) -> None:
    # inspect the *actual* compiled program instead of just trusting timing numbers: this both proves the
    # jitted function only compiles once here (the timing loop's own warmup call below reuses this exact
    # cache entry, since .lower(...).compile() populates the same jit cache keyed by arg avals) and confirms
    # what's really running on device -- e.g. that a genuine Mosaic/pallas custom-call is being dispatched
    # (not some silent fallback), and, where available, the compiled VMEM footprint.
    text = jitted_fn.lower(*args).compile().as_text()

    custom_call_targets = sorted(set(re.findall(r'custom_call_target="([^"]+)"', text)))
    print(f"  [{name}] custom-call targets: {custom_call_targets}")

    vmem_match = re.search(r'"used_scoped_memory_configs":\[\{"memory_space":"1",.*?"size":"(\d+)"', text)
    if vmem_match:
        print(f"  [{name}] VMEM usage: {int(vmem_match.group(1)) / 1024:.1f} KiB")
    else:
        print(f"  [{name}] VMEM usage: not found in compiled text")


n = 100

# Run both phases so the result is not accidentally read as a forward-attention
# number.  The backward phase is a VJP with a precomputed forward residual.
run_forward = True
run_backward = True

dtypes = ["float32", "bfloat16"]
headers = ["kernel"] + dtypes

B = 4
S = 4096
N = 32
D = 128
causal = True

# dense-attention FLOP count: qk^T (2*B*S^2*N*D) + attn @ v (2*B*S^2*N*D); causal masking computes roughly half
# the (row, col) pairs, so the achieved-FLOPs count is halved to match the actual work a causal kernel does
f = 4 * B * S**2 * N * D
if causal:
    f //= 2

# fwd: read q, k, v, write o
bytes_forward_elements = 4 * B * S * N * D
# bwd: read q, k, v, o, do, write dq, dk, dv
bytes_backward_elements = 8 * B * S * N * D

# This benchmark is intended for one TPU v6e chip.  Keep this explicit rather than
# silently treating a different TPU generation as v6e.
peak_flops = 918e12

# Splash's generic defaults are correct but not necessarily fast for every shape.
# These are deliberately explicit so that the backward configuration is visible and
# can be tuned without changing the benchmark plumbing.  Fused backward combines the
# dQ work with the dKV pass; set this to False to reproduce the two-pass default.
splash_use_fused_bwd = True
splash_block_sizes = splash_kernel.BlockSizes(
    block_q=128,
    block_kv=128,
    block_kv_compute=128,
    block_q_dkv=128,
    block_kv_dkv=128,
    block_kv_dkv_compute=128,
    block_q_dq=None if splash_use_fused_bwd else 128,
    block_kv_dq=None if splash_use_fused_bwd else 128,
    use_fused_bwd_kernel=splash_use_fused_bwd,
)

# splash attention's causal structure is a static per-head mask baked into the kernel at construction time
# (unlike flash_attention's `causal=` flag), and batching over B is done via jax.vmap rather than a leading
# batch dim the kernel understands natively -- built once, outside the timed loop
_splash_mask = splash_mask.MultiHeadMask([splash_mask.CausalMask(shape=(S, S)) for _ in range(N)])
_splash_fn = jax.vmap(splash_kernel.make_splash_mha_single_device(mask=_splash_mask, block_sizes=splash_block_sizes))

kernels = [
    ("flash_attention", lambda q, k, v: flash_attention(q, k, v, causal=causal, sm_scale=1.0)),
    ("splash_attention", _splash_fn),
]

phase_specs = []
if run_forward:
    phase_specs.append(("fwd", True, f, bytes_forward_elements))
if run_backward:
    # Standard flash-attention backward is approximately 2.5x the forward FLOPs.
    phase_specs.append(("bwd", False, 2.5 * f, bytes_backward_elements))

mfu_table = []
bw_table = []
latency_table = []

print(f"JAX {jax.__version__}; backend={jax.default_backend()}; device={jax.devices()[0].device_kind}")
print(f"Splash fused backward: {splash_use_fused_bwd}")

for row_header, fn in kernels:
    if jax.default_backend() != "tpu":
        continue

    for phase_name, is_forward, flops, bytes_elements in phase_specs:
        mfu_row = [f"{row_header}/{phase_name}"]
        bw_row = [f"{row_header}/{phase_name}"]
        latency_row = [f"{row_header}/{phase_name}"]

        for dtype in dtypes:
            jax_dtype = getattr(jnp, dtype)
            itemsize = jnp.dtype(jax_dtype).itemsize

            key_q, key_k, key_v, key_do = jax.random.split(jax.random.PRNGKey(0), 4)
            # (batch, num_heads, seq_len, head_dim) for both kernels (splash_attention via the jax.vmap above)
            q = jax.random.normal(key_q, (B, N, S, D), dtype=jnp.float32).astype(jax_dtype)
            k = jax.random.normal(key_k, (B, N, S, D), dtype=jnp.float32).astype(jax_dtype)
            v = jax.random.normal(key_v, (B, N, S, D), dtype=jnp.float32).astype(jax_dtype)

            if is_forward:
                forward = jax.jit(fn)
                _print_compiled_info(forward, f"{row_header}/{dtype}/fwd", q, k, v)

                def run():
                    return forward(q, k, v)

            else:
                do = jax.random.normal(key_do, (B, N, S, D), dtype=jnp.float32).astype(jax_dtype)

                # Compute the primal once and reuse its residuals so every timed iteration only re-runs the
                # backward (pullback), mirroring how torch.autograd.grad(..., retain_graph=True) is used elsewhere.
                _, vjp_fn = jax.vjp(fn, q, k, v)
                backward = jax.jit(vjp_fn)
                _print_compiled_info(backward, f"{row_header}/{dtype}/bwd", do)

                def run():
                    return backward(do)

            jax.block_until_ready(run())

            start = time.perf_counter()
            for _ in range(n):
                output = run()
            jax.block_until_ready(output)
            end = time.perf_counter()

            t = (end - start) / n
            latency_row.append(1000 * t)

            if dtype == "bfloat16":
                mfu_row.append(100 * flops / t / peak_flops)
            else:
                mfu_row.append("NA")

            # This is logical tensor traffic, not a hardware HBM counter.
            bw_row.append(bytes_elements * itemsize / t / 1e12)

        mfu_table.append(mfu_row)
        bw_table.append(bw_row)
        latency_table.append(latency_row)


print("Latency (ms/op)")
print(tabulate(latency_table, headers=headers))
print()
print("MFU (%)")
print(tabulate(mfu_table, headers=headers))
print()
print("Effective logical bandwidth (TB/s)")
print(tabulate(bw_table, headers=headers))
