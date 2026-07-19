# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide


def _linear_attention_forward_pallas_kernel(
    q_ref, k_ref, v_ref, h0_ref, y_ref, h_ref, *, attention_multiplier: float, CHUNK_SIZE: int
) -> None:
    # h_ref is revisited (same block) across the sequential chunk grid dimension, so it doubles as the running
    # (K, V) recurrent state carried from one chunk to the next
    @pl.when(pl.program_id(2) == 0)
    def _():
        h_ref[...] = h0_ref[...]

    dtype = q_ref.dtype

    q = q_ref[...]
    k = k_ref[...]
    v = v_ref[...]
    h = h_ref[...]

    # intra-chunk contribution: strictly causal (token s excludes its own key/value, matching the sequential
    # recurrence y_s = q_s @ h_{s-1}, h_s = h_{s-1} + k_s ⊗ v_s)
    qk = jnp.dot(q, k.T, preferred_element_type=jnp.float32)
    causal_mask = jnp.tril(jnp.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=jnp.bool_), k=-1)
    qk = jnp.where(causal_mask, qk, 0).astype(dtype)

    y = jnp.dot(qk, v, preferred_element_type=jnp.float32)
    # inter-chunk contribution: state accumulated from all strictly-previous chunks
    y += jnp.dot(q, h.astype(dtype), preferred_element_type=jnp.float32)
    y = (y * attention_multiplier).astype(dtype)

    y_ref[...] = y
    h_ref[...] = h + jnp.dot(k.T.astype(jnp.float32), v.astype(jnp.float32))


@partial(jax.jit, static_argnames=("attention_multiplier", "CHUNK_SIZE"))
def _linear_attention_forward_pallas_jit(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array | None,
    attention_multiplier: float,
    CHUNK_SIZE: int,
) -> tuple[jax.Array, jax.Array]:
    B, S, Nq, K = q.shape
    Nk = k.shape[-2]
    Nv, V = v.shape[-2:]

    N = max(Nq, Nk, Nv)
    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    if h0 is None:
        h0 = jnp.zeros((B, N, K, V), dtype=jnp.float32)

    kernel = pl.pallas_call(
        partial(
            _linear_attention_forward_pallas_kernel, attention_multiplier=attention_multiplier, CHUNK_SIZE=CHUNK_SIZE
        ),
        out_shape=(
            jax.ShapeDtypeStruct(shape=(B, S, N, V), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, N, K, V), dtype=jnp.float32),
        ),
        grid=(B, N, ceil_divide(S, CHUNK_SIZE)),
        in_specs=[
            pl.BlockSpec(block_shape=(None, CHUNK_SIZE, None, K), index_map=lambda b, n, c: (b, c, n // Gq, 0)),
            pl.BlockSpec(block_shape=(None, CHUNK_SIZE, None, K), index_map=lambda b, n, c: (b, c, n // Gk, 0)),
            pl.BlockSpec(block_shape=(None, CHUNK_SIZE, None, V), index_map=lambda b, n, c: (b, c, n // Gv, 0)),
            pl.BlockSpec(block_shape=(None, None, K, V), index_map=lambda b, n, c: (b, n, 0, 0)),
        ],
        out_specs=(
            pl.BlockSpec(block_shape=(None, CHUNK_SIZE, None, V), index_map=lambda b, n, c: (b, c, n, 0)),
            pl.BlockSpec(block_shape=(None, None, K, V), index_map=lambda b, n, c: (b, n, 0, 0)),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
    )

    return kernel(q, k, v, h0)
