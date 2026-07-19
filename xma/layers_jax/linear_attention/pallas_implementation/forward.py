# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide


def _state_update(h: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    h += jax.lax.dot_general(k, v, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    return h


def _output_readout(
    h: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    BLOCK_SIZE_S: int,
    attention_multiplier: float,
) -> jax.Array:
    dtype = q.dtype

    causal_row_ids = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 0)
    causal_col_ids = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 1)
    causal_mask = causal_row_ids > causal_col_ids

    qk = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    qk = jnp.where(causal_mask, qk, 0).astype(dtype)

    y = jnp.dot(qk, v, preferred_element_type=jnp.float32)
    y += jnp.dot(q, h.astype(dtype), preferred_element_type=jnp.float32)
    y *= attention_multiplier

    return y.astype(dtype)


def _linear_attention_forward_pallas_kernel(
    q_ref, k_ref, v_ref, h0_ref, y_ref, h_ref, *, attention_multiplier: float, BLOCK_SIZE_S: int, S: int
) -> None:
    @pl.when(pl.program_id(2) == 0)
    def _():
        h_ref[...] = h0_ref[...]

    dtype = q_ref.dtype

    BLOCK_ID_S = pl.program_id(2)
    BLOCK_S = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, 1), 0)
    MASK_S = (BLOCK_ID_S * BLOCK_SIZE_S + BLOCK_S) < S

    q = jnp.where(MASK_S, q_ref[...], 0).astype(dtype)
    k = jnp.where(MASK_S, k_ref[...], 0).astype(dtype)
    v = jnp.where(MASK_S, v_ref[...], 0).astype(dtype)
    h = h_ref[...]

    y_ref[...] = _output_readout(
        h=h, q=q, k=k, v=v, BLOCK_SIZE_S=BLOCK_SIZE_S, attention_multiplier=attention_multiplier
    )

    h = _state_update(h=h, k=k, v=v)
    h_ref[...] = h


@partial(jax.jit, static_argnames=("attention_multiplier", "BLOCK_SIZE_S"))
def _linear_attention_forward_pallas_jit(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
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

    q = jnp.swapaxes(q, 1, 2)
    k = jnp.swapaxes(k, 1, 2)
    v = jnp.swapaxes(v, 1, 2)

    kernel = pl.pallas_call(
        partial(
            _linear_attention_forward_pallas_kernel,
            attention_multiplier=attention_multiplier,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            S=S,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(shape=(B, N, S, V), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, N, K, V), dtype=jnp.float32),
        ),
        grid=(B, N, ceil_divide(S, BLOCK_SIZE_S)),
        in_specs=[
            pl.BlockSpec(block_shape=(None, None, BLOCK_SIZE_S, K), index_map=lambda b, n, c: (b, n // Gq, c, 0)),
            pl.BlockSpec(block_shape=(None, None, BLOCK_SIZE_S, K), index_map=lambda b, n, c: (b, n // Gk, c, 0)),
            pl.BlockSpec(block_shape=(None, None, BLOCK_SIZE_S, V), index_map=lambda b, n, c: (b, n // Gv, c, 0)),
            pl.BlockSpec(block_shape=(None, None, K, V), index_map=lambda b, n, c: (b, n, 0, 0)),
        ],
        out_specs=(
            pl.BlockSpec(block_shape=(None, None, BLOCK_SIZE_S, V), index_map=lambda b, n, c: (b, n, c, 0)),
            pl.BlockSpec(block_shape=(None, None, K, V), index_map=lambda b, n, c: (b, n, 0, 0)),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
    )

    y, h = kernel(q, k, v, h0)
    y = jnp.swapaxes(y, 1, 2)

    return y, h
