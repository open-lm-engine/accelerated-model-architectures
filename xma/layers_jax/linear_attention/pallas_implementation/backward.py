# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide
from .forward import _output_readout, _state_update


@partial(jax.jit, static_argnames=("attention_multiplier", "BLOCK_SIZE_S"))
def _linear_attention_backward_pallas_jit(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    dy: jax.Array,
    h0: jax.Array | None,
    dh: jax.Array,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    B, S, Nq, K = q.shape
    Nk = k.shape[-2]
    Nv, V = v.shape[-2:]

    N = max(Nq, Nk, Nv)

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    if h0 is None:
        h0 = jnp.zeros((B, N, K, V), dtype=jnp.float32)

    if dh is None:
        dh = jnp.zeros((B, N, K, V), dtype=jnp.float32)

    q = jnp.swapaxes(q, 1, 2)
    k = jnp.swapaxes(k, 1, 2)
    v = jnp.swapaxes(v, 1, 2)
    dy = jnp.swapaxes(dy, 1, 2)

    q_spec = pl.BlockSpec(block_shape=(None, None, BLOCK_SIZE_S, K), index_map=lambda b, n, c: (b, n // Gq, c, 0))
    k_spec = pl.BlockSpec(block_shape=(None, None, BLOCK_SIZE_S, K), index_map=lambda b, n, c: (b, n // Gk, c, 0))
    v_spec = pl.BlockSpec(block_shape=(None, None, BLOCK_SIZE_S, V), index_map=lambda b, n, c: (b, n // Gv, c, 0))

    kernel = pl.pallas_call(
        partial(
            _linear_attention_backward_pallas_kernel,
            attention_multiplier=attention_multiplier,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            S=S,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(shape=(B, Nq, S, K), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, Nk, S, K), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, Nv, S, V), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, N, K, V), dtype=jnp.float32),
        ),
        grid=(B, N, ceil_divide(S, BLOCK_SIZE_S)),
        in_specs=[
            q_spec,
            k_spec,
            v_spec,
            pl.BlockSpec(block_shape=(None, None, BLOCK_SIZE_S, V), index_map=lambda b, n, c: (b, n, c, 0)),
            pl.BlockSpec(block_shape=(None, None, K, V), index_map=lambda b, n, c: (b, n, 0, 0)),
        ],
        out_specs=(
            q_spec,
            k_spec,
            v_spec,
            pl.BlockSpec(block_shape=(None, None, K, V), index_map=lambda b, n, c: (b, n, 0, 0)),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
    )

    dq, dk, dv, dh0 = kernel(q, k, v, h0)
    dq = jnp.swapaxes(dq, 1, 2)
    dk = jnp.swapaxes(dk, 1, 2)
    dv = jnp.swapaxes(dv, 1, 2)

    return dq, dk, dv, dh0
