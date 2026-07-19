# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide


def _linear_attention_forward_pallas_kernel(q_ref, k_ref, v_ref, h0_ref, h_ref, y_ref):
    @pl.when(pl.program_id(2) == 0)
    def _():
        if h0_ref is None:
            jnp.zeros_like(h0_ref)
        else:
            h0_ref[...]

    q_ref.dtype

    k_ref[...]
    v_ref[...]
    h0_ref[...]


def _linear_attention_forward_pallas_jit(
    q: jax.Array, k: jax.Array, v: jax.Array, h0: jax.Array | None
) -> tuple[jax.Array, jax.Array]:
    B, S, Nq, K = q.shape
    Nk = k.shape[-2]
    Nv, V = v.shape[-2:]

    N = max(Nq, Nk, Nv)

    BLOCK_SIZE_S = 256

    kernel = pl.pallas_call(
        _linear_attention_forward_pallas_kernel,
        out_shape=(
            jax.ShapeDtypeStruct(shape=(B, S, N, V), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, N, K, V), dtype=jnp.float32),
        ),
        grid=(B, N * V, ceil_divide(S, BLOCK_SIZE_S)),
    )

    return kernel(q, k, v, h0)
