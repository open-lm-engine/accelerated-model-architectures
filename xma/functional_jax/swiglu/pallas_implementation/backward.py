# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jax.nn import sigmoid

from ....math import ceil_divide


def _swiglu_backward_pallas_kernel(g_ref, u_ref, dy_ref, dg_ref, du_ref):
    g = g_ref[...]
    u = u_ref[...]
    dy = dy_ref[...]

    dtype = g.dtype
    g = g.astype(jnp.float32)

    g_sigmoid = sigmoid(g)
    g_silu = g * g_sigmoid

    dg = dy * u * (g_sigmoid + g_silu * (1 - g_sigmoid))
    du = dy * g_silu

    dg_ref[...] = dg.astype(dtype)
    du_ref[...] = du.astype(dtype)


@jax.jit
def _swiglu_backward_pallas_jit(g: jax.Array, u: jax.Array, dy: jax.Array) -> tuple[jax.Array, jax.Array]:
    B, H = g.shape
    BLOCK_SIZE_H = min(ceil_divide(H, 128) * 128, 1024)
    # see forward.py: halve the target VMEM budget to leave headroom for Mosaic's double-buffered
    # pipelining, which pushes actual scoped VMEM usage well above this naive estimate.
    BLOCK_SIZE_B = max(1, 16 * 1024 * 1024 // (5 * BLOCK_SIZE_H * g.dtype.itemsize * 8)) << 3

    kernel = pl.pallas_call(
        _swiglu_backward_pallas_kernel,
        out_shape=[
            jax.ShapeDtypeStruct(shape=g.shape, dtype=g.dtype),
            jax.ShapeDtypeStruct(shape=g.shape, dtype=g.dtype),
        ],
        grid=(ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H)),
        in_specs=[
            pl.BlockSpec(block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_H), index_map=lambda x, y: (x, y)),
            pl.BlockSpec(block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_H), index_map=lambda x, y: (x, y)),
            pl.BlockSpec(block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_H), index_map=lambda x, y: (x, y)),
        ],
        out_specs=[
            pl.BlockSpec(block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_H), index_map=lambda x, y: (x, y)),
            pl.BlockSpec(block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_H), index_map=lambda x, y: (x, y)),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )

    return kernel(g, u, dy)
