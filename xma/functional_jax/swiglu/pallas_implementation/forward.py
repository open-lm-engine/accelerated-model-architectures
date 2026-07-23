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


def _swiglu_forward_pallas_kernel(g_ref, u_ref, y_ref):
    g = g_ref[...]
    u = u_ref[...]

    dtype = g.dtype
    g = g.astype(jnp.float32)

    y = u * g * sigmoid(g)

    y_ref[...] = y.astype(dtype)


@jax.jit
def _swiglu_forward_pallas_jit(g: jax.Array, u: jax.Array) -> jax.Array:
    B, H = g.shape
    BLOCK_SIZE_H = min(ceil_divide(H, 128) * 128, 1024)
    # Mosaic double-buffers the grid for pipelining, so scoped VMEM usage runs well above this
    # naive (block_size_b * block_size_h * 3 live buffers) estimate; target half the 32MB scoped
    # VMEM limit here to leave headroom, instead of exceeding it (see RESOURCE_EXHAUSTED at full budget).
    BLOCK_SIZE_B = max(1, 16 * 1024 * 1024 // (3 * BLOCK_SIZE_H * g.dtype.itemsize * 8)) << 3

    kernel = pl.pallas_call(
        _swiglu_forward_pallas_kernel,
        out_shape=jax.ShapeDtypeStruct(shape=g.shape, dtype=g.dtype),
        grid=(ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H)),
        in_specs=[
            pl.BlockSpec(block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_H), index_map=lambda x, y: (x, y)),
            pl.BlockSpec(block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_H), index_map=lambda x, y: (x, y)),
        ],
        out_specs=pl.BlockSpec(block_shape=(BLOCK_SIZE_B, BLOCK_SIZE_H), index_map=lambda x, y: (x, y)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )

    return kernel(g, u)
