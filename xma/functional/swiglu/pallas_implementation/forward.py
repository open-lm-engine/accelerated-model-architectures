# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from torch_xla.experimental.custom_kernel import jax_import_guard, make_kernel_from_pallas

from ....custom_op import xma_op
from ....math import ceil_divide


jax_import_guard()

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jax.nn import sigmoid


def swiglu_forward_pallas_kernel(g_ref, u_ref, y_ref):
    g = g_ref[...]
    u = u_ref[...]

    dtype = g.dtype
    g = g.astype(jnp.float32)

    y = u * g * sigmoid(g)

    y_ref[...] = y.astype(dtype)


@jax.jit
def swiglu_forward_pallas_jit(g: jax.Array, u: jax.Array) -> jax.Array:
    B, H = g.shape
    BLOCK_SIZE_H = min(ceil_divide(H, 128) * 128, 1024)
    BLOCK_SIZE_B = min(1, 32 * 1024 * 1024 // (3 * BLOCK_SIZE_H * g.dtype.itemsize * 8)) << 3

    kernel = pl.pallas_call(
        swiglu_forward_pallas_kernel,
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


def _fake_function(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    assert g.is_contiguous()
    assert u.is_contiguous()

    return torch.empty_like(g)


_CACHE = None


@xma_op(mutates_args={}, fake_func=_fake_function)
def swiglu_forward_pallas(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    assert g.is_contiguous()
    assert u.is_contiguous()

    global _CACHE

    if _CACHE is None:
        _CACHE = make_kernel_from_pallas(swiglu_forward_pallas_jit, lambda g, u: [(g.shape, g.dtype)])

    return _CACHE(g, u)
