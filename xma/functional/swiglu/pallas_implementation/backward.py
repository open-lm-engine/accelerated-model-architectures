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


def swiglu_backward_pallas_kernel(g_ref, u_ref, dy_ref, dg_ref, du_ref):
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
def swiglu_backward_pallas_jit(g: jax.Array, u: jax.Array, dy: jax.Array) -> tuple[jax.Array, jax.Array]:
    B, H = g.shape
    BLOCK_SIZE_H = min(ceil_divide(H, 128) * 128, 1024)
    BLOCK_SIZE_B = min(1, 32 * 1024 * 1024 // (5 * BLOCK_SIZE_H * g.dtype.itemsize * 8)) << 3

    kernel = pl.pallas_call(
        swiglu_backward_pallas_kernel,
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


def _fake_func(g: torch.Tensor, u: torch.Tensor, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert g.is_contiguous()
    assert u.is_contiguous()
    assert dy.is_contiguous()

    return torch.empty_like(g), torch.empty_like(u)


_CACHE = None


@xma_op(mutates_args={}, fake_func=_fake_func)
def swiglu_backward_pallas(g: torch.Tensor, u: torch.Tensor, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert g.is_contiguous()
    assert u.is_contiguous()
    assert dy.is_contiguous()

    global _CACHE

    if _CACHE is None:
        _CACHE = make_kernel_from_pallas(
            swiglu_backward_pallas_jit, lambda g, u, dy: [(g.shape, g.dtype), (g.shape, g.dtype)]
        )

    return _CACHE(g, u, dy)
