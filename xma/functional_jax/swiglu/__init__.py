# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import jax
import jax.numpy as jnp

from ...accelerator import Accelerator, KernelBackend
from .pallas_implementation import _swiglu_pallas_jax


def swiglu_jax(gate: jax.Array, up: jax.Array, *, kernel_backend: KernelBackend | None = None) -> jax.Array:
    assert gate.shape == up.shape, "gate and up should have the same shape"

    if kernel_backend is None:
        kernel_backend = Accelerator.get_kernel_backend()

    if kernel_backend == KernelBackend.pallas:
        x = _swiglu_pallas_jax(gate, up)
    elif kernel_backend == KernelBackend.jax:
        dtype = gate.dtype

        gate = gate.astype(jnp.float32)
        up = up.astype(jnp.float32)

        x = up * gate * jax.nn.sigmoid(gate)
        x = x.astype(dtype)
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return x
