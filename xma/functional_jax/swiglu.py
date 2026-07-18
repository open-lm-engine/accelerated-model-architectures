# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import jax
import jax.numpy as jnp

from ..accelerator import KernelBackend
from ..functional.swiglu.pallas_implementation import _swiglu_backward_pallas_jit, _swiglu_forward_pallas_jit
from ..utils import is_torch_xla_available


@jax.custom_vjp
def _swiglu_pallas_jax(gate: jax.Array, up: jax.Array) -> jax.Array:
    original_shape = gate.shape
    gate = gate.reshape(-1, original_shape[-1])
    up = up.reshape(-1, original_shape[-1])

    y = _swiglu_forward_pallas_jit(gate, up)

    return y.reshape(original_shape)


def _swiglu_pallas_jax_fwd(gate: jax.Array, up: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    return _swiglu_pallas_jax(gate, up), (gate, up)


def _swiglu_pallas_jax_bwd(residuals: tuple[jax.Array, jax.Array], dy: jax.Array) -> tuple[jax.Array, jax.Array]:
    gate, up = residuals
    original_shape = gate.shape

    gate = gate.reshape(-1, original_shape[-1])
    up = up.reshape(-1, original_shape[-1])
    dy = dy.reshape(-1, original_shape[-1])

    dg, du = _swiglu_backward_pallas_jit(gate, up, dy)

    return dg.reshape(original_shape), du.reshape(original_shape)


_swiglu_pallas_jax.defvjp(_swiglu_pallas_jax_fwd, _swiglu_pallas_jax_bwd)


def swiglu_jax(gate: jax.Array, up: jax.Array, *, kernel_backend: KernelBackend | None = None) -> jax.Array:
    assert gate.shape == up.shape, "gate and up should have the same shape"

    # this codebase only ever treats torch_xla's presence as TPU (see Accelerator.get_accelerator), so skip the
    # `jax.default_backend()` call in that case
    is_tpu = True if is_torch_xla_available() else jax.default_backend() == "tpu"

    if kernel_backend is None:
        kernel_backend = KernelBackend.pallas if is_tpu else KernelBackend.jax

    if kernel_backend == KernelBackend.pallas:
        assert is_tpu, "KernelBackend.pallas is only supported on TPU"
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
