# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import jax

from .backward import _swiglu_backward_pallas, _swiglu_backward_pallas_jit
from .forward import _swiglu_forward_pallas, _swiglu_forward_pallas_jit


@jax.custom_vjp
def swiglu_jax(gate: jax.Array, up: jax.Array) -> jax.Array:
    """
    computes swiglu activation as `up * gate * sigmoid(gate)` using the TPU pallas kernel, differentiable under
    `jax.grad`/`jax.vjp`

    :param gate: `gate` activation array
    :type gate: jax.Array
    :param up: `up` activation array
    :type up: jax.Array
    :return: output array
    :rtype: jax.Array
    """

    assert gate.shape == up.shape, "gate and up should have the same shape"

    original_shape = gate.shape
    gate = gate.reshape(-1, original_shape[-1])
    up = up.reshape(-1, original_shape[-1])

    y = _swiglu_forward_pallas_jit(gate, up)

    return y.reshape(original_shape)


def _swiglu_jax_fwd(gate: jax.Array, up: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    return swiglu_jax(gate, up), (gate, up)


def _swiglu_jax_bwd(residuals: tuple[jax.Array, jax.Array], dy: jax.Array) -> tuple[jax.Array, jax.Array]:
    gate, up = residuals
    original_shape = gate.shape

    gate = gate.reshape(-1, original_shape[-1])
    up = up.reshape(-1, original_shape[-1])
    dy = dy.reshape(-1, original_shape[-1])

    dg, du = _swiglu_backward_pallas_jit(gate, up, dy)

    return dg.reshape(original_shape), du.reshape(original_shape)


swiglu_jax.defvjp(_swiglu_jax_fwd, _swiglu_jax_bwd)
