# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose


jax = pytest.importorskip("jax")

import jax.numpy as jnp

from xma import KernelBackend, swiglu_jax

from ..utils import get_2d_tensor_sizes


def _generate_args() -> list:
    return list(product(get_2d_tensor_sizes(), ["float32", "bfloat16"]))


_TOLERANCES = {
    "float32": {"atol": 1e-5, "rtol": 1e-5},
    "bfloat16": {"atol": 1e-2, "rtol": 1e-2},
}


@pytest.mark.parametrize("size,dtype", _generate_args())
def test_swiglu_jax(size: tuple[int, int], dtype: str) -> None:
    jax_dtype = getattr(jnp, dtype)
    tolerance = _TOLERANCES[dtype]

    key_gate, key_up = jax.random.split(jax.random.PRNGKey(0))
    gate = jax.random.normal(key_gate, size, dtype=jnp.float32).astype(jax_dtype)
    up = jax.random.normal(key_up, size, dtype=jnp.float32).astype(jax_dtype)

    y_kernel = swiglu_jax(gate, up)
    y_expected = swiglu_jax(gate, up, kernel_backend=KernelBackend.jax)

    assert_allclose(np.asarray(y_kernel, dtype=np.float32), np.asarray(y_expected, dtype=np.float32), **tolerance)

    loss_kernel = lambda gate, up: swiglu_jax(gate, up).astype(jnp.float32).sum()
    loss_reference = lambda gate, up: swiglu_jax(gate, up, kernel_backend=KernelBackend.jax).astype(jnp.float32).sum()

    dgate_kernel, dup_kernel = jax.grad(loss_kernel, argnums=(0, 1))(gate, up)
    dgate_expected, dup_expected = jax.grad(loss_reference, argnums=(0, 1))(gate, up)

    assert_allclose(
        np.asarray(dgate_kernel, dtype=np.float32), np.asarray(dgate_expected, dtype=np.float32), **tolerance
    )

    assert_allclose(np.asarray(dup_kernel, dtype=np.float32), np.asarray(dup_expected, dtype=np.float32), **tolerance)
