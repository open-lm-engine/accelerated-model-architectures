# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose


jax = pytest.importorskip("jax")

import jax.numpy as jnp

from xma import KernelBackend, linear_attention_jax


_ATTENTION_MULTIPLIER = 0.3
_TOLERANCES = {
    "float32": {"atol": 1e-4, "rtol": 1e-4},
    "bfloat16": {"atol": 2e-2, "rtol": 2e-2},
}


def _get_problem_shapes() -> list[tuple[int, int, int, int, int]]:
    # (K, V, Nq, Nk, Nv)
    return [
        (16, 16, 1, 1, 1),
        (32, 24, 4, 4, 4),
        (16, 16, 4, 2, 1),
    ]


def _generate_args() -> list:
    return list(
        product(
            [3, 16, 37, 64, 130],  # sequence length: shorter than, equal to, or not a multiple of BLOCK_SIZE_S
            [16, 32],  # BLOCK_SIZE_S
            _get_problem_shapes(),
            ["float32", "bfloat16"],
            [False, True],  # has_input_state
        )
    )


@pytest.mark.parametrize("S,BLOCK_SIZE_S,problem_shape,dtype,has_input_state", _generate_args())
def test_linear_attention_pallas(
    S: int,
    BLOCK_SIZE_S: int,
    problem_shape: tuple[int, int, int, int, int],
    dtype: str,
    has_input_state: bool,
) -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("KernelBackend.pallas is only supported on TPU")

    K, V, Nq, Nk, Nv = problem_shape
    N = max(Nq, Nk, Nv)
    B = 2

    jax_dtype = getattr(jnp, dtype)
    tolerance = _TOLERANCES[dtype]

    key_q, key_k, key_v, key_h0, key_dy, key_dht = jax.random.split(jax.random.PRNGKey(0), 6)
    std = 0.01

    q = jax.random.normal(key_q, (B, S, Nq, K), dtype=jnp.float32).astype(jax_dtype) * std
    k = jax.random.normal(key_k, (B, S, Nk, K), dtype=jnp.float32).astype(jax_dtype) * std
    v = jax.random.normal(key_v, (B, S, Nv, V), dtype=jnp.float32).astype(jax_dtype) * std
    h0 = jax.random.normal(key_h0, (B, N, K, V), dtype=jnp.float32) * std if has_input_state else None

    def _run(kernel_backend: KernelBackend, q: jax.Array, k: jax.Array, v: jax.Array, h0: jax.Array | None):
        return linear_attention_jax(
            q,
            k,
            v,
            h0,
            attention_multiplier=_ATTENTION_MULTIPLIER,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            kernel_backend=kernel_backend,
        )

    (y_kernel, ht_kernel), vjp_kernel = jax.vjp(
        lambda q, k, v, h0: _run(KernelBackend.pallas, q, k, v, h0), q, k, v, h0
    )
    (y_expected, ht_expected), vjp_expected = jax.vjp(
        lambda q, k, v, h0: _run(KernelBackend.jax, q, k, v, h0), q, k, v, h0
    )

    assert_allclose(np.asarray(y_kernel, dtype=np.float32), np.asarray(y_expected, dtype=np.float32), **tolerance)
    assert_allclose(np.asarray(ht_kernel, dtype=np.float32), np.asarray(ht_expected, dtype=np.float32), **tolerance)

    # ht is always returned (even when no input_state was given), so it always needs a cotangent
    dy = jax.random.normal(key_dy, y_kernel.shape, dtype=jnp.float32).astype(jax_dtype) * std
    dht = jax.random.normal(key_dht, ht_kernel.shape, dtype=jnp.float32) * std

    dq_kernel, dk_kernel, dv_kernel, dh0_kernel = vjp_kernel((dy, dht))
    dq_expected, dk_expected, dv_expected, dh0_expected = vjp_expected((dy, dht))

    assert_allclose(np.asarray(dq_kernel, dtype=np.float32), np.asarray(dq_expected, dtype=np.float32), **tolerance)
    assert_allclose(np.asarray(dk_kernel, dtype=np.float32), np.asarray(dk_expected, dtype=np.float32), **tolerance)
    assert_allclose(np.asarray(dv_kernel, dtype=np.float32), np.asarray(dv_expected, dtype=np.float32), **tolerance)

    if has_input_state:
        assert_allclose(
            np.asarray(dh0_kernel, dtype=np.float32), np.asarray(dh0_expected, dtype=np.float32), **tolerance
        )
    else:
        assert dh0_kernel is None
        assert dh0_expected is None
