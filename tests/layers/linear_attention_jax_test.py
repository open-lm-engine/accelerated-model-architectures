# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose


jax = pytest.importorskip("jax")

import jax.numpy as jnp

from xma import KernelBackend
from xma.layers_jax.linear_attention.op import linear_attention_jax


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
            [3, 16, 37, 64, 130],  # sequence length: shorter than, equal to, or not a multiple of CHUNK_SIZE
            [16, 32],  # CHUNK_SIZE
            _get_problem_shapes(),
            ["float32", "bfloat16"],
            [False, True],  # has_input_state
        )
    )


@pytest.mark.parametrize("S,CHUNK_SIZE,problem_shape,dtype,has_input_state", _generate_args())
def test_linear_attention_forward_pallas(
    S: int,
    CHUNK_SIZE: int,
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

    key_q, key_k, key_v, key_h0 = jax.random.split(jax.random.PRNGKey(0), 4)

    q = jax.random.normal(key_q, (B, S, Nq, K), dtype=jnp.float32).astype(jax_dtype)
    k = jax.random.normal(key_k, (B, S, Nk, K), dtype=jnp.float32).astype(jax_dtype)
    v = jax.random.normal(key_v, (B, S, Nv, V), dtype=jnp.float32).astype(jax_dtype)
    h0 = jax.random.normal(key_h0, (B, N, K, V), dtype=jnp.float32) if has_input_state else None

    y_kernel, ht_kernel = linear_attention_jax(
        q,
        k,
        v,
        h0,
        attention_multiplier=_ATTENTION_MULTIPLIER,
        CHUNK_SIZE=CHUNK_SIZE,
        kernel_backend=KernelBackend.pallas,
    )
    y_expected, ht_expected = linear_attention_jax(
        q,
        k,
        v,
        h0,
        attention_multiplier=_ATTENTION_MULTIPLIER,
        CHUNK_SIZE=CHUNK_SIZE,
        kernel_backend=KernelBackend.jax,
    )

    assert_allclose(np.asarray(y_kernel, dtype=np.float32), np.asarray(y_expected, dtype=np.float32), **tolerance)
    assert_allclose(np.asarray(ht_kernel, dtype=np.float32), np.asarray(ht_expected, dtype=np.float32), **tolerance)
