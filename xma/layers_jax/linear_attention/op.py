# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import math

import jax
import jax.numpy as jnp

from ...accelerator import Accelerator, KernelBackend
from .pallas_implementation import _linear_attention_forward_pallas_jit


def _get_num_heads(q: jax.Array, k: jax.Array, v: jax.Array) -> tuple[int, int, int, int]:
    Nq = q.shape[-2]
    Nk = k.shape[-2]
    Nv = v.shape[-2]

    N = max(Nq, Nk, Nv)

    assert N % Nq == 0
    assert N % Nk == 0
    assert N % Nv == 0

    return Nq, Nk, Nv, N


def _linear_attention_reference(
    q: jax.Array, k: jax.Array, v: jax.Array, h0: jax.Array | None, attention_multiplier: float
) -> tuple[jax.Array, jax.Array]:
    # sequential recurrence y_s = q_s @ h_{s-1}, h_s = h_{s-1} + k_s ⊗ v_s, matching the torch reference in
    # xma/layers/linear_attention/op.py (_LinearAttention.forward_backward_torch)
    B, S, Nq, K = q.shape
    Nk = k.shape[-2]
    Nv, V = v.shape[-2:]
    N = max(Nq, Nk, Nv)

    q = jnp.repeat(q, N // Nq, axis=-2).astype(jnp.float32)
    k = jnp.repeat(k, N // Nk, axis=-2).astype(jnp.float32)
    v = jnp.repeat(v, N // Nv, axis=-2).astype(jnp.float32)

    h = jnp.zeros((B, N, K, V), dtype=jnp.float32) if h0 is None else h0.astype(jnp.float32)

    y = []
    for s in range(S):
        y.append(jnp.einsum("bnk,bnkv->bnv", q[:, s], h))
        h = h + k[:, s][..., :, None] * v[:, s][..., None, :]

    y = jnp.stack(y, axis=1) * attention_multiplier

    return y.astype(q.dtype), h


def linear_attention_jax(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    input_state: jax.Array | None = None,
    attention_multiplier: float | None = None,
    BLOCK_SIZE_S: int = 64,
    *,
    kernel_backend: KernelBackend | None = None,
) -> tuple[jax.Array, jax.Array]:
    B, S, _, K = query.shape
    V = value.shape[-1]

    Nq, Nk, Nv, N = _get_num_heads(query, key, value)

    assert query.shape == (B, S, Nq, K)
    assert key.shape == (B, S, Nk, K)
    assert value.shape == (B, S, Nv, V)

    if input_state is not None:
        assert input_state.shape == (B, N, K, V)

    if attention_multiplier is None:
        attention_multiplier = 1 / math.sqrt(K)

    if kernel_backend is None:
        kernel_backend = Accelerator.get_kernel_backend()

    if kernel_backend == KernelBackend.pallas:
        y, h = _linear_attention_forward_pallas_jit(
            query, key, value, input_state, attention_multiplier=attention_multiplier, BLOCK_SIZE_S=BLOCK_SIZE_S
        )
    elif kernel_backend == KernelBackend.jax:
        y, h = _linear_attention_reference(query, key, value, input_state, attention_multiplier)
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return y, h
