# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide


def _linear_attention_forward_pallas_kernel(
    q_ref, k_ref, v_ref, h0_ref, y_ref, h_ref, *, attention_multiplier: float, BLOCK_SIZE_S: int, S: int
) -> None:
    # h_ref is revisited (same block) across the sequential chunk grid dimension, so it doubles as the running
    # (K, V) recurrent state carried from one chunk to the next
    @pl.when(pl.program_id(2) == 0)
    def _():
        h_ref[...] = h0_ref[...]

    dtype = q_ref.dtype

    # the last chunk is padded when S is not a multiple of BLOCK_SIZE_S; padding values are unspecified garbage
    # (not guaranteed zero), so mask it out here rather than relying on it for the state accumulation below
    chunk_id = pl.program_id(2)
    row_ids = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, 1), 0)
    valid = (chunk_id * BLOCK_SIZE_S + row_ids) < S

    q = jnp.where(valid, q_ref[...], 0).astype(dtype)
    k = jnp.where(valid, k_ref[...], 0).astype(dtype)
    v = jnp.where(valid, v_ref[...], 0).astype(dtype)
    h = h_ref[...]

    # intra-chunk contribution: strictly causal (token s excludes its own key/value, matching the sequential
    # recurrence y_s = q_s @ h_{s-1}, h_s = h_{s-1} + k_s ⊗ v_s). dot_general fuses the k transpose into the
    # matmul instead of materializing it.
    qk = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    # built via iota comparison rather than jnp.tril(jnp.ones(..., dtype=bool)): the latter selects between two
    # boolean vectors internally, which Mosaic cannot legalize on TPU ("failed to legalize operation
    # 'arith.select'" on vector<.., i1>). Comparing int32 iotas produces the mask directly, with no bool select.
    causal_row_ids = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 0)
    causal_col_ids = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 1)
    causal_mask = causal_row_ids > causal_col_ids
    qk = jnp.where(causal_mask, qk, 0).astype(dtype)

    y = jnp.dot(qk, v, preferred_element_type=jnp.float32)
    # inter-chunk contribution: state accumulated from all strictly-previous chunks
    y += jnp.dot(q, h.astype(dtype), preferred_element_type=jnp.float32)
    y = (y * attention_multiplier).astype(dtype)

    y_ref[...] = y

    dh = jax.lax.dot_general(k, v, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    h_ref[...] = h + dh


@partial(jax.jit, static_argnames=("attention_multiplier", "BLOCK_SIZE_S"))
def _linear_attention_forward_pallas_jit(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h0: jax.Array | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
) -> tuple[jax.Array, jax.Array]:
    B, S, Nq, K = q.shape
    Nk = k.shape[-2]
    Nv, V = v.shape[-2:]

    N = max(Nq, Nk, Nv)

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    if h0 is None:
        h0 = jnp.zeros((B, N, K, V), dtype=jnp.float32)

    q = jnp.swapaxes(q, 1, 2)
    k = jnp.swapaxes(k, 1, 2)
    v = jnp.swapaxes(v, 1, 2)

    kernel = pl.pallas_call(
        partial(
            _linear_attention_forward_pallas_kernel,
            attention_multiplier=attention_multiplier,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            S=S,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(shape=(B, N, S, V), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, N, K, V), dtype=jnp.float32),
        ),
        grid=(B, N, ceil_divide(S, BLOCK_SIZE_S)),
        in_specs=[
            pl.BlockSpec(block_shape=(None, None, BLOCK_SIZE_S, K), index_map=lambda b, n, c: (b, n // Gq, c, 0)),
            pl.BlockSpec(block_shape=(None, None, BLOCK_SIZE_S, K), index_map=lambda b, n, c: (b, n // Gk, c, 0)),
            pl.BlockSpec(block_shape=(None, None, BLOCK_SIZE_S, V), index_map=lambda b, n, c: (b, n // Gv, c, 0)),
            pl.BlockSpec(block_shape=(None, None, K, V), index_map=lambda b, n, c: (b, n, 0, 0)),
        ],
        out_specs=(
            pl.BlockSpec(block_shape=(None, None, BLOCK_SIZE_S, V), index_map=lambda b, n, c: (b, n, c, 0)),
            pl.BlockSpec(block_shape=(None, None, K, V), index_map=lambda b, n, c: (b, n, 0, 0)),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
    )

    y, ht = kernel(q, k, v, h0)
    y = jnp.swapaxes(y, 1, 2)

    return y, ht
