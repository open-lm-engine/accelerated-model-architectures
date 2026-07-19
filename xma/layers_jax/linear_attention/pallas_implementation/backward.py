# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide
from .forward import _state_update


def _linear_attention_checkpoint_pallas_kernel(
    k_ref, v_ref, h0_ref, h_checkpoint_ref, h_ref, *, BLOCK_SIZE_S: int, S: int
) -> None:
    # forward-direction sweep over chunks that only saves the state h^{(c)} at the START of every chunk c
    # (before it is updated with chunk c's own key/value) - the reverse pass below needs these, and they
    # aren't otherwise available since the forward kernel only returns the FINAL state. h_ref is revisited
    # (same block) across chunks, doubling as the running state, exactly like in the forward kernel.
    @pl.when(pl.program_id(2) == 0)
    def _():
        h_ref[...] = h0_ref[...]

    dtype = k_ref.dtype

    BLOCK_ID_S = pl.program_id(2)
    BLOCK_S = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, 1), 0)
    MASK_S = (BLOCK_ID_S * BLOCK_SIZE_S + BLOCK_S) < S

    k = jnp.where(MASK_S, k_ref[...], 0).astype(dtype)
    v = jnp.where(MASK_S, v_ref[...], 0).astype(dtype)
    h = h_ref[...]

    h_checkpoint_ref[...] = h[None]

    h_ref[...] = _state_update(h=h, k=k, v=v)


def _linear_attention_backward_pallas_kernel(
    q_ref,
    k_ref,
    v_ref,
    dy_ref,
    h_checkpoint_ref,
    dh_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    dh0_ref,
    *,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    S: int,
    NUM_BLOCKS_S: int,
) -> None:
    # reverse-direction sweep: grid position rc (0..NUM_BLOCKS_S-1) processes physical chunk
    # NUM_BLOCKS_S - 1 - rc, i.e. the last chunk first. dh0_ref is revisited across rc, doubling as the running
    # "future gradient" state g: g^{(C)} = dh (the incoming gradient on the final state), g^{(c)} =
    # g^{(c+1)} + q_c^T @ dy_c, and dh0 = g^{(0)} once the sweep reaches chunk 0 - the same revisit trick
    # h_ref uses in the forward pass, just accumulating backwards instead of forwards.
    rc = pl.program_id(2)

    @pl.when(rc == 0)
    def _():
        dh0_ref[...] = dh_ref[...]

    dtype = q_ref.dtype

    physical_chunk = NUM_BLOCKS_S - 1 - rc
    BLOCK_S = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, 1), 0)
    MASK_S = (physical_chunk * BLOCK_SIZE_S + BLOCK_S) < S

    q = jnp.where(MASK_S, q_ref[...], 0).astype(dtype)
    k = jnp.where(MASK_S, k_ref[...], 0).astype(dtype)
    v = jnp.where(MASK_S, v_ref[...], 0).astype(dtype)
    dy = jnp.where(MASK_S, dy_ref[...], 0).astype(jnp.float32) * attention_multiplier
    dy = dy.astype(dtype)

    h_c = h_checkpoint_ref[...][0].astype(dtype)
    g = dh0_ref[...]

    causal_row_ids = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 0)
    causal_col_ids = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 1)
    causal_mask = causal_row_ids > causal_col_ids

    # qk[i, j] = q_i . k_j (i = query position, j = key position), masked strictly causal (i > j) exactly as
    # in the forward pass
    qk = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    masked_qk = jnp.where(causal_mask, qk, 0).astype(dtype)

    # d(masked_qk)[i, j] = dy_i . v_j, then re-apply the same causal mask (masked_qk = causal_mask * qk, so
    # the gradient flowing back through that elementwise product carries the same mask)
    d_masked_qk = jax.lax.dot_general(dy, v, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    d_qk = jnp.where(causal_mask, d_masked_qk, 0).astype(dtype)

    # dq_i = sum_j d_qk[i, j] * k_j  (intra-chunk)  +  dy_i @ h_c^T  (inter-chunk, through y_inter = q @ h_c)
    dq = jax.lax.dot_general(d_qk, k, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    dq += jax.lax.dot_general(dy, h_c, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

    # dk_j = sum_i d_qk[i, j] * q_i  (intra-chunk)  +  v_j @ g^T  (through the state update h += k^T @ v)
    dk = jax.lax.dot_general(d_qk, q, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    dk += jax.lax.dot_general(v, g.astype(dtype), (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

    # dv_j = sum_i masked_qk[i, j] * dy_i  (intra-chunk)  +  k_j @ g  (through the state update)
    dv = jax.lax.dot_general(masked_qk, dy, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    dv += jax.lax.dot_general(k, g.astype(dtype), (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)

    dq_ref[...] = dq.astype(dtype)
    dk_ref[...] = dk.astype(dtype)
    dv_ref[...] = dv.astype(dtype)

    # g^{(c)} = g^{(c+1)} + q_c^T @ dy_c, ready for the next (earlier) chunk
    dh0_ref[...] = g + jax.lax.dot_general(q, dy, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)


@partial(jax.jit, static_argnames=("attention_multiplier", "BLOCK_SIZE_S"))
def _linear_attention_backward_pallas_jit(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    dy: jax.Array,
    h0: jax.Array | None,
    dh: jax.Array | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    B, S, Nq, K = q.shape
    Nk = k.shape[-2]
    Nv, V = v.shape[-2:]

    N = max(Nq, Nk, Nv)

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)

    if h0 is None:
        h0 = jnp.zeros((B, N, K, V), dtype=jnp.float32)

    if dh is None:
        dh = jnp.zeros((B, N, K, V), dtype=jnp.float32)

    q = jnp.swapaxes(q, 1, 2)
    k = jnp.swapaxes(k, 1, 2)
    v = jnp.swapaxes(v, 1, 2)
    dy = jnp.swapaxes(dy, 1, 2)

    k_in_spec = pl.BlockSpec(block_shape=(None, None, BLOCK_SIZE_S, K), index_map=lambda b, n, c: (b, n // Gk, c, 0))
    v_in_spec = pl.BlockSpec(block_shape=(None, None, BLOCK_SIZE_S, V), index_map=lambda b, n, c: (b, n // Gv, c, 0))
    h_running_spec = pl.BlockSpec(block_shape=(None, None, K, V), index_map=lambda b, n, c: (b, n, 0, 0))

    kernel = pl.pallas_call(
        partial(_linear_attention_checkpoint_pallas_kernel, BLOCK_SIZE_S=BLOCK_SIZE_S, S=S),
        out_shape=(
            jax.ShapeDtypeStruct(shape=(B, N, NUM_BLOCKS_S, K, V), dtype=jnp.float32),
            jax.ShapeDtypeStruct(shape=(B, N, K, V), dtype=jnp.float32),
        ),
        grid=(B, N, NUM_BLOCKS_S),
        in_specs=[k_in_spec, v_in_spec, h_running_spec],
        out_specs=(
            pl.BlockSpec(block_shape=(None, None, 1, K, V), index_map=lambda b, n, c: (b, n, c, 0, 0)),
            h_running_spec,
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
    )

    h_checkpoints, _ = kernel(k, v, h0)

    kernel = pl.pallas_call(
        partial(
            _linear_attention_backward_pallas_kernel,
            attention_multiplier=attention_multiplier,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            S=S,
            NUM_BLOCKS_S=NUM_BLOCKS_S,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(shape=(B, N, S, K), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, N, S, K), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, N, S, V), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, N, K, V), dtype=jnp.float32),
        ),
        grid=(B, N, NUM_BLOCKS_S),
        in_specs=[
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, K),
                index_map=lambda b, n, rc: (b, n // Gq, NUM_BLOCKS_S - 1 - rc, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, K),
                index_map=lambda b, n, rc: (b, n // Gk, NUM_BLOCKS_S - 1 - rc, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, V),
                index_map=lambda b, n, rc: (b, n // Gv, NUM_BLOCKS_S - 1 - rc, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, V), index_map=lambda b, n, rc: (b, n, NUM_BLOCKS_S - 1 - rc, 0)
            ),
            pl.BlockSpec(
                block_shape=(None, None, 1, K, V), index_map=lambda b, n, rc: (b, n, NUM_BLOCKS_S - 1 - rc, 0, 0)
            ),
            h_running_spec,
        ],
        out_specs=(
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, K), index_map=lambda b, n, rc: (b, n, NUM_BLOCKS_S - 1 - rc, 0)
            ),
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, K), index_map=lambda b, n, rc: (b, n, NUM_BLOCKS_S - 1 - rc, 0)
            ),
            pl.BlockSpec(
                block_shape=(None, None, BLOCK_SIZE_S, V), index_map=lambda b, n, rc: (b, n, NUM_BLOCKS_S - 1 - rc, 0)
            ),
            h_running_spec,
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
    )

    dq, dk, dv, dh0 = kernel(q, k, v, dy, h_checkpoints, dh)

    dq = jnp.swapaxes(dq, 1, 2)
    dk = jnp.swapaxes(dk, 1, 2)
    dv = jnp.swapaxes(dv, 1, 2)

    dq = dq.reshape(B, S, Nq, Gq, K).sum(axis=3)
    dk = dk.reshape(B, S, Nk, Gk, K).sum(axis=3)
    dv = dv.reshape(B, S, Nv, Gv, V).sum(axis=3)

    return dq, dk, dv, dh0
