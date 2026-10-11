# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from ....math import ceil_divide


def _backward_kernel(
    x_ref,
    W_ref,
    b_ref,
    h_ref,
    dy_ref,
    dht_ref,
    dx_ref,
    dW_ref,
    db_ref,
    dh0_ref,
    dy_scratch,
    dht_scratch,
    BLOCK_SIZE_S: int,
    S: int,
    K: int,
    PAD: int,
    HEAD: int,
    NUM_BLOCKS_S: int,
    ACTIVATION: str | None,
) -> None:
    BLOCK_ID_B = pl.program_id(0)
    BLOCK_ID_S_REVERSE = pl.program_id(1)
    BLOCK_ID_S = NUM_BLOCKS_S - 1 - BLOCK_ID_S_REVERSE

    H = x_ref.shape[-1]
    dtype = x_ref.dtype
    offset = PAD - K + 1

    @pl.when(BLOCK_ID_S_REVERSE == 0)
    def _():
        dy_scratch[...] = jnp.zeros_like(dy_scratch)
        dht_scratch[...] = jnp.zeros_like(dht_scratch)
        if dht_ref is not None:
            dht_scratch[offset:, :] = dht_ref[...].astype(jnp.float32)

    @pl.when((BLOCK_ID_B == 0) & (BLOCK_ID_S_REVERSE == 0))
    def _():
        dW_ref[...] = jnp.zeros(dW_ref.shape, dtype=dW_ref.dtype)
        db_ref[...] = jnp.zeros(db_ref.shape, dtype=db_ref.dtype)

    BLOCK_S = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, 1), 0)
    MASK_S = (BLOCK_ID_S * BLOCK_SIZE_S + BLOCK_S) < S
    MASKED = S % BLOCK_SIZE_S != 0

    # the dW / db reductions run on the (otherwise idle) MXU as (1, BLOCK_SIZE_S) x (BLOCK_SIZE_S, H) dots. a single
    # bf16 pass would round every fp32 product before it is summed, but the row of ones / zeros is exact in bf16, so
    # splitting only the fp32 operand into hi + lo bf16 halves recovers ~16 mantissa bits in two passes
    def dot(row: jax.Array, rhs: jax.Array) -> jax.Array:
        if rhs.dtype != jnp.float32:
            return jax.lax.dot(row, rhs, preferred_element_type=jnp.float32)

        row = row.astype(jnp.bfloat16)
        hi = rhs.astype(jnp.bfloat16)
        lo = (rhs - hi.astype(jnp.float32)).astype(jnp.bfloat16)
        return jax.lax.dot(row, hi, preferred_element_type=jnp.float32) + jax.lax.dot(
            row, lo, preferred_element_type=jnp.float32
        )

    ones_row = jnp.ones((1, BLOCK_SIZE_S), dtype=dtype)
    body_row = (jax.lax.broadcasted_iota(jnp.int32, (1, BLOCK_SIZE_S), 1) < BLOCK_SIZE_S - HEAD).astype(dtype)

    # this kernel is bound by vector issue slots, so every full-tile op counts:
    # - the sequence shifts are plain rolls over the whole block. only the rows that wrap around are wrong, and
    #   those are recomputed on small (PAD + HEAD, H) tiles with the neighbouring rows stacked on and spliced
    #   back in with sublane-aligned concatenates, instead of full-tile selects
    # - x is rolled in fp32, while rolling packed 16-bit rows by an odd amount costs 3 extra VALU ops per vreg
    # - dW reuses the shifted gradient taps of dx (sum_t g[t] x[t - s] == sum_u g[u + s] x[u]), so x is never
    #   shifted for the dW reduction
    if MASKED:
        dy = jnp.where(MASK_S, dy_ref[...], 0).astype(jnp.float32)
        x = jnp.where(MASK_S, x_ref[...], 0)
    else:
        dy = dy_ref[...].astype(jnp.float32)
        x = x_ref[...]

    x32 = x.astype(jnp.float32)
    W = W_ref[...]
    W32 = W.astype(jnp.float32)

    if ACTIVATION in ["silu", "swish"]:
        # fp32 all the way through (activation math must not touch bf16)
        hist = h_ref[0].astype(jnp.float32)  # hist[PAD + j] == x[j] for j < 0
        x_head = jnp.concatenate([hist, x32[:HEAD, :]], axis=0)

        y = jnp.zeros((BLOCK_SIZE_S, H), dtype=jnp.float32)
        y_head = jnp.zeros((HEAD, H), dtype=jnp.float32)
        if b_ref is not None:
            b32 = b_ref[...].astype(jnp.float32)
            y += b32
            y_head += b32

        for k in range(K):
            shift = K - 1 - k
            y += W32[k][None, :] * (pltpu.roll(x32, shift, axis=0) if shift > 0 else x32)
            y_head += W32[k][None, :] * (pltpu.roll(x_head, shift, axis=0) if shift > 0 else x_head)[PAD:, :]

        y = jnp.concatenate([y_head, y[HEAD:, :]], axis=0)

        sigmoid = jax.nn.sigmoid(y)
        dy = dy * sigmoid * (1 + y * (1 - sigmoid))

    dy_bf = dy.astype(dtype)

    # the last HEAD rows of each gradient tap read the (already gated) head of the next block from dy_scratch
    dy_tail = jnp.concatenate([dy[BLOCK_SIZE_S - HEAD :, :], dy_scratch[...]], axis=0)
    x_tail = x32[BLOCK_SIZE_S - HEAD :, :]

    db_ref[...] += dot(ones_row, dy_bf)

    dx = jnp.zeros((BLOCK_SIZE_S, H), dtype=dtype)
    dx_tail = jnp.zeros((HEAD, H), dtype=jnp.float32)
    for k in range(K):
        shift = K - 1 - k
        if shift == 0:
            dx += W[k][None, :] * dy_bf
            dx_tail += W32[k][None, :] * dy_tail[:HEAD, :]
            dW_ref[k, :] += dot(ones_row, dy_bf * x)[0]
        else:
            # g_tap[t] == g[t + shift], valid for t < BLOCK_SIZE_S - shift
            g_tap = pltpu.roll(dy_bf, BLOCK_SIZE_S - shift, axis=0)
            g_tap_tail = pltpu.roll(dy_tail, HEAD + PAD - shift, axis=0)[:HEAD, :]

            dx += W[k][None, :] * g_tap
            dx_tail += W32[k][None, :] * g_tap_tail
            dW_ref[k, :] += dot(body_row, g_tap * x)[0] + jnp.sum(g_tap_tail * x_tail, axis=0)

    dx = jnp.concatenate([dx[: BLOCK_SIZE_S - HEAD, :], dx_tail.astype(dtype)], axis=0)
    dx_ref[...] = (jnp.where(MASK_S, dx, 0) if MASKED else dx).astype(dtype)

    dy_scratch[...] = dy[:PAD, :]

    state_prefix = max(K - 1 - S, 0)
    x_state_start = max(S - (K - 1), 0)

    if dht_ref is not None:
        dht_rows = {}
        for p in range(state_prefix, K - 1):
            block_index, row_in_block = divmod(x_state_start + p - state_prefix, BLOCK_SIZE_S)
            dht_rows.setdefault(block_index, []).append((p, row_in_block))

        for block_index, rows in dht_rows.items():

            @pl.when(BLOCK_ID_S == block_index)
            def _(rows=rows):
                update = jnp.zeros((BLOCK_SIZE_S, H), dtype=jnp.float32)
                for p, row_in_block in rows:
                    update += jnp.where(BLOCK_S == row_in_block, dht_scratch[offset + p, :], 0)
                dx_ref[...] += update.astype(dtype)

    @pl.when(BLOCK_ID_S == 0)
    def _():
        # the history rows: dh0[PAD + u] = sum_k W[k] g[u + shift_k] for u < 0, and they also close the dW sums (the
        # pairs g[t] x[t - shift] with t - shift < 0). dy_scratch now holds the first PAD gated rows of this block
        W32 = W_ref[...].astype(jnp.float32)
        hist = h_ref[0].astype(jnp.float32)
        dy_head = jnp.concatenate([jnp.zeros((PAD, H), dtype=jnp.float32), dy_scratch[...]], axis=0)

        dh0 = jnp.zeros((PAD, H), dtype=jnp.float32)
        for k in range(K - 1):
            shift = K - 1 - k
            g_tap = pltpu.roll(dy_head, 2 * PAD - shift, axis=0)[:PAD, :]
            dh0 += W32[k][None, :] * g_tap
            dW_ref[k, :] += jnp.sum(g_tap * hist, axis=0)

        dh0_ref[...] = dh0

        for p in range(state_prefix):
            dh0_ref[offset + S + p, :] += dht_scratch[offset + p, :]


@partial(jax.jit, static_argnames=("BLOCK_SIZE_S", "K", "ACTIVATION"))
def _backward_core(
    x: jax.Array,
    W: jax.Array,
    b: jax.Array | None,
    h: jax.Array,
    dy: jax.Array,
    dht: jax.Array | None,
    BLOCK_SIZE_S: int,
    K: int,
    ACTIVATION: str | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    B, S, H = x.shape
    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)
    PAD = ceil_divide(K - 1, 8) * 8
    # rows patched separately at the block boundaries, a multiple of the packed sublane tile of x.dtype
    HEAD = ceil_divide(PAD, 32 // x.dtype.itemsize) * (32 // x.dtype.itemsize)
    state_size = K - 1

    # a block has to be long enough to carry the whole history in its leading/trailing HEAD rows
    assert BLOCK_SIZE_S >= 2 * HEAD, f"BLOCK_SIZE_S ({BLOCK_SIZE_S}) must be >= {2 * HEAD} for kernel_size ({K})"

    kernel = pl.pallas_call(
        partial(
            _backward_kernel,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            S=S,
            K=K,
            PAD=PAD,
            HEAD=HEAD,
            NUM_BLOCKS_S=NUM_BLOCKS_S,
            ACTIVATION=ACTIVATION,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((B, S, H), x.dtype),
            jax.ShapeDtypeStruct((K, H), jnp.float32),
            jax.ShapeDtypeStruct((1, H), jnp.float32),
            jax.ShapeDtypeStruct((B, PAD, H), jnp.float32),
        ),
        grid=(B, NUM_BLOCKS_S),
        in_specs=(
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, H),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, NUM_BLOCKS_S - 1 - BLOCK_ID_S, 0),
            ),
            pl.BlockSpec(block_shape=(K, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (0, 0)),
            None if b is None else pl.BlockSpec(block_shape=(1, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (0, 0)),
            pl.BlockSpec(
                block_shape=(None, 1, PAD, H),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, NUM_BLOCKS_S - 1 - BLOCK_ID_S, 0, 0),
            ),
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, H),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, NUM_BLOCKS_S - 1 - BLOCK_ID_S, 0),
            ),
            (
                None
                if dht is None
                else pl.BlockSpec(
                    block_shape=(None, state_size, H),
                    index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, 0, 0),
                )
            ),
        ),
        out_specs=(
            pl.BlockSpec(
                block_shape=(None, BLOCK_SIZE_S, H),
                index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, NUM_BLOCKS_S - 1 - BLOCK_ID_S, 0),
            ),
            pl.BlockSpec(block_shape=(K, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (0, 0)),
            pl.BlockSpec(block_shape=(1, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (0, 0)),
            pl.BlockSpec(block_shape=(None, PAD, H), index_map=lambda BLOCK_ID_B, BLOCK_ID_S: (BLOCK_ID_B, 0, 0)),
        ),
        scratch_shapes=[pltpu.VMEM((PAD, H), jnp.float32), pltpu.VMEM((PAD, H), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )

    dx, dW, db, dh0 = kernel(x, W, b, h, dy, dht)

    return dx, dW, db, dh0
