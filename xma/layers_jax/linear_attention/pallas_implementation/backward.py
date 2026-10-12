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
from .forward import _BATCHED_HEAD_GROUP, _dot_exact_lhs, _get_causal_mask, _head, _heads, _scalar_decays, _store_heads


_BACKWARD_HEAD_GROUP = 8


def _linear_attention_backward_batched_kernel(
    q_ref,
    k_ref,
    v_ref,
    log_f_cumsum_ref,
    h_ref,
    dy_ref,
    dht_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    df_ref,
    dh0_ref,
    dh_scratch,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
    N: int,
    K: int,
    V: int,
    f_diagonal: bool,
    NUM_BLOCKS_V: int,
    NUM_BLOCKS_S: int,
    fused_scan: bool,
    G: int,
    gate_rows: bool = False,
) -> None:
    S_CELLS_VISITED = pl.program_id(1)

    @pl.when(S_CELLS_VISITED == 0)
    def _():
        if dht_ref is None:
            dh_scratch[...] = jnp.zeros_like(dh_scratch)
        else:
            dh_scratch[...] = dht_ref[...].astype(jnp.float32)

    dtype = q_ref.dtype

    h_ = h_ref[...].astype(dtype)

    log_f_cumsum_ = None
    if log_f_cumsum_ref is not None and not f_diagonal:
        log_f_cumsum_ = log_f_cumsum_ref[...]

    causal_mask = _get_causal_mask(BLOCK_SIZE_S)

    TriL_batched = None
    if log_f_cumsum_ref is not None and f_diagonal and fused_scan:
        TriL_batched = jnp.tril(jnp.ones((G, BLOCK_SIZE_S, BLOCK_SIZE_S), jnp.float32))

    TriU_batched = None
    if log_f_cumsum_ref is not None:
        TriU_batched = jnp.triu(jnp.ones((G, BLOCK_SIZE_S, BLOCK_SIZE_S), jnp.float32))

    scalar_per_head = log_f_cumsum_ref is not None and not f_diagonal and log_f_cumsum_ref.shape[-1] != 1
    if scalar_per_head:
        log_f_all = log_f_cumsum_.astype(jnp.float32)
    elif log_f_cumsum_ref is not None and not f_diagonal:
        f_ij_scalar, f_scalar, f_last_scalar, f_to_last_scalar = _scalar_decays(
            log_f_cumsum_[:, 0].astype(jnp.float32)
        )

    for n in range(0, N, G):
        g = slice(n, n + G)
        q = _heads(q_ref, n, G, K).astype(dtype)
        k = _heads(k_ref, n, G, K).astype(dtype)
        dy_full = (_heads(dy_ref, n, G, V).astype(jnp.float32) * attention_multiplier).astype(dtype)

        if log_f_cumsum_ref is None:
            qk = jax.lax.dot_general(q, k, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32)
            qk = jnp.where(causal_mask[None], qk, 0).astype(dtype)

            qf = q
            kf_last = k
            kf_inv = None
        else:
            if f_diagonal:
                log_f_g = _heads(log_f_cumsum_ref, n, G, K).astype(jnp.float32)
                if fused_scan:
                    log_f_g = jax.lax.dot_general(
                        TriL_batched, log_f_g, (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32
                    )
                log_f_last = log_f_g[:, -1]

                f = jnp.exp(log_f_g)
                f_last = jnp.exp(log_f_last)
                f_inv = jnp.exp(-log_f_g)
                f_to_last = f_last[:, None, :] * f_inv if fused_scan else jnp.exp(log_f_last[:, None, :] - log_f_g)

                qf = (q * f).astype(dtype)
                kf_inv = (k * f_inv).astype(dtype)
                kf_last = (k * f_to_last).astype(dtype)

                qk = jax.lax.dot_general(qf, kf_inv, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32)
            else:
                if scalar_per_head:
                    f_ij_scalar, f_scalar, f_last_scalar, f_to_last_scalar = _scalar_decays(log_f_all[g])
                f_to_last_b = f_to_last_scalar
                f_b = f_scalar

                qf = (q * f_b[..., :, None]).astype(dtype)
                kf_last = (k * f_to_last_b[..., :, None]).astype(dtype)
                kf_inv = None

                qk = jax.lax.dot_general(q, k, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32)
                qk = qk * f_ij_scalar

            qk = jnp.where(causal_mask[None], qk, 0).astype(dtype)

        dyv = jnp.zeros((G, BLOCK_SIZE_S, BLOCK_SIZE_S), jnp.float32)
        dq = jnp.zeros((G, BLOCK_SIZE_S, K), jnp.float32)
        dk = jnp.zeros((G, BLOCK_SIZE_S, K), jnp.float32)

        accumulate_df = log_f_cumsum_ref is not None and not f_diagonal

        if log_f_cumsum_ref is not None:
            df_last = jnp.zeros((G, K), jnp.float32)
        if accumulate_df:
            df = jnp.zeros((G, BLOCK_SIZE_S, K), jnp.float32)

        dh_group = dh_scratch[g][...]
        dh_out = None

        for BLOCK_ID_V in range(NUM_BLOCKS_V):
            start = BLOCK_ID_V * BLOCK_SIZE_V
            end = start + BLOCK_SIZE_V

            v = _heads(v_ref, n, G, V, start, end).astype(dtype)
            dy = dy_full[:, :, start:end]
            h = h_[g][:, :, start:end]
            dh = dh_group[:, :, start:end]

            dv = jax.lax.dot_general(qk, dy, (((1,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32)
            dv += jax.lax.dot_general(
                kf_last, dh.astype(dtype), (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32
            )
            _store_heads(dv_ref, n, V, dv, start, end)

            _dq = jax.lax.dot_general(dy, h, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32)
            if log_f_cumsum_ref is not None:
                _dq *= f if f_diagonal else f_b[..., :, None]
            dq += _dq

            _dk = jax.lax.dot_general(v, dh, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32)
            if log_f_cumsum_ref is not None:
                _dk *= f_to_last if f_diagonal else f_to_last_b[..., :, None]
            dk += _dk

            if accumulate_df:
                df_k = k.astype(jnp.float32) * _dk
                df += q.astype(jnp.float32) * _dq - df_k
                df_last += df_k.sum(axis=1)

            if log_f_cumsum_ref is not None:
                df_decay = h.astype(jnp.float32) * dh
                df_last += (f_last if f_diagonal else f_last_scalar[..., None]) * df_decay.sum(axis=2)
                dh *= f_last[:, :, None] if f_diagonal else f_last_scalar[..., None, None]
            dh += jax.lax.dot_general(qf, dy, (((1,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32)
            dh_out = dh if dh_out is None else jnp.concatenate((dh_out, dh), axis=2)

            dyv += jax.lax.dot_general(dy, v, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32)

        dh_scratch[g] = dh_out

        if log_f_cumsum_ref is not None and not f_diagonal:
            dyv *= f_ij_scalar
        dyv = jnp.where(causal_mask[None], dyv, 0)
        dyv = dyv.astype(dtype)

        dq_intra = jax.lax.dot_general(
            dyv, k if kf_inv is None else kf_inv, (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32
        )

        dk_intra = jax.lax.dot_general(
            dyv, q if kf_inv is None else qf, (((1,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32
        )

        if f_diagonal:
            dq_intra *= f
            dk_intra *= f_inv

        if f_diagonal:
            df_last += (k.astype(jnp.float32) * dk).sum(axis=1)

        dq += dq_intra
        dk += dk_intra

        _store_heads(dq_ref, n, K, dq)
        _store_heads(dk_ref, n, K, dk)

        if log_f_cumsum_ref is not None:
            if accumulate_df:
                df += q.astype(jnp.float32) * dq_intra - k.astype(jnp.float32) * dk_intra
            else:
                df = q.astype(jnp.float32) * dq - k.astype(jnp.float32) * dk

            dlog_f = _dot_exact_lhs(TriU_batched, df, (((2,), (1,)), ((0,), (0,)))) + df_last[:, None, :]

            if f_diagonal:
                _store_heads(df_ref, n, K, dlog_f)
            elif gate_rows:
                df_ref[g, :] = dlog_f.sum(axis=-1).astype(df_ref.dtype)
            else:
                df_ref[:, g] = dlog_f.sum(axis=-1).transpose(1, 0).astype(df_ref.dtype)

    @pl.when(S_CELLS_VISITED == NUM_BLOCKS_S - 1)
    def _():
        dh0_ref[...] = dh_scratch[...]


def _linear_attention_backward_kernel(
    q_ref,
    k_ref,
    v_ref,
    log_f_cumsum_ref,
    h_ref,
    dy_ref,
    dht_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    df_ref,
    dh0_ref,
    dh_scratch,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
    N: int,
    K: int,
    V: int,
    Gq: int,
    Gk: int,
    Gv: int,
    Gf: int | None,
    f_diagonal: bool,
    NUM_BLOCKS_V: int,
    NUM_BLOCKS_S: int,
    fused_scan: bool,
) -> None:
    S_CELLS_VISITED = pl.program_id(1)

    @pl.when(S_CELLS_VISITED == 0)
    def _():
        if dht_ref is None:
            dh_scratch[...] = jnp.zeros_like(dh_scratch)
        else:
            dh_scratch[...] = dht_ref[...].astype(jnp.float32)

    dtype = q_ref.dtype

    h_ = h_ref[...].astype(dtype)
    log_f_cumsum_ = log_f_cumsum_ref

    causal_mask = _get_causal_mask(BLOCK_SIZE_S)

    if fused_scan:
        TriL_scan = jnp.tril(jnp.ones((BLOCK_SIZE_S, BLOCK_SIZE_S), jnp.float32))

    for n in range(N):
        q = _head(q_ref, n // Gq, K).astype(dtype)
        k = _head(k_ref, n // Gk, K).astype(dtype)
        dy_full = (_head(dy_ref, n, V).astype(jnp.float32) * attention_multiplier).astype(dtype)

        if log_f_cumsum_ref is None:
            qk = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            qk = jnp.where(causal_mask, qk, 0).astype(dtype)

            qf = q
            kf_last = k
            kf_inv = None
        else:
            if f_diagonal:
                log_f = _head(log_f_cumsum_ref, n // Gf, K).astype(jnp.float32)
            else:
                log_f = log_f_cumsum_ref[:, n // Gf].astype(jnp.float32)
            if fused_scan:
                log_f = jax.lax.dot_general(TriL_scan, log_f, (((1,), (0,)), ((), ())))
            log_f_last = log_f[-1]

            f = jnp.exp(log_f)
            f_last = jnp.exp(log_f_last)

            if f_diagonal:
                f_to_last = jnp.exp(log_f_last[None, :] - log_f)
                f_inv = jnp.exp(-log_f)

                qf = (q * f).astype(dtype)
                kf_inv = (k * f_inv).astype(dtype)
                kf_last = (k * f_to_last).astype(dtype)

                qk = jax.lax.dot_general(qf, kf_inv, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            else:
                f_to_last = jnp.exp(log_f_last - log_f)
                f_ij = jnp.exp(log_f[:, None] - log_f[None, :])

                qf = (q * f[:, None]).astype(dtype)
                kf_last = (k * f_to_last[:, None]).astype(dtype)
                kf_inv = None

                qk = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
                qk *= f_ij

            qk = jnp.where(causal_mask, qk, 0).astype(dtype)

        dyv = jnp.zeros((BLOCK_SIZE_S, BLOCK_SIZE_S), jnp.float32)
        dq = jnp.zeros((BLOCK_SIZE_S, K), jnp.float32)
        dk = jnp.zeros((BLOCK_SIZE_S, K), jnp.float32)

        if log_f_cumsum_ is not None:
            df_last = jnp.zeros((K,), jnp.float32)

        for BLOCK_ID_V in range(NUM_BLOCKS_V):
            start = BLOCK_ID_V * BLOCK_SIZE_V
            end = start + BLOCK_SIZE_V

            v = _head(v_ref, n // Gv, V, start, end).astype(dtype)
            dy = dy_full[:, start:end]
            h = h_[n][:, start:end]
            dh = dh_scratch[n][:, start:end]

            dv = jax.lax.dot_general(qk, dy, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
            dv += jax.lax.dot_general(
                kf_last, dh.astype(dtype), (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
            )
            dv_ref[:, n * V + start : n * V + end] = dv.astype(dtype)

            _dq = jax.lax.dot_general(dy, h, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            if log_f_cumsum_ is not None:
                _dq *= f if f_diagonal else f[:, None]
            dq += _dq

            _dk = jax.lax.dot_general(v, dh, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            if log_f_cumsum_ is not None:
                _dk *= f_to_last if f_diagonal else f_to_last[:, None]
            dk += _dk

            if log_f_cumsum_ is not None:
                df_decay = h.astype(jnp.float32) * dh
                df_last += f_last * df_decay.sum(axis=1)

            if log_f_cumsum_ is not None:
                dh *= f_last[:, None] if f_diagonal else f_last
            dh += jax.lax.dot_general(qf, dy, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
            dh_scratch[n, :, start:end] = dh

            dyv += jax.lax.dot_general(dy, v, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

        if log_f_cumsum_ is not None and not f_diagonal:
            dyv *= f_ij
        dyv = jnp.where(causal_mask, dyv, 0)
        dyv = dyv.astype(dtype)

        dq_intra = jax.lax.dot_general(
            dyv, k if kf_inv is None else kf_inv, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )

        dk_intra = jax.lax.dot_general(
            dyv, q if kf_inv is None else qf, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )

        if f_diagonal:
            dq_intra *= f
            dk_intra *= f_inv

        if log_f_cumsum_ is not None:
            df_last += (k.astype(jnp.float32) * dk).sum(axis=0)

        dq += dq_intra
        dk += dk_intra

        dq_ref[:, n * K : (n + 1) * K] = dq.astype(dtype)
        dk_ref[:, n * K : (n + 1) * K] = dk.astype(dtype)

        if log_f_cumsum_ is not None:
            suffix_sum = jnp.where(_get_causal_mask(BLOCK_SIZE_S, transpose=True), 1.0, 0.0)

            df = q.astype(jnp.float32) * dq - k.astype(jnp.float32) * dk
            dlog_f = _dot_exact_lhs(suffix_sum, df, (((1,), (0,)), ((), ()))) + df_last

            if f_diagonal:
                df_ref[:, n * K : (n + 1) * K] = dlog_f.astype(df_ref.dtype)
            else:
                df_ref[:, n] = dlog_f.sum(axis=-1).astype(df_ref.dtype)

    @pl.when(S_CELLS_VISITED == NUM_BLOCKS_S - 1)
    def _():
        dh0_ref[...] = dh_scratch[...]


@partial(
    jax.jit,
    static_argnames=(
        "attention_multiplier",
        "BLOCK_SIZE_S",
        "BLOCK_SIZE_V",
        "heads",
        "f_diagonal",
        "fused_scan",
        "batched",
        "packed",
    ),
)
def _linear_attention_backward_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_f_cumsum: jax.Array | None,
    h: jax.Array,
    dy: jax.Array,
    dht: jax.Array | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
    heads: tuple[int, int, int, int],
    f_diagonal: bool = False,
    fused_scan: bool = False,
    batched: bool = False,
    packed: bool = False,
) -> tuple[jax.Array, ...]:
    Nq, Nk, Nv, Nf = heads
    B, S = q.shape[:2]
    N = max(Nq, Nk, Nv, Nf)

    if packed:
        assert Nq == Nk == Nv == N and q.shape[-1] % 3 == 0
        k = v = q
        K = V = q.shape[-1] // (3 * N)
    else:
        K = q.shape[-1] // Nq
        V = v.shape[-1] // Nv

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv
    Gf = None if log_f_cumsum is None else N // Nf

    if log_f_cumsum is not None:
        assert log_f_cumsum.shape == ((B, S, Nf * K) if f_diagonal else (B, S, Nf))

    assert S % BLOCK_SIZE_S == 0

    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)
    NUM_BLOCKS_V = ceil_divide(V, BLOCK_SIZE_V)
    assert (
        V == NUM_BLOCKS_V * BLOCK_SIZE_V
    ), "V must be an integer multiple of BLOCK_SIZE_V (host padding guarantees this)"

    def seq_spec(width: int, lane_block: int = 0) -> pl.BlockSpec:
        return pl.BlockSpec(
            block_shape=(None, BLOCK_SIZE_S, width), index_map=lambda b, s: (b, NUM_BLOCKS_S - 1 - s, lane_block)
        )

    f_spec = None
    df_spec = None
    if log_f_cumsum is not None:
        f_spec = seq_spec(Nf * K if f_diagonal else Nf)
        df_spec = seq_spec(N * K if f_diagonal else N)

    gate_rows = batched and log_f_cumsum is not None and not f_diagonal
    if gate_rows:
        rows_spec = lambda heads: pl.BlockSpec(
            block_shape=(None, heads, BLOCK_SIZE_S), index_map=lambda b, s: (b, 0, NUM_BLOCKS_S - 1 - s)
        )
        if Nf != 1:
            log_f_cumsum = log_f_cumsum.transpose(0, 2, 1)
            f_spec = rows_spec(Nf)
        df_spec = rows_spec(N)

    target = (
        partial(
            _linear_attention_backward_batched_kernel,
            attention_multiplier=attention_multiplier,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
            N=N,
            K=K,
            V=V,
            f_diagonal=f_diagonal,
            NUM_BLOCKS_V=NUM_BLOCKS_V,
            NUM_BLOCKS_S=NUM_BLOCKS_S,
            fused_scan=fused_scan,
            G=(
                _BACKWARD_HEAD_GROUP
                if N % _BACKWARD_HEAD_GROUP == 0 and (f_diagonal or log_f_cumsum is None or Nf == 1)
                else _BATCHED_HEAD_GROUP
            ),
            gate_rows=gate_rows,
        )
        if batched
        else partial(
            _linear_attention_backward_kernel,
            attention_multiplier=attention_multiplier,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
            N=N,
            K=K,
            V=V,
            Gq=Gq,
            Gk=Gk,
            Gv=Gv,
            Gf=Gf,
            f_diagonal=f_diagonal,
            NUM_BLOCKS_V=NUM_BLOCKS_V,
            NUM_BLOCKS_S=NUM_BLOCKS_S,
            fused_scan=fused_scan,
        )
    )

    if packed:
        W = N * K
        body = target

        def target(q_ref, k_ref, v_ref, f_ref, h_ref, dy_ref, dht_ref, dqkv_ref, df_ref, dh0_ref, dh_scratch):
            body(
                q_ref,
                k_ref,
                v_ref,
                f_ref,
                h_ref,
                dy_ref,
                dht_ref,
                dqkv_ref.at[:, 0:W],
                dqkv_ref.at[:, W : 2 * W],
                dqkv_ref.at[:, 2 * W : 3 * W],
                df_ref,
                dh0_ref,
                dh_scratch,
            )

    kernel = pl.pallas_call(
        target,
        out_shape=(
            *(
                [jax.ShapeDtypeStruct(shape=(B, S, 3 * N * K), dtype=q.dtype)]
                if packed
                else [
                    jax.ShapeDtypeStruct(shape=(B, S, N * K), dtype=q.dtype),
                    jax.ShapeDtypeStruct(shape=(B, S, N * K), dtype=q.dtype),
                    jax.ShapeDtypeStruct(shape=(B, S, N * V), dtype=q.dtype),
                ]
            ),
            (
                None
                if log_f_cumsum is None
                else jax.ShapeDtypeStruct(
                    shape=(B, S, N * K) if f_diagonal else ((B, N, S) if gate_rows else (B, S, N)),
                    dtype=q.dtype if f_diagonal else jnp.float32,
                )
            ),
            jax.ShapeDtypeStruct(shape=(B, N, K, V), dtype=jnp.float32),
        ),
        grid=(B, NUM_BLOCKS_S),
        in_specs=(
            seq_spec(Nq * K),
            seq_spec(Nk * K, 1 if packed else 0),
            seq_spec(Nv * V, 2 if packed else 0),
            f_spec,
            pl.BlockSpec(
                block_shape=(None, N, K, V),
                index_map=lambda b, s: (b, NUM_BLOCKS_S - 1 - s, 0, 0),
            ),
            seq_spec(N * V),
            (
                None
                if dht is None
                else pl.BlockSpec(
                    block_shape=(None, N, K, V),
                    index_map=lambda b, _: (b, 0, 0, 0),
                )
            ),
        ),
        out_specs=(
            *([seq_spec(3 * N * K)] if packed else [seq_spec(N * K), seq_spec(N * K), seq_spec(N * V)]),
            df_spec,
            pl.BlockSpec(
                block_shape=(None, N, K, V),
                index_map=lambda b, s: (b, 0, 0, 0),
            ),
        ),
        scratch_shapes=[pltpu.VMEM((N, K, V), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=(
                (48 << 20)
                if q.dtype.itemsize == 4
                else (40 << 20) if f_diagonal and log_f_cumsum.dtype.itemsize == 4 else None
            ),
        ),
    )

    outputs = kernel(q, k, v, log_f_cumsum, h, dy, dht)
    if gate_rows:
        outputs = (*outputs[:-2], outputs[-2].transpose(0, 2, 1), outputs[-1])

    return outputs
