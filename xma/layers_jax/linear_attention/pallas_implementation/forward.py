# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp


_BATCHED_HEAD_GROUP = 4


def _get_causal_mask(BLOCK_SIZE_S, transpose: bool = False):
    row = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 0)
    col = jax.lax.broadcasted_iota(jnp.int32, (BLOCK_SIZE_S, BLOCK_SIZE_S), 1)

    if transpose:
        causal_mask = row <= col
    else:
        causal_mask = row >= col

    return causal_mask


def _scalar_decays(log_f: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    log_f_last = log_f[..., -1]
    return (
        jnp.exp(log_f[..., :, None] - log_f[..., None, :]),
        jnp.exp(log_f),
        jnp.exp(log_f_last),
        jnp.exp(log_f_last[..., None] - log_f),
    )


def _dot_exact_lhs(lhs: jax.Array, rhs: jax.Array, dimension_numbers) -> jax.Array:
    lhs = lhs.astype(jnp.bfloat16)
    hi = rhs.astype(jnp.bfloat16)
    lo = (rhs - hi.astype(jnp.float32)).astype(jnp.bfloat16)
    return jax.lax.dot_general(lhs, hi, dimension_numbers, preferred_element_type=jnp.float32) + jax.lax.dot_general(
        lhs, lo, dimension_numbers, preferred_element_type=jnp.float32
    )


def _head(ref, n: int, D: int, start: int = 0, end: int | None = None) -> jax.Array:
    return ref[:, n * D + start : n * D + (D if end is None else end)]


def _heads(ref, n: int, G: int, D: int, start: int = 0, end: int | None = None) -> jax.Array:
    return jnp.stack([_head(ref, n + i, D, start, end) for i in range(G)], axis=0)


def _store_heads(ref, n: int, D: int, x: jax.Array, start: int = 0, end: int | None = None) -> None:
    for i in range(x.shape[0]):
        ref[:, (n + i) * D + start : (n + i) * D + (D if end is None else end)] = x[i].astype(ref.dtype)


def _linear_attention_forward_kernel(
    q_ref,
    k_ref,
    v_ref,
    log_f_cumsum_ref,
    h0_ref,
    y_ref,
    ht_ref,
    h_scratch,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    N: int,
    K: int,
    V: int,
    Gq: int,
    Gk: int,
    Gv: int,
    Gf: int | None,
    f_diagonal: bool,
) -> None:
    @pl.when(pl.program_id(1) == 0)
    def _():
        if h0_ref is None:
            h_scratch[...] = jnp.zeros_like(h_scratch)
        else:
            h_scratch[...] = h0_ref[...].astype(jnp.float32)

    dtype = q_ref.dtype
    causal_mask = _get_causal_mask(BLOCK_SIZE_S)

    for n in range(N):
        q = _head(q_ref, n // Gq, K).astype(dtype)
        k = _head(k_ref, n // Gk, K).astype(dtype)
        v = _head(v_ref, n // Gv, V).astype(dtype)
        h = h_scratch[n]

        if log_f_cumsum_ref is None:
            qk = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            qk = jnp.where(causal_mask, qk, 0).astype(dtype)

            y = jnp.dot(qk, v, preferred_element_type=jnp.float32)
            y += jnp.dot(q, h.astype(dtype), preferred_element_type=jnp.float32)
        else:
            if f_diagonal:
                log_f = _head(log_f_cumsum_ref, n // Gf, K).astype(jnp.float32)
            else:
                log_f = log_f_cumsum_ref[:, n // Gf].astype(jnp.float32)

            log_f_last = log_f[-1]

            if f_diagonal:
                qf = (q * jnp.exp(log_f)).astype(dtype)
                kf_inv = (k * jnp.exp(-log_f)).astype(dtype)

                qk = jax.lax.dot_general(qf, kf_inv, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            else:
                qf = (q * jnp.exp(log_f[:, None])).astype(dtype)

                qk = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
                qk *= jnp.exp(log_f[:, None] - log_f[None, :])

            qk = jnp.where(causal_mask, qk, 0).astype(dtype)
            y = jnp.dot(qk, v, preferred_element_type=jnp.float32)
            y += jnp.dot(qf, h.astype(dtype), preferred_element_type=jnp.float32)

            if f_diagonal:
                h *= jnp.exp(log_f_last[:, None])
                k *= jnp.exp(log_f_last[None, :] - log_f)
            else:
                h *= jnp.exp(log_f_last)
                k *= jnp.exp(log_f_last - log_f)[:, None]

        k = k.astype(dtype)

        y *= attention_multiplier
        y_ref[:, n * V : (n + 1) * V] = y.astype(y_ref.dtype)

        h += jax.lax.dot_general(k, v, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
        h_scratch[n] = h

        if ht_ref is not None:
            ht_ref[n] = h.astype(ht_ref.dtype)


def _linear_attention_forward_batched_kernel(
    q_ref,
    k_ref,
    v_ref,
    log_f_cumsum_ref,
    h0_ref,
    y_ref,
    ht_ref,
    h_scratch,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
    N: int,
    K: int,
    V: int,
    f_diagonal: bool,
    fused_scan: bool,
    G: int,
) -> None:
    dtype = q_ref.dtype

    @pl.when(pl.program_id(1) == 0)
    def _():
        if h0_ref is None:
            h_scratch[...] = jnp.zeros_like(h_scratch)
        else:
            h_scratch[...] = h0_ref[...].astype(jnp.float32)

    causal_mask = _get_causal_mask(BLOCK_SIZE_S)

    if log_f_cumsum_ref is not None and f_diagonal and fused_scan:
        causal_batched = jnp.tril(jnp.ones((G, BLOCK_SIZE_S, BLOCK_SIZE_S), jnp.float32))

    scalar_per_head = log_f_cumsum_ref is not None and not f_diagonal and log_f_cumsum_ref.shape[-1] != 1
    if log_f_cumsum_ref is not None and not f_diagonal:
        if scalar_per_head:
            log_f_all = log_f_cumsum_ref[...].astype(jnp.float32)
        else:
            exp_decay, exp_log_f, h_decay, k_decay = _scalar_decays(log_f_cumsum_ref[...][:, 0].astype(jnp.float32))

    for n in range(0, N, G):
        g = slice(n, n + G)
        q = _heads(q_ref, n, G, K)
        k = _heads(k_ref, n, G, K)
        v = _heads(v_ref, n, G, V)
        h = h_scratch[g]

        if scalar_per_head:
            exp_decay, exp_log_f, h_decay, k_decay = _scalar_decays(log_f_all[g])

        if log_f_cumsum_ref is None:
            qf = q
            qk = jax.lax.dot_general(q, k, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32)
        elif f_diagonal:
            log_f_g = _heads(log_f_cumsum_ref, n, G, K).astype(jnp.float32)
            if fused_scan:
                log_f_g = jax.lax.dot_general(
                    causal_batched, log_f_g, (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32
                )

            log_f_last = log_f_g[:, -1]

            qf = (q * jnp.exp(log_f_g)).astype(dtype)
            kf_inv = (k * jnp.exp(-log_f_g)).astype(dtype)
            qk = jax.lax.dot_general(qf, kf_inv, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32)
        else:
            qf = (q * exp_log_f[..., :, None]).astype(dtype)

            qk = jax.lax.dot_general(q, k, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32)
            qk *= exp_decay

        qk = jnp.where(causal_mask[None], qk, 0).astype(dtype)

        y = jax.lax.dot_general(qk, v, (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32)
        y += jax.lax.dot_general(qf, h.astype(dtype), (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32)
        y *= attention_multiplier

        _store_heads(y_ref, n, V, y)

        if log_f_cumsum_ref is not None:
            if f_diagonal:
                h *= jnp.exp(log_f_last)[:, :, None]
                k = (k * jnp.exp(log_f_last[:, None, :] - log_f_g)).astype(dtype)
            else:
                h *= h_decay[..., None, None]
                k *= k_decay[..., :, None]

        h += jax.lax.dot_general(k, v, (((1,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32)
        h_scratch[g] = h

        if ht_ref is not None:
            ht_ref[g] = h.astype(ht_ref.dtype)


@partial(
    jax.jit,
    static_argnames=(
        "attention_multiplier",
        "BLOCK_SIZE_S",
        "heads",
        "output_state",
        "f_diagonal",
        "fused_scan",
        "packed",
    ),
)
def _linear_attention_forward_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_f_cumsum: jax.Array | None,
    h0: jax.Array | None,
    attention_multiplier: float,
    output_state: bool,
    BLOCK_SIZE_S: int,
    heads: tuple[int, int, int, int],
    f_diagonal: bool = False,
    fused_scan: bool = False,
    packed: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    Nq, Nk, Nv, Nf = heads
    B, S = q.shape[:2]

    if packed:
        assert Nq == Nk == Nv and q.shape[-1] % 3 == 0
        k = v = q
        K = V = q.shape[-1] // (3 * Nq)
    else:
        K = q.shape[-1] // Nq
        V = v.shape[-1] // Nv

    N = max(Nq, Nk, Nv, Nf)
    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv
    Gf = None if log_f_cumsum is None else N // Nf

    assert S % BLOCK_SIZE_S == 0
    NUM_BLOCKS_S = S // BLOCK_SIZE_S

    def seq_spec(width: int, lane_block: int = 0) -> pl.BlockSpec:
        return pl.BlockSpec(block_shape=(None, BLOCK_SIZE_S, width), index_map=lambda b, s: (b, s, lane_block))

    h_spec = pl.BlockSpec(block_shape=(None, N, K, V), index_map=lambda b, s: (b, 0, 0, 0))

    f_spec = None
    if log_f_cumsum is not None:
        f_spec = seq_spec(Nf * K if f_diagonal else Nf)

    batched = (
        Gq == 1
        and Gk == 1
        and Gv == 1
        and N % _BATCHED_HEAD_GROUP == 0
        and (log_f_cumsum is None or (not f_diagonal and Nf in (1, N)) or (f_diagonal and Gf == 1))
    )

    if batched and log_f_cumsum is not None and not f_diagonal and Nf != 1:
        log_f_cumsum = log_f_cumsum.transpose(0, 2, 1)
        f_spec = pl.BlockSpec(block_shape=(None, Nf, BLOCK_SIZE_S), index_map=lambda b, s: (b, 0, s))

    if batched:
        kernel_body = partial(
            _linear_attention_forward_batched_kernel,
            attention_multiplier=attention_multiplier,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            N=N,
            K=K,
            V=V,
            f_diagonal=f_diagonal,
            fused_scan=fused_scan,
            G=8 if f_diagonal and N % 8 == 0 else _BATCHED_HEAD_GROUP,
        )
    else:
        kernel_body = partial(
            _linear_attention_forward_kernel,
            attention_multiplier=attention_multiplier,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            N=N,
            K=K,
            V=V,
            Gq=Gq,
            Gk=Gk,
            Gv=Gv,
            Gf=Gf,
            f_diagonal=f_diagonal,
        )

    kernel = pl.pallas_call(
        kernel_body,
        out_shape=(
            jax.ShapeDtypeStruct(shape=(B, S, N * V), dtype=q.dtype),
            jax.ShapeDtypeStruct(shape=(B, N, K, V), dtype=jnp.float32) if output_state else None,
        ),
        grid=(B, NUM_BLOCKS_S),
        in_specs=(
            seq_spec(Nq * K),
            seq_spec(Nk * K, 1 if packed else 0),
            seq_spec(Nv * V, 2 if packed else 0),
            f_spec,
            None if h0 is None else h_spec,
        ),
        out_specs=(
            seq_spec(N * V),
            h_spec if output_state else None,
        ),
        scratch_shapes=[pltpu.VMEM((N, K, V), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            dimension_semantics=("parallel", "arbitrary"),
        ),
    )

    return kernel(q, k, v, log_f_cumsum, h0)
