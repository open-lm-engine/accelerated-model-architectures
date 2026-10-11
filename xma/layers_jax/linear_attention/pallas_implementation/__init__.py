# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from ....math import ceil_divide
from .backward import _linear_attention_backward_core
from .forward import _BATCHED_HEAD_GROUP, _linear_attention_forward_core
from .state_passing import _linear_attention_state_passing_core


_MAX_HEADS_PER_PALLAS_CELL = 16


def _cumulative_log_decay(log_f: jax.Array | None, BLOCK_SIZE_S: int) -> jax.Array | None:
    if log_f is None:
        return None

    B, S = log_f.shape[:2]
    assert S % BLOCK_SIZE_S == 0

    log_f = log_f.reshape(B, S // BLOCK_SIZE_S, BLOCK_SIZE_S, *log_f.shape[2:])
    log_f = log_f.astype(jnp.float32)
    log_f_cumsum = jnp.cumsum(log_f, axis=2)
    log_f_cumsum = log_f_cumsum.reshape(B, S, *log_f.shape[3:])

    return log_f_cumsum


def _flatten_heads(x: jax.Array) -> jax.Array:
    # (B, S, heads, head_dim) -> (B, S, heads * head_dim): the kernels' operand layout. inside a model this cancels the
    # reshape that split a projection into heads, so no relayout copy is materialized around the kernels
    return x.reshape(*x.shape[:2], -1)


def _layout(
    q: jax.Array, k: jax.Array, v: jax.Array, log_f: jax.Array | None
) -> tuple[tuple[int, int, int, int], bool, bool]:
    heads = (q.shape[2], k.shape[2], v.shape[2], 0 if log_f is None else log_f.shape[2])
    N = max(heads)

    f_diagonal = log_f is not None and log_f.ndim == 4
    # the batched diagonal path consumes raw log_f and scans it in-kernel (see forward.py)
    fused_scan = f_diagonal and all(i == N for i in heads) and N % _BATCHED_HEAD_GROUP == 0

    return heads, f_diagonal, fused_scan


def _prepare(
    q: jax.Array, k: jax.Array, v: jax.Array, log_f: jax.Array | None, BLOCK_SIZE_S: int
) -> tuple[jax.Array | None, tuple[int, int, int, int], bool, bool]:
    heads, f_diagonal, fused_scan = _layout(q, k, v, log_f)
    log_f_cumsum = log_f if fused_scan else _cumulative_log_decay(log_f, BLOCK_SIZE_S)
    return log_f_cumsum, heads, f_diagonal, fused_scan


def _forward(q, k, v, log_f, h0, attention_multiplier, output_state, BLOCK_SIZE_S):
    log_f_cumsum, heads, f_diagonal, fused_scan = _prepare(q, k, v, log_f, BLOCK_SIZE_S)
    B, S, N, V = *q.shape[:2], max(heads), v.shape[-1]

    y, ht = _linear_attention_forward_core(
        q=_flatten_heads(q),
        k=_flatten_heads(k),
        v=_flatten_heads(v),
        log_f_cumsum=_flatten_heads(log_f_cumsum) if f_diagonal else log_f_cumsum,
        h0=h0,
        attention_multiplier=attention_multiplier,
        output_state=output_state,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        heads=heads,
        f_diagonal=f_diagonal,
        fused_scan=fused_scan,
    )

    return (y.reshape(B, S, N, V), ht), log_f_cumsum


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _linear_attention_pallas(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_f: jax.Array | None,
    h0: jax.Array | None,
    attention_multiplier: float,
    output_state: bool,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[jax.Array, jax.Array]:
    (y, ht), _ = _forward(q, k, v, log_f, h0, attention_multiplier, output_state, BLOCK_SIZE_S)
    return y, ht


def _linear_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_f: jax.Array | None,
    h0: jax.Array | None,
    attention_multiplier: float,
    output_state: bool,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[tuple[jax.Array, jax.Array | None], tuple]:
    # the per-chunk states are not kept as residuals (B * S / BLOCK_SIZE_S * N * K * V per layer); the backward
    # recomputes them with the state-passing kernel
    (y, ht), log_f_cumsum = _forward(q, k, v, log_f, h0, attention_multiplier, output_state, BLOCK_SIZE_S)
    return (y, ht), (q, k, v, log_f_cumsum, h0)


@partial(jax.jit, static_argnames=("attention_multiplier", "output_state", "BLOCK_SIZE_S", "BLOCK_SIZE_V"))
def _linear_attention_backward(
    attention_multiplier: float,
    output_state: bool,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
    residuals: tuple,
    cotangents: tuple,
) -> tuple:
    q, k, v, log_f_cumsum, h0 = residuals
    dy, dht = cotangents
    dht = dht if output_state else None

    # log_f_cumsum has log_f's shape (it is raw log_f on the fused diagonal path), so the layout is recovered exactly
    heads, f_diagonal, fused_scan = _layout(q, k, v, log_f_cumsum)

    Nq, Nk, Nv, Nf = heads
    B, S, _, K = q.shape
    V = v.shape[-1]
    N = max(heads)

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    # batched bwd kernel: same eligibility as the forward's batched path (un-gated, scalar-shared-gate, or the fused
    # full-head diagonal layout; a cross-head-shared diagonal gate keeps the per-head kernel)
    batched_bwd = (
        Nq == N
        and Nk == Nv == N
        and N % _BATCHED_HEAD_GROUP == 0
        and (log_f_cumsum is None or (not f_diagonal and Nf in (1, N)) or (f_diagonal and fused_scan))
    )

    log_f_cumsum_ = _flatten_heads(log_f_cumsum) if f_diagonal else log_f_cumsum

    h = _linear_attention_state_passing_core(
        k=_flatten_heads(k),
        v=_flatten_heads(v),
        log_f_cumsum=log_f_cumsum_,
        h0=h0,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        heads=heads,
        f_diagonal=f_diagonal,
        fused_scan=fused_scan,
    )

    dq, dk, dv, dlog_f, dh0 = _linear_attention_backward_core(
        q=_flatten_heads(q),
        k=_flatten_heads(k),
        v=_flatten_heads(v),
        log_f_cumsum=log_f_cumsum_,
        h=h,
        dy=_flatten_heads(dy),
        dht=dht,
        attention_multiplier=attention_multiplier,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
        heads=heads,
        f_diagonal=f_diagonal,
        fused_scan=fused_scan,
        batched=batched_bwd,
    )

    dq = dq.reshape(B, S, Nq, Gq, K).sum(axis=3)
    dk = dk.reshape(B, S, Nk, Gk, K).sum(axis=3)
    dv = dv.reshape(B, S, Nv, Gv, V).sum(axis=3)

    if log_f_cumsum is not None:
        Gf = N // Nf
        dlog_f = dlog_f.reshape(B, S, Nf, Gf, *log_f_cumsum.shape[3:]).sum(axis=3)
        # the diagonal kernel writes dlog_f in the operand dtype; the cotangent must match the gate's
        dlog_f = dlog_f.astype(log_f_cumsum.dtype)

    if h0 is None:
        dh0 = None

    return dq, dk, dv, dlog_f, dh0


_linear_attention_pallas.defvjp(_linear_attention_forward, _linear_attention_backward)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def _linear_attention_pallas_packed(
    qkv: jax.Array,
    log_f: jax.Array | None,
    h0: jax.Array | None,
    num_heads: int,
    attention_multiplier: float,
    output_state: bool,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[jax.Array, jax.Array | None]:
    y, ht, _ = _packed_forward(qkv, log_f, h0, num_heads, attention_multiplier, output_state, BLOCK_SIZE_S)
    return y, ht


def _packed_layout(
    qkv: jax.Array, log_f: jax.Array | None, num_heads: int
) -> tuple[tuple[int, int, int, int], bool, bool]:
    # q | k | v packed along the last dim, num_heads each, K == V
    N = num_heads
    heads = (N, N, N, 0 if log_f is None else log_f.shape[2])
    f_diagonal = log_f is not None and log_f.ndim == 4
    fused_scan = f_diagonal and heads[3] == N and N % _BATCHED_HEAD_GROUP == 0
    return heads, f_diagonal, fused_scan


def _packed_forward(qkv, log_f, h0, num_heads, attention_multiplier, output_state, BLOCK_SIZE_S):
    heads, f_diagonal, fused_scan = _packed_layout(qkv, log_f, num_heads)
    log_f_cumsum = log_f if fused_scan else _cumulative_log_decay(log_f, BLOCK_SIZE_S)

    y, ht = _linear_attention_forward_core(
        q=qkv,
        k=None,
        v=None,
        log_f_cumsum=_flatten_heads(log_f_cumsum) if f_diagonal else log_f_cumsum,
        h0=h0,
        attention_multiplier=attention_multiplier,
        output_state=output_state,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        heads=heads,
        f_diagonal=f_diagonal,
        fused_scan=fused_scan,
        packed=True,
    )

    return y, ht, log_f_cumsum


def _linear_attention_packed_forward(
    qkv, log_f, h0, num_heads, attention_multiplier, output_state, BLOCK_SIZE_S, BLOCK_SIZE_V
):
    y, ht, log_f_cumsum = _packed_forward(qkv, log_f, h0, num_heads, attention_multiplier, output_state, BLOCK_SIZE_S)
    return (y, ht), (qkv, log_f_cumsum, h0)


@partial(
    jax.jit, static_argnames=("num_heads", "attention_multiplier", "output_state", "BLOCK_SIZE_S", "BLOCK_SIZE_V")
)
def _linear_attention_packed_backward(
    num_heads: int,
    attention_multiplier: float,
    output_state: bool,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
    residuals: tuple,
    cotangents: tuple,
) -> tuple:
    qkv, log_f_cumsum, h0 = residuals
    dy, dht = cotangents

    heads, f_diagonal, fused_scan = _packed_layout(qkv, log_f_cumsum, num_heads)
    N, Nf = num_heads, heads[3]
    log_f_cumsum_ = _flatten_heads(log_f_cumsum) if f_diagonal else log_f_cumsum

    h = _linear_attention_state_passing_core(
        k=qkv,
        v=None,
        log_f_cumsum=log_f_cumsum_,
        h0=h0,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        heads=heads,
        f_diagonal=f_diagonal,
        fused_scan=fused_scan,
        packed=True,
    )

    dqkv, dlog_f, dh0 = _linear_attention_backward_core(
        q=qkv,
        k=None,
        v=None,
        log_f_cumsum=log_f_cumsum_,
        h=h,
        dy=dy,
        dht=dht if output_state else None,
        attention_multiplier=attention_multiplier,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
        heads=heads,
        f_diagonal=f_diagonal,
        fused_scan=fused_scan,
        batched=N % _BATCHED_HEAD_GROUP == 0
        and (log_f_cumsum is None or (not f_diagonal and Nf in (1, N)) or fused_scan),
        packed=True,
    )

    if log_f_cumsum is not None:
        B, S = qkv.shape[:2]
        dlog_f = dlog_f.reshape(B, S, Nf, N // Nf, *log_f_cumsum.shape[3:]).sum(axis=3)
        # the diagonal kernel writes dlog_f in the operand dtype; the cotangent must match the gate's
        dlog_f = dlog_f.astype(log_f_cumsum.dtype)

    return dqkv, dlog_f, (None if h0 is None else dh0)


_linear_attention_pallas_packed.defvjp(_linear_attention_packed_forward, _linear_attention_packed_backward)


def _linear_attention_pallas_chunked(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_f: jax.Array | None,
    h0: jax.Array | None,
    attention_multiplier: float,
    output_state: bool,
    BLOCK_SIZE_S: int,
    BLOCK_SIZE_V: int,
) -> tuple[jax.Array, jax.Array]:
    Nq = q.shape[-2]
    Nk = k.shape[-2]
    Nv = v.shape[-2]
    Nf = 0 if log_f is None else log_f.shape[2]

    N = max(Nq, Nk, Nv, Nf)

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv
    Gf = None if log_f is None else N // Nf

    groups = [(Gq, "query"), (Gk, "key"), (Gv, "value")]
    if log_f is not None:
        groups.append((Gf, "log_forget"))

    for G, name in groups:
        if _MAX_HEADS_PER_PALLAS_CELL % G != 0:
            raise ValueError(
                f"grouped head layout with a {name} group size of {G} cannot be split across "
                f"{_MAX_HEADS_PER_PALLAS_CELL}-head chunks (N={N}, Nq={Nq}, Nk={Nk}, Nv={Nv}, Nf={Nf}); "
                "choose q/k/v head counts whose group sizes all divide "
                f"{_MAX_HEADS_PER_PALLAS_CELL}, or use KernelBackend.jax"
            )

    NUM_CHUNKS = ceil_divide(N, _MAX_HEADS_PER_PALLAS_CELL)

    y = []
    ht = []

    for i in range(NUM_CHUNKS):
        start = i * _MAX_HEADS_PER_PALLAS_CELL
        end = min(N, start + _MAX_HEADS_PER_PALLAS_CELL)

        _log_f = None
        if log_f is not None:
            head_slice = slice(start // Gf, end // Gf)
            _log_f = log_f[..., head_slice] if log_f.ndim == 3 else log_f[..., head_slice, :]

        _y, _ht = _linear_attention_pallas(
            q=q[..., start // Gq : end // Gq, :],
            k=k[..., start // Gk : end // Gk, :],
            v=v[..., start // Gv : end // Gv, :],
            log_f=_log_f,
            h0=None if h0 is None else h0[:, start:end],
            attention_multiplier=attention_multiplier,
            output_state=output_state,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
        )

        y.append(_y)
        ht.append(_ht)

    y = jnp.concatenate(y, axis=2)
    ht = jnp.concatenate(ht, axis=1) if output_state else None

    return y, ht
