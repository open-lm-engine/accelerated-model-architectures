# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from functools import partial

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from .forward import _head, _heads


def _linear_attention_state_passing_kernel(
    k_ref,
    v_ref,
    log_f_cumsum_ref,
    h0_ref,
    h_ref,
    h_scratch,
    N: int,
    K: int,
    V: int,
    Gk: int,
    Gv: int,
    Gf: int | None,
    f_diagonal: bool,
    fused_scan: bool,
    G: int | None,
) -> None:
    @pl.when(pl.program_id(1) == 0)
    def _():
        if h0_ref is None:
            h_scratch[...] = jnp.zeros_like(h_scratch)
        else:
            h_scratch[...] = h0_ref[...].astype(jnp.float32)

    dtype = k_ref.dtype
    BLOCK_SIZE_S = k_ref.shape[0]

    if fused_scan:
        tril = jnp.tril(jnp.ones((BLOCK_SIZE_S, BLOCK_SIZE_S), jnp.float32))
        if G is not None:
            tril = jnp.broadcast_to(tril, (G, BLOCK_SIZE_S, BLOCK_SIZE_S))

    scalar_per_head = log_f_cumsum_ref is not None and not f_diagonal and log_f_cumsum_ref.shape[-1] != 1
    if log_f_cumsum_ref is not None and not f_diagonal and G is not None:
        if scalar_per_head:
            log_f_all = log_f_cumsum_ref[...].astype(jnp.float32)
        else:
            log_f = log_f_cumsum_ref[...][:, 0].astype(jnp.float32)
            h_decay = jnp.exp(log_f[-1])
            k_decay = jnp.exp(log_f[-1] - log_f)

    if G is not None:
        for n in range(0, N, G):
            g = slice(n, n + G)
            k = _heads(k_ref, n, G, K)
            v = _heads(v_ref, n, G, V)
            h = h_scratch[g]
            h_ref[g] = h.astype(h_ref.dtype)

            if log_f_cumsum_ref is not None:
                if f_diagonal:
                    log_f = _heads(log_f_cumsum_ref, n, G, K).astype(jnp.float32)
                    if fused_scan:
                        log_f = jax.lax.dot_general(
                            tril, log_f, (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32
                        )
                    log_f_last = log_f[:, -1]  # (G, K)

                    h *= jnp.exp(log_f_last)[:, :, None]
                    k = k * jnp.exp(log_f_last[:, None, :] - log_f)
                else:
                    if scalar_per_head:
                        log_f = log_f_all[g]
                        h_decay = jnp.exp(log_f[:, -1])
                        k_decay = jnp.exp(log_f[:, -1:] - log_f)
                    h *= h_decay[..., None, None]
                    k = k * k_decay[..., :, None]

            h += jax.lax.dot_general(
                k.astype(dtype), v, (((1,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32
            )
            h_scratch[g] = h
    else:
        for n in range(N):
            k = _head(k_ref, n // Gk, K)
            v = _head(v_ref, n // Gv, V)
            h = h_scratch[n]
            h_ref[n] = h.astype(h_ref.dtype)

            if log_f_cumsum_ref is not None:
                if f_diagonal:
                    log_f = _head(log_f_cumsum_ref, n // Gf, K).astype(jnp.float32)
                    if fused_scan:
                        log_f = jax.lax.dot_general(tril, log_f, (((1,), (0,)), ((), ())))
                    log_f_last = log_f[-1]

                    h *= jnp.exp(log_f_last)[:, None]
                    k = k * jnp.exp(log_f_last[None, :] - log_f)
                else:
                    log_f = log_f_cumsum_ref[:, n // Gf].astype(jnp.float32)

                    h *= jnp.exp(log_f[-1])
                    k = k * jnp.exp(log_f[-1] - log_f)[:, None]

            h += jax.lax.dot_general(k.astype(dtype), v, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
            h_scratch[n] = h


@partial(jax.jit, static_argnames=("BLOCK_SIZE_S", "heads", "f_diagonal", "fused_scan", "packed", "HEAD_GROUP"))
def _linear_attention_state_passing_core(
    k: jax.Array,
    v: jax.Array | None,
    log_f_cumsum: jax.Array | None,
    h0: jax.Array | None,
    BLOCK_SIZE_S: int,
    heads: tuple[int, int, int, int],
    f_diagonal: bool = False,
    fused_scan: bool = False,
    packed: bool = False,
    HEAD_GROUP: int | None = None,
) -> jax.Array:
    Nq, Nk, Nv, Nf = heads
    B, S = k.shape[:2]
    N = max(heads)

    if packed:
        assert Nq == Nk == Nv == N and k.shape[-1] % 3 == 0
        v = k
        K = V = k.shape[-1] // (3 * N)
    else:
        K = k.shape[-1] // Nk
        V = v.shape[-1] // Nv

    Gk = N // Nk
    Gv = N // Nv
    Gf = None if log_f_cumsum is None else N // Nf

    assert S % BLOCK_SIZE_S == 0
    NUM_BLOCKS_S = S // BLOCK_SIZE_S

    if HEAD_GROUP is None:
        per_head = log_f_cumsum is not None and not f_diagonal and Nf != 1
        HEAD_GROUP = 16 if per_head and N % 16 == 0 else 8

    batched = (
        Gk == 1
        and Gv == 1
        and N % HEAD_GROUP == 0
        and (log_f_cumsum is None or (not f_diagonal and Nf in (1, N)) or (f_diagonal and Gf == 1))
    )

    def seq_spec(width: int, lane_block: int = 0) -> pl.BlockSpec:
        return pl.BlockSpec(block_shape=(None, BLOCK_SIZE_S, width), index_map=lambda b, s: (b, s, lane_block))

    f_spec = None
    if log_f_cumsum is not None:
        f_spec = seq_spec(Nf * K if f_diagonal else Nf)

    if batched and log_f_cumsum is not None and not f_diagonal and Nf != 1:
        log_f_cumsum = log_f_cumsum.transpose(0, 2, 1)
        f_spec = pl.BlockSpec(block_shape=(None, Nf, BLOCK_SIZE_S), index_map=lambda b, s: (b, 0, s))

    kernel = pl.pallas_call(
        partial(
            _linear_attention_state_passing_kernel,
            N=N,
            K=K,
            V=V,
            Gk=Gk,
            Gv=Gv,
            Gf=Gf,
            f_diagonal=f_diagonal,
            fused_scan=fused_scan,
            G=HEAD_GROUP if batched else None,
        ),
        out_shape=jax.ShapeDtypeStruct(shape=(B, NUM_BLOCKS_S * N, K, V), dtype=k.dtype),
        grid=(B, NUM_BLOCKS_S),
        in_specs=(
            seq_spec(Nk * K, 1 if packed else 0),
            seq_spec(Nv * V, 2 if packed else 0),
            f_spec,
            None if h0 is None else pl.BlockSpec(block_shape=(None, N, K, V), index_map=lambda b, s: (b, 0, 0, 0)),
        ),
        out_specs=pl.BlockSpec(block_shape=(None, N, K, V), index_map=lambda b, s: (b, s, 0, 0)),
        scratch_shapes=[pltpu.VMEM((N, K, V), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, dimension_semantics=("parallel", "arbitrary")
        ),
    )

    return kernel(k, v, log_f_cumsum, h0)
