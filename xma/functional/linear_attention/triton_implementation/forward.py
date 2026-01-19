# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import ceil_divide, get_powers_of_2
from ....triton_utils import matmul
from ....xtuner import XTuneConfig, xtune
from ..utils import _get_num_heads


def _get_autotune_configs__recurrent_state_forward_triton_kernel() -> list[triton.Config]:
    configs = []
    for num_warps in get_powers_of_2(4, 8):
        for num_stages in range(1, 5):
            for BLOCK_SIZE_S in [1] + get_powers_of_2(16, 64):
                for BLOCK_SIZE_K in get_powers_of_2(32, 64):
                    for BLOCK_SIZE_V in get_powers_of_2(32, 64):
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_SIZE_S": BLOCK_SIZE_S,
                                    "BLOCK_SIZE_K": BLOCK_SIZE_K,
                                    "BLOCK_SIZE_V": BLOCK_SIZE_V,
                                },
                                num_stages=num_stages,
                                num_warps=num_warps,
                            )
                        )

    return configs


@triton.autotune(configs=_get_autotune_configs__recurrent_state_forward_triton_kernel(), key=[])
@triton.jit
def recurrent_state_forward_triton_kernel(
    q_ptr,
    q_stride,
    k_ptr,
    k_stride,
    v_ptr,
    v_stride,
    h0_ptr,
    h0_stride,
    h_ptr,
    h_stride,
    ht_ptr,
    ht_stride,
    y_ptr,
    y_stride,
    attention_multiplier,
    cu_seqlens_ptr,
    cu_seqlens_stride,
    S,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    Gq: tl.constexpr,
    Gk: tl.constexpr,
    Gv: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    tl.static_assert(CHUNK_SIZE % BLOCK_SIZE_S == 0)

    if q_ptr is not None:
        tl.static_assert(y_ptr is not None)
    else:
        tl.static_assert(y_ptr is None)

    BLOCK_ID_BN = tl.program_id(0)
    BLOCK_ID_K = tl.program_id(1)
    BLOCK_ID_V = tl.program_id(2)

    BLOCK_ID_B = BLOCK_ID_BN // N
    BLOCK_ID_N = BLOCK_ID_BN % N

    BLOCK_ID_Nq = BLOCK_ID_N // Gq
    BLOCK_ID_Nk = BLOCK_ID_N // Gk
    BLOCK_ID_Nv = BLOCK_ID_N // Gv

    BLOCK_S = tl.arange(0, BLOCK_SIZE_S)
    BLOCK_K = BLOCK_ID_K * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    BLOCK_V = BLOCK_ID_V * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)

    MASK_K = BLOCK_K < K
    MASK_V = BLOCK_V < V

    MASK_KV = MASK_K[:, None] & MASK_V[None, :]
    CAUSAL_MASK = BLOCK_S[:, None] >= BLOCK_S[None, :]

    if h0_ptr is None:
        h = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_V), dtype=tl.float32)
    else:
        h = tl.load(
            h0_ptr
            + BLOCK_ID_B * h0_stride[0]
            + BLOCK_ID_N * h0_stride[1]
            + BLOCK_K[:, None] * h0_stride[2]
            + BLOCK_V[None, :] * h0_stride[3],
            mask=MASK_KV,
        ).to(tl.float32)

    IS_VARLEN: tl.constexpr = cu_seqlens_ptr is not None
    S_DIM: tl.constexpr = 1 - IS_VARLEN
    N_DIM: tl.constexpr = 2 - IS_VARLEN
    K_DIM: tl.constexpr = 3 - IS_VARLEN

    if IS_VARLEN:
        cu_seqlens_ptrs = cu_seqlens_ptr + BLOCK_ID_B * cu_seqlens_stride[0]
        start = tl.load(cu_seqlens_ptrs)
        end = tl.load(cu_seqlens_ptrs + cu_seqlens_stride[0])

        S = end - start
        BLOCK = start + BLOCK_S

    _B = BLOCK[:, None] if IS_VARLEN else BLOCK_ID_B
    _S = 0 if IS_VARLEN else BLOCK_S[:, None]

    if q_ptr is not None:
        q_ptrs = (
            q_ptr
            + _B * q_stride[0]
            + _S * q_stride[S_DIM]
            + BLOCK_ID_Nq * q_stride[N_DIM]
            + BLOCK_K[None, :] * q_stride[K_DIM]
        )

    k_ptrs = (
        k_ptr
        + _B * k_stride[0]
        + _S * k_stride[S_DIM]
        + BLOCK_ID_Nk * k_stride[N_DIM]
        + BLOCK_K[None, :] * k_stride[K_DIM]
    )

    v_ptrs = (
        v_ptr
        + _B * v_stride[0]
        + _S * v_stride[S_DIM]
        + BLOCK_ID_Nv * v_stride[N_DIM]
        + BLOCK_V[None, :] * v_stride[K_DIM]
    )

    if y_ptr is not None:
        y_ptrs = (
            y_ptr
            + _B * y_stride[0]
            + _S * y_stride[S_DIM]
            + BLOCK_ID_N * y_stride[N_DIM]
            + BLOCK_V[None, :] * y_stride[K_DIM]
        )

    if h_ptr is not None:
        h_ptrs = (
            h_ptr
            + _B * h_stride[0]
            + BLOCK_ID_N * h_stride[2]
            + BLOCK_K[:, None] * h_stride[3]
            + BLOCK_V[None, :] * h_stride[4]
        )

    NUM_BLOCKS_S = tl.cdiv(S, BLOCK_SIZE_S)

    for s in range(1, NUM_BLOCKS_S + 1):
        MASK_S = BLOCK_S < S

        MASK_SK = MASK_S[:, None] & MASK_K[None, :]
        MASK_SV = MASK_S[:, None] & MASK_V[None, :]

        k = tl.load(k_ptrs, mask=MASK_SK)
        k_ptrs += BLOCK_SIZE_S * k_stride[S_DIM]

        v = tl.load(v_ptrs, mask=MASK_SV)
        v_ptrs += BLOCK_SIZE_S * v_stride[S_DIM]

        if q_ptr is not None:
            q = tl.load(q_ptrs, mask=MASK_SK)
            q_ptrs += BLOCK_SIZE_S * q_stride[S_DIM]

            qk = matmul(A=q, B=k.T, C=None, output_dtype=q.dtype)
            qk = tl.where(CAUSAL_MASK, qk, 0)

            y = matmul(A=qk, B=v, C=None, output_dtype=tl.float32)
            y = matmul(A=q, B=h.to(q.dtype), C=y, output_dtype=tl.float32)
            y *= attention_multiplier

            tl.store(y_ptrs, y, mask=MASK_SV)
            y_ptrs += BLOCK_SIZE_S * y_stride[S_DIM]

        h = matmul(A=k.T, B=v, C=h, output_dtype=h.dtype)

        if h_ptr is not None and (s * BLOCK_SIZE_S) % CHUNK_SIZE == 0 and s != NUM_BLOCKS_S:
            tl.store(h_ptrs, h, mask=MASK_KV)
            h_ptrs += h_stride[S_DIM]

        BLOCK_S += BLOCK_SIZE_S

    tl.store(
        ht_ptr
        + BLOCK_ID_B * ht_stride[0]
        + BLOCK_ID_N * ht_stride[1]
        + BLOCK_K[:, None] * ht_stride[2]
        + BLOCK_V[None, :] * ht_stride[3],
        h,
        mask=MASK_KV,
    )


def _get_autotune_configs__output_forward_triton_kernel() -> list[triton.Config]:
    configs = []
    for num_warps in get_powers_of_2(4, 8):
        for num_stages in range(1, 5):
            for BLOCK_SIZE_K in get_powers_of_2(32, 64):
                for BLOCK_SIZE_V in get_powers_of_2(32, 64):
                    configs.append(
                        triton.Config(
                            {
                                "BLOCK_SIZE_K": BLOCK_SIZE_K,
                                "BLOCK_SIZE_V": BLOCK_SIZE_V,
                            },
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )

    return configs


@triton.autotune(configs=_get_autotune_configs__output_forward_triton_kernel(), key=[])
@triton.jit
def output_forward_triton_kernel(
    q_ptr,
    q_stride,
    k_ptr,
    k_stride,
    v_ptr,
    v_stride,
    h0_ptr,
    h0_stride,
    h_ptr,
    h_stride,
    y_ptr,
    y_stride,
    attention_multiplier,
    cu_seqlens_ptr,
    cu_seqlens_stride,
    S,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    Gq: tl.constexpr,
    Gk: tl.constexpr,
    Gv: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
):
    BLOCK_ID_BN = tl.program_id(0)
    BLOCK_ID_S = tl.program_id(1)
    BLOCK_ID_V = tl.program_id(2)

    BLOCK_ID_B = BLOCK_ID_BN // N
    BLOCK_ID_N = BLOCK_ID_BN % N

    BLOCK_ID_Nq = BLOCK_ID_N // Gq
    BLOCK_ID_Nk = BLOCK_ID_N // Gk
    BLOCK_ID_Nv = BLOCK_ID_N // Gv

    BLOCK_S = BLOCK_ID_S * BLOCK_SIZE_S + tl.arange(0, BLOCK_SIZE_S)
    BLOCK_K = tl.arange(0, BLOCK_SIZE_K)
    BLOCK_V = BLOCK_ID_V * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)

    MASK_S = BLOCK_S < S
    MASK_V = BLOCK_V < V

    MASK_SV = MASK_S[:, None] & MASK_V[None, :]
    CAUSAL_MASK = (BLOCK_S[:, None] >= BLOCK_S[None, :]) & MASK_S[:, None] & MASK_S[None, :]

    if BLOCK_ID_S == 0:
        if h0_ptr is None:
            h = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_V), dtype=tl.float32)
        else:
            h = tl.load(
                h0_ptr
                + BLOCK_ID_B * h0_stride[0]
                + BLOCK_ID_N * h0_stride[1]
                + BLOCK_K[:, None] * h0_stride[2]
                + BLOCK_V[None, :] * h0_stride[3],
                mask=MASK_KV,
            ).to(tl.float32)
    else:
        h = tl.load(
            h_ptr
            + BLOCK_ID_B * h_stride[0]
            + (BLOCK_ID_S - 1) * h_stride[1]
            + BLOCK_ID_N * h_stride[2]
            + BLOCK_K[:, None] * h_stride[3]
            + BLOCK_V[None, :] * h_stride[4],
            mask=MASK_KV,
        ).to(tl.float32)

    IS_VARLEN: tl.constexpr = cu_seqlens_ptr is not None
    S_DIM: tl.constexpr = 1 - IS_VARLEN
    N_DIM: tl.constexpr = 2 - IS_VARLEN
    K_DIM: tl.constexpr = 3 - IS_VARLEN

    if IS_VARLEN:
        cu_seqlens_ptrs = cu_seqlens_ptr + BLOCK_ID_B * cu_seqlens_stride[0]
        start = tl.load(cu_seqlens_ptrs)
        end = tl.load(cu_seqlens_ptrs + cu_seqlens_stride[0])

        S = end - start
        BLOCK = start + BLOCK_S

    _B = BLOCK[:, None] if IS_VARLEN else BLOCK_ID_B
    _S = 0 if IS_VARLEN else BLOCK_S[:, None]

    q_ptrs = (
        q_ptr
        + _B * q_stride[0]
        + _S * q_stride[S_DIM]
        + BLOCK_ID_Nq * q_stride[N_DIM]
        + BLOCK_K[None, :] * q_stride[K_DIM]
    )

    k_ptrs = (
        k_ptr
        + _B * k_stride[0]
        + _S * k_stride[S_DIM]
        + BLOCK_ID_Nk * k_stride[N_DIM]
        + BLOCK_K[None, :] * k_stride[K_DIM]
    )

    v = tl.load(
        v_ptr
        + _B * v_stride[0]
        + _S * v_stride[S_DIM]
        + BLOCK_ID_Nv * v_stride[N_DIM]
        + BLOCK_V[None, :] * v_stride[K_DIM],
        mask=MASK_SV,
    )

    qk = tl.zeros((BLOCK_SIZE_S, BLOCK_SIZE_S), dtype=tl.float32)
    y = tl.zeros((BLOCK_SIZE_S, BLOCK_SIZE_V), dtype=tl.float32)

    for _ in range(tl.cdiv(K, BLOCK_SIZE_K)):
        MASK_K = BLOCK_K < K

        MASK_SK = MASK_S[:, None] & MASK_K[None, :]
        MASK_KV = MASK_K[:, None] & MASK_V[None, :]

        h = tl.load(
            h_ptr
            + _B * h_stride[0]
            + _S * h_stride[S_DIM]
            + BLOCK_ID_N * h_stride[N_DIM]
            + BLOCK_K[:, None] * h_stride[K_DIM]
            + BLOCK_V[None, :] * h_stride[K_DIM + 1],
            mask=MASK_KV,
        )

        q = tl.load(q_ptrs, mask=MASK_SK)
        q_ptrs += BLOCK_SIZE_S * q_stride[S_DIM]

        k = tl.load(k_ptrs, mask=MASK_SK)
        k_ptrs += BLOCK_SIZE_S * k_stride[S_DIM]

        y = matmul(A=q, B=h, C=y, output_dtype=y.dtype)
        qk = matmul(A=q, B=k.T, C=qk, output_dtype=qk.dtype)

        BLOCK_K += BLOCK_SIZE_K

    qk = tl.where(CAUSAL_MASK, qk, 0)

    y = matmul(A=qk, B=v, C=y, output_dtype=tl.float32)
    y *= attention_multiplier

    tl.store(
        y_ptr
        + _B * y_stride[0]
        + _S * y_stride[S_DIM]
        + BLOCK_ID_N * y_stride[N_DIM]
        + BLOCK_V[None, :] * y_stride[K_DIM],
        y,
        mask=MASK_SV,
    )


@xtune(
    configs=[XTuneConfig({"use_fused_kernel": False}), XTuneConfig({"use_fused_kernel": True})],
    functional_triggers={
        "_": lambda **kwargs: (kwargs["q"].size(1) if kwargs["cu_seqlens"] is None else kwargs["max_seqlen"]) <= 64
    },
)
def _autotuned_linear_attention_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h0: torch.Tensor | None,
    h: torch.Tensor,
    ht: torch.Tensor,
    y: torch.Tensor,
    attention_multiplier: float,
    cu_seqlens: torch.Tensor | None,
    CHUNK_SIZE: int,
    use_fused_kernel: bool,
) -> None:
    Nq, Nk, Nv, N = _get_num_heads(q=q, k=k, v=v, run_check=False)

    if cu_seqlens is None:
        B, S, _, K = k.size()
    else:
        B = cu_seqlens.size(0) - 1
        S = None
        K = k.size(-1)

    V = v.size(-1)

    kwargs = {
        "k_ptr": k,
        "k_stride": k.stride(),
        "v_ptr": v,
        "v_stride": v.stride(),
        "h0_ptr": h0,
        "h0_stride": None if h0 is None else h0.stride(),
        "h_ptr": h,
        "h_stride": None if h is None else h.stride(),
        "attention_multiplier": attention_multiplier,
        "cu_seqlens_ptr": cu_seqlens,
        "cu_seqlens_stride": None if cu_seqlens is None else cu_seqlens.stride(),
        "S": S,
        "N": N,
        "K": K,
        "V": V,
        "Gq": N // Nq,
        "Gk": N // Nk,
        "Gv": N // Nv,
    }

    GRID = lambda kwargs: (B * N, ceil_divide(K, kwargs["BLOCK_SIZE_K"]), ceil_divide(V, kwargs["BLOCK_SIZE_V"]))

    recurrent_state_forward_triton_kernel[GRID](
        q_ptr=q if use_fused_kernel else None,
        q_stride=q.stride() if use_fused_kernel else None,
        ht_ptr=ht,
        ht_stride=ht.stride(),
        y_ptr=y if use_fused_kernel else None,
        y_stride=y.stride() if use_fused_kernel else None,
        CHUNK_SIZE=CHUNK_SIZE,
        **kwargs,
    )

    if not use_fused_kernel:
        NUM_CHUNKS = h.size(1)
        GRID = lambda kwargs: (B * N, NUM_CHUNKS + 1, ceil_divide(V, kwargs["BLOCK_SIZE_V"]))

        output_forward_triton_kernel[GRID](
            q_ptr=q, q_stride=q.stride(), y_ptr=y, y_stride=y.stride(), BLOCK_SIZE_S=CHUNK_SIZE
        )


@xma_op(mutates_args={"y", "h", "ht"})
def linear_attention_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h0: torch.Tensor | None,
    h: torch.Tensor,
    ht: torch.Tensor,
    y: torch.Tensor,
    attention_multiplier: float,
    cu_seqlens: torch.Tensor | None,
    CHUNK_SIZE: int,
) -> None:
    _autotuned_linear_attention_forward_triton(
        q=q,
        k=k,
        v=v,
        h0=h0,
        h=h,
        ht=ht,
        y=y,
        attention_multiplier=attention_multiplier,
        cu_seqlens=cu_seqlens,
        CHUNK_SIZE=CHUNK_SIZE,
    )
