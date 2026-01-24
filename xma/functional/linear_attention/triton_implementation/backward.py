# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import ceil_divide
from ....triton_utils import matmul
from ..utils import _get_num_heads
from .output_forward import _get_autotune_configs


@triton.autotune(configs=_get_autotune_configs(), key=[])
@triton.jit
def dq_triton_kernel(
    k_ptr,
    k_stride,
    v_ptr,
    v_stride,
    h_ptr,
    h_stride,
    dy_ptr,
    dy_stride,
    h0_ptr,
    h0_stride,
    dq_ptr,
    dq_stride,
    attention_multiplier,
    S,
    N,
    K,
    V,
    Gq: tl.constexpr,
    Gk: tl.constexpr,
    Gv: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
):
    BLOCK_ID_BN = tl.program_id(0)
    BLOCK_ID_S = tl.program_id(1)
    BLOCK_ID_K = tl.program_id(2)

    NUM_BLOCKS_S = tl.num_programs(1)

    BLOCK_ID_B = BLOCK_ID_BN // N
    BLOCK_ID_N = BLOCK_ID_BN % N

    BLOCK_ID_Nq = BLOCK_ID_N // Gq
    BLOCK_ID_Nk = BLOCK_ID_N // Gk
    BLOCK_ID_Nv = BLOCK_ID_N // Gv

    BLOCK_S = BLOCK_ID_S * BLOCK_SIZE_S + tl.arange(0, BLOCK_SIZE_S)
    BLOCK_K = BLOCK_ID_K * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    BLOCK_V = tl.arange(0, BLOCK_SIZE_V)

    MASK_S = BLOCK_S < S
    MASK_K = BLOCK_K < K

    MASK_SK = MASK_S[:, None] & MASK_K[None, :]

    dy_ptrs = (
        dy_ptr
        + BLOCK_ID_B * dy_stride[0]
        + BLOCK_S[:, None] * dy_stride[1]
        + BLOCK_ID_N * dy_stride[2]
        + BLOCK_V[None, :] * dy_stride[3]
    )

    h_ptrs = (
        h_ptr
        + BLOCK_ID_B * h_stride[0]
        + BLOCK_ID_S * h_stride[1]
        + BLOCK_ID_N * h_stride[2]
        + BLOCK_K[:, None] * h_stride[3]
        + BLOCK_V[None, :] * h_stride[4]
    )

    v_ptrs = (
        v_ptr
        + BLOCK_ID_B * v_stride[0]
        + BLOCK_ID_S[:, None] * v_stride[1]
        + BLOCK_ID_Nv * v_stride[2]
        + BLOCK_V[None, :]
    )

    dq = tl.zeros((BLOCK_SIZE_S, BLOCK_SIZE_K), dtype=tl.float32)
    dyv = tl.zeros((BLOCK_SIZE_S, BLOCK_SIZE_S), dtype=tl.float32)

    for _ in range(tl.cdiv(V, BLOCK_SIZE_V)):
        MASK_V = BLOCK_V < V

        MASK_SV = MASK_S[:, None] & MASK_V[None, :]
        MASK_KV = MASK_K[:, None] & MASK_V[None, :]

        dy = tl.load(dy_ptrs, mask=MASK_SV)
        dy_ptrs += BLOCK_SIZE_V * dy_stride[3]

        if BLOCK_ID_S == 0:
            if h0_ptr is None:
                h = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_V), dtype=dy.dtype)
            else:
                h = tl.load(
                    h0_ptr
                    + BLOCK_ID_B * h0_stride[0]
                    + BLOCK_ID_N * h0_stride[1]
                    + BLOCK_K[:, None] * h0_stride[2]
                    + BLOCK_V[None, :] * h0_stride[3],
                    mask=MASK_KV,
                )
        else:
            h = tl.load(h_ptrs, mask=MASK_KV)
            h_ptrs += BLOCK_SIZE_V * h_stride[4]

        v = tl.load(v_ptrs, mask=MASK_SV)
        v_ptrs += BLOCK_SIZE_V * v_stride[3]

        dq = matmul(A=dy, B=h.T, C=dq, output_dtype=dy.dtype)
        dyv = matmul(A=dy, B=v.T, C=dyv, output_dtype=dyv.dtype)

        BLOCK_V += BLOCK_SIZE_V

    CAUSAL_MASK = BLOCK_S[:, None] >= BLOCK_S[None, :] & MASK_S[:, None] & MASK_S[None, :]
    dyv = tl.where(CAUSAL_MASK, dyv, 0)

    k = tl.load(
        k_ptr
        + BLOCK_ID_B * k_stride[0]
        + BLOCK_S[:, None] * k_stride[1]
        + BLOCK_ID_Nk * k_stride[2]
        + BLOCK_K * k_stride[3],
        mask=MASK_SK,
    )

    dq = matmul(A=dyv, B=k, C=dq, output_dtype=dq.dtype)
    dq *= attention_multiplier

    if Gq == 1:
        tl.store(
            dq_ptr
            + BLOCK_ID_B * dq_stride[0]
            + BLOCK_S[:, None] * dq_stride[1]
            + BLOCK_ID_Nq * dq_stride[2]
            + BLOCK_K[None, :] * dq_stride[3],
            dq,
            mask=MASK_SK,
        )
    else:
        tl.atomic_add(
            dq_ptr
            + BLOCK_ID_B * dq_stride[0]
            + BLOCK_S[:, None] * dq_stride[1]
            + BLOCK_ID_Nq * dq_stride[2]
            + BLOCK_K[None, :] * dq_stride[3],
            dq,
            mask=MASK_SK,
            sem="relaxed",
        )


@xma_op(mutates_args={"dq"})
def dq_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    dy: torch.Tensor,
    h0: torch.Tensor | None,
    dq: torch.Tensor,
    attention_multiplier: float,
    cu_seqlens: torch.Tensor | None,
    CHUNK_SIZE: int,
) -> None:
    Nq, Nk, Nv, N = _get_num_heads(q=q, k=k, v=v, run_check=False)

    if cu_seqlens is None:
        B, S, _, K = k.size()
    else:
        B = cu_seqlens.size(0) - 1
        S = None
        K = k.size(-1)

    V = v.size(-1)

    NUM_CHUNKS = h.size(1)
    GRID = lambda kwargs: (B * N, NUM_CHUNKS + 1, ceil_divide(V, kwargs["BLOCK_SIZE_V"]))

    dq_triton_kernel[GRID](
        v_ptr=v,
        v_stride=None if v is None else v.stride(),
        h_ptr=h,
        h_stride=None if h is None else h.stride(),
        dy_ptr=dy,
        dy_stride=dy.stride(),
        h0_ptr=h0,
        h0_stride=None if h0 is None else h0.stride(),
        dq_ptr=dq,
        dq_stride=dq.stride(),
        attention_multiplier=attention_multiplier,
        S=S,
        N=N,
        K=K,
        V=V,
        Gq=N // Nq,
        Gk=N // Nk,
        Gv=N // Nv,
        BLOCK_SIZE_S=CHUNK_SIZE,
    )
