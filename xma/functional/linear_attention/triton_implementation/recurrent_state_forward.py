# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import ceil_divide, get_powers_of_2
from ....triton_utils import matmul


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for num_warps in get_powers_of_2(4, 8):
        for num_stages in range(1, 5):
            for BLOCK_SIZE_K in get_powers_of_2(32, 64):
                for BLOCK_SIZE_V in get_powers_of_2(32, 64):
                    for BLOCK_SIZE_S in [1] + get_powers_of_2(16, 64):
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_SIZE_K": BLOCK_SIZE_K,
                                    "BLOCK_SIZE_V": BLOCK_SIZE_V,
                                    "BLOCK_SIZE_S": BLOCK_SIZE_S,
                                },
                                num_stages=num_stages,
                                num_warps=num_warps,
                            )
                        )

    return configs


@triton.autotune(configs=_get_autotune_configs(), key=["CHUNK_SIZE"])
@triton.jit
def recurrent_state_forward_triton_kernel(
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
    cu_seqlens_ptr,
    cu_seqlens_stride,
    S,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    Gk: tl.constexpr,
    Gv: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    tl.static_assert(CHUNK_SIZE % BLOCK_SIZE_S == 0)

    BLOCK_ID_BN = tl.program_id(0)
    BLOCK_ID_K = tl.program_id(1)
    BLOCK_ID_V = tl.program_id(2)

    BLOCK_ID_B = BLOCK_ID_BN // N
    BLOCK_ID_N = BLOCK_ID_BN % N

    BLOCK_ID_Nk = BLOCK_ID_N // Gk
    BLOCK_ID_Nv = BLOCK_ID_N // Gv

    BLOCK_K = BLOCK_ID_K * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    BLOCK_V = BLOCK_ID_V * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
    BLOCK_S = tl.arange(0, BLOCK_SIZE_S)

    MASK_K = BLOCK_K < K
    MASK_V = BLOCK_V < V
    MASK_KV = MASK_K[:, None] & MASK_V[None, :]

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

    if IS_VARLEN:
        cu_seqlens_ptrs = cu_seqlens_ptr + BLOCK_ID_B * cu_seqlens_stride[0]
        start = tl.load(cu_seqlens_ptrs)
        end = tl.load(cu_seqlens_ptrs + cu_seqlens_stride[0])

        S = end - start
        BLOCK = start + BLOCK_S

        k_ptrs = k_ptr + BLOCK[:, None] * k_stride[0] + BLOCK_ID_Nk * k_stride[1] + BLOCK_K[None, :] * k_stride[2]
        v_ptrs = v_ptr + BLOCK[:, None] * v_stride[0] + BLOCK_ID_Nv * v_stride[1] + BLOCK_V[None, :] * v_stride[2]

        if h_ptr is not None:
            h_ptrs = (
                h_ptr
                + BLOCK[:, None] * h_stride[0]
                + BLOCK_ID_N * h_stride[1]
                + BLOCK_K[:, None] * h_stride[2]
                + BLOCK_V[None, :] * h_stride[3]
            )
    else:
        k_ptrs = (
            k_ptr
            + BLOCK_ID_B * k_stride[0]
            + BLOCK_S[:, None] * k_stride[1]
            + BLOCK_ID_Nk * k_stride[2]
            + BLOCK_K[None, :] * k_stride[3]
        )

        v_ptrs = (
            v_ptr
            + BLOCK_ID_B * v_stride[0]
            + BLOCK_S[:, None] * v_stride[1]
            + BLOCK_ID_Nv * v_stride[2]
            + BLOCK_V[None, :] * v_stride[3]
        )

        if h_ptr is not None:
            h_ptrs = (
                h_ptr
                + BLOCK_ID_B * h_stride[0]
                + BLOCK_ID_N * h_stride[2]
                + BLOCK_K[:, None] * h_stride[3]
                + BLOCK_V[None, :] * h_stride[4]
            )

    NUM_BLOCKS_S = tl.cdiv(S, BLOCK_SIZE_S)

    for s in range(1, NUM_BLOCKS_S + 1):
        MASK_S = BLOCK_S < S

        k = tl.load(k_ptrs, mask=MASK_S[:, None] & MASK_K[None, :])
        v = tl.load(v_ptrs, mask=MASK_S[:, None] & MASK_V[None, :])

        h = matmul(A=k.T, B=v, C=h, output_dtype=h.dtype)

        if h_ptr is not None and ((s * BLOCK_SIZE_S) % CHUNK_SIZE == 0 or s == NUM_BLOCKS_S):
            tl.store(h_ptrs, h, mask=MASK_KV)
            h_ptrs += h_stride[1 - IS_VARLEN]

        BLOCK_S += BLOCK_SIZE_S
        k_ptrs += BLOCK_SIZE_S * k_stride[1 - IS_VARLEN]
        v_ptrs += BLOCK_SIZE_S * v_stride[1 - IS_VARLEN]

    tl.store(
        ht_ptr
        + BLOCK_ID_B * ht_stride[0]
        + BLOCK_ID_N * ht_stride[1]
        + BLOCK_K[:, None] * ht_stride[2]
        + BLOCK_V[None, :] * ht_stride[3],
        h,
        mask=MASK_KV,
    )


@xma_op(mutates_args={"h", "ht"})
def recurrent_state_forward_triton(
    k: torch.Tensor,
    v: torch.Tensor,
    h0: torch.Tensor | None,
    h: torch.Tensor | None,
    ht: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    CHUNK_SIZE: int,
) -> None:
    if cu_seqlens is None:
        B, S, Nk, K = k.size()
    else:
        B = cu_seqlens.size(0) - 1
        S = None
        _, Nk, K = k.size()

    Nv, V = v.size()[-2:]
    N = h.size(2)

    GRID = lambda kwargs: (B * N, ceil_divide(K, kwargs["BLOCK_SIZE_K"]), ceil_divide(V, kwargs["BLOCK_SIZE_V"]))

    recurrent_state_forward_triton_kernel[GRID](
        k_ptr=k,
        k_stride=k.stride(),
        v_ptr=v,
        v_stride=v.stride(),
        h0_ptr=h0,
        h0_stride=None if h0 is None else h0.stride(),
        h_ptr=h,
        h_stride=None if h is None else h.stride(),
        ht_ptr=ht,
        ht_stride=ht.stride(),
        cu_seqlens_ptr=cu_seqlens,
        cu_seqlens_stride=None if cu_seqlens is None else cu_seqlens.stride(),
        S=S,
        N=N,
        K=K,
        V=V,
        Gk=N // Nk,
        Gv=N // Nv,
        CHUNK_SIZE=CHUNK_SIZE,
    )
