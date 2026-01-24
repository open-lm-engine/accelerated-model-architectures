# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import triton
import triton.language as tl

from ....triton_utils import matmul


@triton.jit
def linear_attention_backward_triton_kernel(
    h_ptr,
    h_stride,
    dy_ptr,
    dy_stride,
    dq_ptr,
    dq_stride,
    S,
    N,
    K,
    V,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
):
    BLOCK_ID_BN = tl.program_id(0)
    BLOCK_ID_S = tl.program_id(1)
    BLOCK_ID_K = tl.program_id(2)

    BLOCK_ID_B = BLOCK_ID_BN // N
    BLOCK_ID_N = BLOCK_ID_BN % N

    BLOCK_S = BLOCK_ID_S * BLOCK_SIZE_S + tl.arange(0, BLOCK_SIZE_S)
    BLOCK_K = BLOCK_ID_K * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    BLOCK_V = tl.arange(0, BLOCK_SIZE_V)

    MASK_S = BLOCK_S < S
    MASK_K = BLOCK_K < K

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
        + BLOCK_ID_S[:, None] * h_stride[1]
        + BLOCK_ID_N * h_stride[2]
        + BLOCK_K * h_stride[3]
        + BLOCK_V[None, :] * h_stride[4]
    )

    for _ in range(tl.cdiv(V, BLOCK_SIZE_V)):
        MASK_V = BLOCK_V < V

        MASK_SV = MASK_S[:, None] & MASK_V[None, :]
        MASK_KV = MASK_K[:, None] & MASK_V[None, :]

        dy = tl.load(dy_ptrs, mask=MASK_SV)

        h = tl.load(h_ptrs, mask=MASK_KV)

        dq = matmul(A=dy, B=h, C=None, output_dtype=dy.dtype)
