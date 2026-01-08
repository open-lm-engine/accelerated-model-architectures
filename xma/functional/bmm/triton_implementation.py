# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ...custom_op import xma_op
from ...math import ceil_divide


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2
        ),
    ],
    key=[],
)
@triton.jit
def bmm_triton_kernel(
    A_ptr,
    A_stride,
    B_ptr,
    B_stride,
    C_ptr,
    C_stride,
    D_ptr,
    D_stride,
    alpha,
    beta,
    IS_A_TRANSPOSED: tl.constexpr,
    IS_B_TRANSPOSED: tl.constexpr,
    M,
    K,
    N,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # A -> K x M if is_A_transposed else M x K
    # B -> N x K if is_B_transposed else K x N
    # C -> M x N

    BLOCK_ID = tl.program_id(0)
    BLOCK_ID_L = tl.program_id(1)

    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_SIZE_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_SIZE_N)

    NUM_BLOCKS_IN_GROUP = GROUP_SIZE_M * NUM_BLOCKS_N
    GROUP_ID = BLOCK_ID // NUM_BLOCKS_IN_GROUP

    FIRST_BLOCK_M_IN_GROUP = GROUP_ID * GROUP_SIZE_M
    CURRENT_GROUP_SIZE_M = min(NUM_BLOCKS_M - FIRST_BLOCK_M_IN_GROUP, GROUP_SIZE_M)

    BLOCK_ID_M = FIRST_BLOCK_M_IN_GROUP + ((BLOCK_ID % NUM_BLOCKS_IN_GROUP) % CURRENT_GROUP_SIZE_M)
    BLOCK_ID_N = (BLOCK_ID % NUM_BLOCKS_IN_GROUP) // CURRENT_GROUP_SIZE_M

    if BLOCK_ID_N >= NUM_BLOCKS_N:
        return

    BLOCK_M = BLOCK_ID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    BLOCK_N = BLOCK_ID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    MASK_M = BLOCK_M < M
    MASK_N = BLOCK_N < N

    D = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    BLOCK_K = tl.arange(0, BLOCK_SIZE_K)

    for _ in range(tl.cdiv(K, BLOCK_SIZE_K)):
        MASK_K = BLOCK_K < K

        if IS_A_TRANSPOSED:
            A_ptrs = A_ptr + BLOCK_ID_L * A_stride[0] + BLOCK_K[:, None] * A_stride[1] + BLOCK_M[None, :] * A_stride[2]
            MASK_A = MASK_K[:, None] & MASK_M[None, :]
        else:
            A_ptrs = A_ptr + BLOCK_ID_L * A_stride[0] + BLOCK_M[:, None] * A_stride[1] + BLOCK_K[None, :] * A_stride[2]
            MASK_A = MASK_M[:, None] & MASK_K[None, :]

        A = tl.load(A_ptrs, mask=MASK_A)

        if IS_A_TRANSPOSED:
            A = A.T

        if IS_B_TRANSPOSED:
            B_ptrs = B_ptr + BLOCK_ID_L * B_stride[0] + BLOCK_N[:, None] * B_stride[1] + BLOCK_K[None, :] * B_stride[2]
            MASK_B = MASK_N[:, None] & MASK_K[None, :]
        else:
            B_ptrs = B_ptr + BLOCK_ID_L * B_stride[0] + BLOCK_K[:, None] * B_stride[1] + BLOCK_N[None, :] * B_stride[2]
            MASK_B = MASK_K[:, None] & MASK_N[None, :]

        B = tl.load(B_ptrs, mask=MASK_B)

        if IS_B_TRANSPOSED:
            B = B.T

        D = tl.dot(A, B, D, allow_tf32=True)
        BLOCK_K += BLOCK_SIZE_K

    D = D.to(A_ptr.dtype.element_ty)

    if alpha is not None:
        D *= alpha

    MASK_MN = MASK_M[:, None] & MASK_N[None, :]

    if C_ptr is not None:
        C = tl.load(
            C_ptr + BLOCK_ID_L * C_stride[0] + BLOCK_M[:, None] * C_stride[1] + BLOCK_N[None, :] * C_stride[2],
            mask=MASK_MN,
        )

        if beta is not None:
            C *= beta

        D += C

    tl.store(
        D_ptr + BLOCK_ID_L * D_stride[0] + BLOCK_M[:, None] * D_stride[1] + BLOCK_N[None, :] * D_stride[2],
        D,
        mask=MASK_MN,
    )


@xma_op(mutates_args={"D"})
def bmm_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor | None,
    D: torch.Tensor,
    is_A_transposed: bool,
    is_B_transposed: bool,
    alpha: float,
    beta: float,
) -> None:
    L, M, K = A.size()
    if is_A_transposed:
        M, K = K, M

    N = B.size(1 if is_B_transposed else 2)

    GRID = lambda meta: (
        ceil_divide(M, meta["BLOCK_SIZE_M"]) * ceil_divide(N, meta["BLOCK_SIZE_N"]),
        L,
    )

    bmm_triton_kernel[GRID](
        A_ptr=A,
        A_stride=A.stride(),
        B_ptr=B,
        B_stride=B.stride(),
        C_ptr=C,
        C_stride=None if C is None else C.stride(),
        D_ptr=D,
        D_stride=D.stride(),
        alpha=None if alpha == 1 else alpha,
        beta=None if beta == 1 else beta,
        IS_A_TRANSPOSED=is_A_transposed,
        IS_B_TRANSPOSED=is_B_transposed,
        M=M,
        K=K,
        N=N,
    )
