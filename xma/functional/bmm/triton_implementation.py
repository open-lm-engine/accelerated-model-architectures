# **************************************************
# Copyright (c) 2026, Mayank Mishra
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
def _bmm_triton_kernel(
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
    L,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # A -> K x M if is_A_transposed else M x K
    # B -> N x K if is_B_transposed else K x N
    # C -> M x N

    BLOCK_ID = tl.program_id(0)
    NUM_BLOCKS = tl.num_programs(0)

    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_SIZE_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_SIZE_N)
    NUM_TILES_MN = NUM_BLOCKS_M * NUM_BLOCKS_N
    total_tiles = L * NUM_TILES_MN

    for tile_idx in range(BLOCK_ID, total_tiles, NUM_BLOCKS):
        BLOCK_ID_L = tile_idx // NUM_TILES_MN
        BLOCK_ID_MN = tile_idx % NUM_TILES_MN

        NUM_BLOCKS_IN_GROUP = GROUP_SIZE_M * NUM_BLOCKS_N
        GROUP_ID = BLOCK_ID_MN // NUM_BLOCKS_IN_GROUP

        FIRST_BLOCK_M_IN_GROUP = GROUP_ID * GROUP_SIZE_M
        CURRENT_GROUP_SIZE_M = min(NUM_BLOCKS_M - FIRST_BLOCK_M_IN_GROUP, GROUP_SIZE_M)

        BLOCK_ID_M = FIRST_BLOCK_M_IN_GROUP + ((BLOCK_ID_MN % NUM_BLOCKS_IN_GROUP) % CURRENT_GROUP_SIZE_M)
        BLOCK_ID_N = (BLOCK_ID_MN % NUM_BLOCKS_IN_GROUP) // CURRENT_GROUP_SIZE_M

        if BLOCK_ID_N < NUM_BLOCKS_N:
            D = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
                if IS_A_TRANSPOSED:
                    A_ptrs = tl.make_block_ptr(
                        A_ptr + BLOCK_ID_L * A_stride[0],
                        shape=(K, M),
                        strides=(A_stride[1], A_stride[2]),
                        offsets=(k * BLOCK_SIZE_K, BLOCK_ID_M * BLOCK_SIZE_M),
                        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_M),
                        order=(1, 0),
                    )
                else:
                    A_ptrs = tl.make_block_ptr(
                        A_ptr + BLOCK_ID_L * A_stride[0],
                        shape=(M, K),
                        strides=(A_stride[1], A_stride[2]),
                        offsets=(BLOCK_ID_M * BLOCK_SIZE_M, k * BLOCK_SIZE_K),
                        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                        order=(1, 0),
                    )

                A = tl.load(A_ptrs, boundary_check=(0, 1))

                if IS_A_TRANSPOSED:
                    A = A.T

                if IS_B_TRANSPOSED:
                    B_ptrs = tl.make_block_ptr(
                        B_ptr + BLOCK_ID_L * B_stride[0],
                        shape=(N, K),
                        strides=(B_stride[1], B_stride[2]),
                        offsets=(BLOCK_ID_N * BLOCK_SIZE_N, k * BLOCK_SIZE_K),
                        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
                        order=(1, 0),
                    )
                else:
                    B_ptrs = tl.make_block_ptr(
                        B_ptr + BLOCK_ID_L * B_stride[0],
                        shape=(K, N),
                        strides=(B_stride[1], B_stride[2]),
                        offsets=(k * BLOCK_SIZE_K, BLOCK_ID_N * BLOCK_SIZE_N),
                        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                        order=(1, 0),
                    )

                B = tl.load(B_ptrs, boundary_check=(0, 1))

                if IS_B_TRANSPOSED:
                    B = B.T

                D = tl.dot(A, B, D, allow_tf32=True)

            if alpha is not None:
                D *= alpha

            if C_ptr is not None:
                C = tl.load(
                    tl.make_block_ptr(
                        C_ptr + BLOCK_ID_L * C_stride[0],
                        shape=(M, N),
                        strides=(C_stride[1], C_stride[2]),
                        offsets=(BLOCK_ID_M * BLOCK_SIZE_M, BLOCK_ID_N * BLOCK_SIZE_N),
                        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
                        order=(1, 0),
                    ),
                    boundary_check=(0, 1),
                )

                if beta is not None:
                    C *= beta

                D += C

            tl.store(
                tl.make_block_ptr(
                    D_ptr + BLOCK_ID_L * D_stride[0],
                    shape=(M, N),
                    strides=(D_stride[1], D_stride[2]),
                    offsets=(BLOCK_ID_M * BLOCK_SIZE_M, BLOCK_ID_N * BLOCK_SIZE_N),
                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
                    order=(1, 0),
                ),
                D.to(D_ptr.dtype.element_ty),
                boundary_check=(0, 1),
            )


@xma_op(mutates_args={"D"})
def _bmm_triton(
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

    NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count

    GRID = lambda kwargs: (
        min(L * ceil_divide(M, kwargs["BLOCK_SIZE_M"]) * ceil_divide(N, kwargs["BLOCK_SIZE_N"]), NUM_SMS),
    )

    _bmm_triton_kernel[GRID](
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
        L=L,
    )
