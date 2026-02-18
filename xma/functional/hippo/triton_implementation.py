# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ...custom_op import xma_op
from ...math import ceil_divide, get_next_power_of_2, get_powers_of_2
from ...triton_utils import matmul


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for num_warps in get_powers_of_2(4, 8):
        for num_stages in range(1, 5):
            for BLOCK_SIZE_H in [1] + get_powers_of_2(16, 32):
                configs.append(
                    triton.Config({"BLOCK_SIZE_H": BLOCK_SIZE_H}, num_stages=num_stages, num_warps=num_warps)
                )

    return configs


@triton.autotune(configs=_get_autotune_configs(), key=[])
@triton.jit
def hippo_triton_kernel(
    x_ptr,
    x_stride,
    A_ptr,
    A_stride,
    B_ptr,
    B_stride,
    h0_ptr,
    h0_stride,
    h_ptr,
    h_stride,
    S,
    H,
    N,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(0)
    BLOCK_ID_H = tl.program_id(1)

    BLOCK_H = BLOCK_ID_H * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    BLOCK_N = tl.arange(0, BLOCK_SIZE_N)

    MASK_H = BLOCK_H < H
    MASK_N = BLOCK_N < N

    MASK_HN = MASK_H[:, None] & MASK_N[None, :]
    MASK_NN = MASK_N[:, None] & MASK_N[None, :]

    A = tl.load(A_ptr + BLOCK_N[:, None] * A_stride[0] + BLOCK_N[None, :] * A_stride, mask=MASK_NN)
    B = tl.load(B_ptr + BLOCK_N * B_stride[0], mask=MASK_N)

    if h0_ptr is None:
        h = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_N), dtype=x_ptr.dtype.element_ty)
    else:
        h = tl.load(
            h0_ptr + BLOCK_ID_B * h0_stride[0] + BLOCK_H[:, None] * h_stride[2] + BLOCK_N[None, :] * h0_stride[2],
            mask=MASK_HN,
        ).to(tl.float32)

    x_ptrs = x_ptr + BLOCK_ID_B * x_stride[0] + BLOCK_H * x_stride[2]
    h_ptrs = h_ptr + BLOCK_ID_B * h_stride[0] + BLOCK_H * h_stride[2]

    for _ in range(S):
        x = tl.load(x_ptrs, mask=MASK_H)
        x_ptrs += x_stride[1]

        z = matmul(A=x[:, None], B=B[None, :], C=None)
        h = matmul(A=h, B=A.T, C=z, output_dtype=tl.float32)

        tl.store(h_ptrs, h, mask=MASK_HN)
        h_ptrs += h_stride[1]


@xma_op(mutates_args={"h"})
def hippo_triton(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    h0: torch.Tensor | None,
    h: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
) -> None:
    if cu_seqlens is None:
        BS, S, H = x.size()
    else:
        H = x.size(-1)
        BS = cu_seqlens.size(0) - 1
        S = max_seqlen

    N = B.size(0)
    BLOCK_SIZE_N = get_next_power_of_2(N)

    GRID = lambda kwargs: (BS, ceil_divide(H, kwargs["BLOCK_SIZE_H"]))

    hippo_triton_kernel[GRID](
        x_ptr=x,
        x_stride=x.stride(),
        A_ptr=A,
        A_stride=A.stride(),
        B_ptr=B,
        B_stride=B.stride(),
        h0_ptr=h0,
        h0_stride=None if h0 is None else h0.stride(),
        h_ptr=h,
        h_stride=h.stride(),
        S=S,
        H=H,
        N=N,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
