# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....accelerator import Accelerator
from ....custom_op import xma_op
from ....math import ceil_divide, get_next_power_of_2


@triton.jit
def causal_convolution_triton_kernel(
    x_ptr,
    x_stride,
    h0_ptr,
    h0_stride,
    W_ptr,
    W_stride,
    b_ptr,
    b_stride,
    B,
    S,
    H: tl.constexpr,
    K: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    BLOCK_ID_H = tl.program_id(0)
    BLOCK_ID_S = tl.program_id(1)
    BLOCK_ID_B = tl.program_id(2)

    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = BLOCK_ID_H * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    BLOCK_K = tl.arange(0, BLOCK_SIZE_K)
    BLOCK_S = (BLOCK_ID_S - 1) * K + BLOCK_K + 1

    MASK_B = BLOCK_B < B
    MASK_H = BLOCK_H < H
    MASK_K = BLOCK_K < K

    MASK_BHK = MASK_B[:, None, None] & MASK_H[None, :, None] & MASK_K[None, None, :]

    W = tl.load(
        W_ptr + BLOCK_H[:, None] * W_stride[0] + BLOCK_K[None, :] * W_stride[2], mask=MASK_H[:, None] & MASK_K[None, :]
    )

    x = tl.load(
        x_ptr
        + BLOCK_B[:, None, None] * x_stride[0]
        + BLOCK_K[None, :, None] * x_stride[1]
        + BLOCK_H[None, None, :] * x_stride[2],
        mask=MASK_BHK,
    )

    tl.load(x_ptr + BLOCK_B[:, None] * x_stride[0])


@xma_op(mutates_args={"y"})
def causal_convolution_triton(
    x: torch.Tensor,
    h0: torch.Tensor | None,
    W: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
    activation_function: str,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
) -> None:
    if cu_seqlens is None:
        B, S, H = x.size()
    else:
        T, H = x.size()
        B = cu_seqlens.size(0) - 1
        S = max_seqlen

    K = W.size(-1)

    GRID = lambda kwargs: (
        ceil_divide(kwargs["H"], kwargs["BLOCK_SIZE_H"]),
        ceil_divide(kwargs["S"], kwargs["BLOCK_SIZE_S"]),
        ceil_divide(kwargs["B"], kwargs["BLOCK_SIZE_B"]),
    )

    causal_convolution_triton_kernel[GRID](
        x_ptr=x,
        x_stride=x.stride(),
        h0_ptr=h0,
        h0_stride=h0.stride(),
        W_ptr=W,
        W_stride=W.stride(),
        b_ptr=b,
        b_stride=b.stride(),
        B=B,
        S=S,
        H=H,
        K=K,
        ACTIVATION=activation_function,
        BLOCK_SIZE_K=get_next_power_of_2(K),
    )
