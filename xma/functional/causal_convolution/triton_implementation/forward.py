# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....accelerator import Accelerator
from ....custom_op import xma_op
from ....math import ceil_divide


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
):
    BLOCK_ID = tl.program_id(0)

    BLOCK_B = tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)

    NUM_BLOCKS_B = tl.cdiv(B, BLOCK_SIZE_B)
    NUM_BLOCKS_H = tl.cdiv(H, BLOCK_SIZE_H)
    NUM_BLOCKS_S = tl.cdiv(S, BLOCK_SIZE_S)

    for BLOCK_ID_H in range(NUM_BLOCKS_H):
        for BLOCK_ID_S in range(NUM_BLOCKS_S):
            for BLOCK_ID_B in range(NUM_BLOCKS_B):
                ...

        BLOCK_B += BLOCK_SIZE_B
        BLOCK_H += BLOCK_SIZE_H

    BLOCK_B = BLOCK_ID_B + BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = BLOCK_ID_H + BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

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

    causal_convolution_triton_kernel[Accelerator.get_sm_count(),](
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
    )
