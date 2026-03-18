# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import ceil_divide, get_next_power_of_2
from ....triton_utils import silu


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
    y_ptr,
    y_stride,
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

    MASK_B = BLOCK_B < B
    MASK_H = BLOCK_H < H
    MASK_K = BLOCK_K < K

    MASK_BSH = MASK_B[:, None, None] & MASK_S[None, :, None] & MASK_H[None, None, :]
    MASK_HK = MASK_H[:, None] & MASK_K[None, :]

    W = tl.load(W_ptr + BLOCK_H[:, None] * W_stride[0] + BLOCK_K[None, :] * W_stride[2], mask=MASK_HK)

    x = tl.load(
        x_ptr
        + BLOCK_B[:, None, None] * x_stride[0]
        + BLOCK_S[None, :, None] * x_stride[1]
        + BLOCK_H[None, None, :] * x_stride[2],
        mask=MASK_BSH,
    )

    W = W.T
    W = W[None, :, :]
    y = W * x

    if b_ptr is not None:
        b = tl.load(b_ptr + BLOCK_H * b_stride[0], mask=MASK_H)
        y = y + b[None, :]

    if ACTIVATION == "swiglu" or ACTIVATION == "silu":
        y = silu(y)

    tl.load(
        y_ptr
        + BLOCK_B[:, None, None] * y_stride[0]
        + BLOCK_S[None, :, None] * y_stride[1]
        + BLOCK_H[None, None, :] * y_stride[2],
        y,
        mask=MASK_BSH,
    )


@xma_op(mutates_args={"y"})
def causal_convolution_triton(
    x: torch.Tensor,
    h0: torch.Tensor | None,
    W: torch.Tensor,
    b: torch.Tensor | None,
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

    if h0 is not None:
        # Update conv state in-place: roll and insert current input
        # h0: [B, H, K], x: [B, 1, H]
        h0.copy_(h0.roll(shifts=-1, dims=-1))
        h0[..., -1] = x[:, 0]

    BLOCK_SIZE_H = max(16, get_next_power_of_2(H))
    BLOCK_SIZE_B = 1

    GRID = lambda kwargs: (
        ceil_divide(kwargs["H"], kwargs["BLOCK_SIZE_H"]),
        S,
        ceil_divide(kwargs["B"], kwargs["BLOCK_SIZE_B"]),
    )

    causal_convolution_triton_kernel[GRID](
        x_ptr=x,
        x_stride=x.stride(),
        h0_ptr=h0,
        h0_stride=None if h0 is None else h0.stride(),
        W_ptr=W,
        W_stride=W.stride(),
        b_ptr=b,
        b_stride=None if b is None else b.stride(),
        y_ptr=y,
        y_stride=y.stride(),
        B=B,
        S=S,
        H=H,
        K=K,
        ACTIVATION=activation_function,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_K=get_next_power_of_2(K),
    )
