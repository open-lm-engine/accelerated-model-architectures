# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....constants import MAX_TRITON_BLOCK_SIZE
from ....custom_op import xma_op
from ....math import ceil_divide, get_next_power_of_2


@triton.jit
def norm_forward_triton_kernel(
    x_ptr,
    x_stride,
    y_ptr,
    y_stride,
    multiplier,
    B,
    H,
    is_P_inf: tl.constexpr,
    P: tl.constexpr,
    P_inv: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(0)

    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)

    MASK_B = BLOCK_B < B
    MASK_H = BLOCK_H < H

    MASK_BH = MASK_B[:, None] & MASK_H[None, :]

    x = tl.load(x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_H[None, :] * x_stride[1], mask=MASK_BH)

    if multiplier is not None:
        x *= multiplier

    if is_P_inf:
        x = tl.max(tl.abs(x), axis=1)
    elif P == 1:
        x = tl.sum(tl.abs(x), axis=1)
    elif P == 2:
        x = tl.sum(x * x, axis=1)
    else:
        x = x.to(tl.float32)
        x = tl.abs(x)
        x = tl.log2(x)
        x *= P
        x = tl.exp2(x)
        x = tl.sum(x, axis=1)
        x = tl.log2(x)
        x *= P_inv
        x = tl.exp2(x)

    tl.store(y_ptr + BLOCK_B * y_stride[0], x, mask=MASK_B)


@xma_op(mutates_args={"y"})
def norm_forward_triton(
    x: torch.Tensor, y: torch.Tensor, multiplier: float | None, p: int | None, is_p_inf: bool
) -> None:
    B, H = x.size()

    BLOCK_SIZE_B = 1
    BLOCK_SIZE_H = get_next_power_of_2(H)
    assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE
    NUM_WARPS = 8

    norm_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B),](
        x_ptr=x,
        x_stride=x.stride(),
        y_ptr=y,
        y_stride=y.stride(),
        multiplier=multiplier,
        B=B,
        H=H,
        is_P_inf=is_p_inf,
        P=p,
        P_inv=None if is_p_inf else 1 / p,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        num_warps=NUM_WARPS,
    )
