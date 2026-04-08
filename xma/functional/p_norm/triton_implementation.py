# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ...constants import MAX_TRITON_BLOCK_SIZE
from ...custom_op import xma_op
from ...math import ceil_divide, get_next_power_of_2, get_powers_of_2
from ...triton_utils import compute_p_norm


@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for num_warps in get_powers_of_2(2, 16)], key=[])
@triton.jit
def p_norm_triton_kernel(
    x_ptr,
    x_stride,
    y_ptr,
    y_stride,
    multiplier,
    B,
    H,
    eps,
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

    y = compute_p_norm(x=x, P=P, P_inv=P_inv, is_P_inf=is_P_inf, eps=eps)
    tl.store(y_ptr + BLOCK_B[:, None] * y_stride[0], y, mask=MASK_B[:, None])


@xma_op(mutates_args={"y"})
def p_norm_triton(x: torch.Tensor, y: torch.Tensor, multiplier: float | None, p: int | None, is_p_inf: bool) -> None:
    B, H = x.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

    GRID = lambda kwargs: (ceil_divide(B, kwargs["BLOCK_SIZE_B"]),)

    p_norm_triton_kernel[GRID](
        x_ptr=x,
        x_stride=x.stride(),
        y_ptr=y,
        y_stride=y.stride(),
        multiplier=multiplier,
        B=B,
        H=H,
        eps=torch.finfo(torch.float32).eps,
        is_P_inf=is_p_inf,
        P=p,
        P_inv=None if is_p_inf else 1 / p,
        BLOCK_SIZE_B=1,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
