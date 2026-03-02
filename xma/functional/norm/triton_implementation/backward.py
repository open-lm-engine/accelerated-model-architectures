# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....accelerator import Accelerator
from ....constants import MAX_TRITON_BLOCK_SIZE
from ....custom_op import xma_op
from ....math import ceil_divide, get_next_power_of_2


@triton.jit
def norm_backward_triton_kernel(
    x_ptr,
    x_stride,
    y_ptr,
    y_stride,
    dy_ptr,
    dy_stride,
    dx_ptr,
    dx_stride,
    multiplier,
    B,
    H,
    P: tl.constexpr,
    P_inv: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID = tl.program_id(0)
    NUM_BLOCKS = tl.num_programs(0)

    NUM_ELEMENTS_PER_BLOCK = tl.cdiv(B, NUM_BLOCKS)

    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    MASK_H = BLOCK_H < H

    start = BLOCK_ID * NUM_ELEMENTS_PER_BLOCK
    end = min(start + NUM_ELEMENTS_PER_BLOCK, B)
    NUM_ELEMENTS_IN_CURRENT_BLOCK = end - start

    NUM_LOOPS = tl.cdiv(NUM_ELEMENTS_IN_CURRENT_BLOCK, BLOCK_SIZE_B)

    for i in range(NUM_LOOPS):
        BLOCK_B = start + i * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)

        MASK_B = BLOCK_B < end
        MASK_BH = MASK_B[:, None] & MASK_H[None, :]

        x = tl.load(x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_H[None, :] * x_stride[1], mask=MASK_BH).to(
            tl.float32
        )

        if s_ptr is None:
            r = tl.sum(xr * xr, axis=1)
            r = tl.rsqrt(r / H + eps)
        else:
            r = tl.load(s_ptr + BLOCK_B * s_stride[0], mask=MASK_B)

        dy = tl.load(dy_ptr + BLOCK_B[:, None] * dy_stride[0] + BLOCK_H[None, :] * dy_stride[1], mask=MASK_BH)

        dyW = dy
        if W_ptr is not None:
            dyW *= W

        dyW = dyW.to(tl.float32)

        dx = r[:, None] * dyW
        dx -= (1 / H) * r[:, None] * r[:, None] * r[:, None] * xr * tl.sum(dyW * xr, axis=1, keep_dims=True)

        if dxr_ptr is not None:
            dx += tl.load(dxr_ptr + BLOCK_B[:, None] * dxr_stride[0] + BLOCK_H[None, :] * dxr_stride[1], mask=MASK_BH)

        if dr_ptr is not None:
            tl.store(dr_ptr + BLOCK_B[:, None] * dr_stride[0] + BLOCK_H[None, :] * dr_stride[1], dx, mask=MASK_BH)

        if multiplier is not None:
            dx *= multiplier

        tl.store(dx_ptr + BLOCK_B[:, None] * dx_stride[0] + BLOCK_H[None, :] * dx_stride[1], dx, mask=MASK_BH)

        if W_ptr is not None:
            dW += tl.sum(dy * (xr * r[:, None]), axis=0)


@xma_op(mutates_args={"dx"})
def norm_backward_triton(
    x: torch.Tensor, y: torch.Tensor, dy: torch.Tensor, dx: torch.Tensor, multiplier: float | None, p: int
) -> None:
    B, H = x.size()

    BLOCK_SIZE_B = 1
    BLOCK_SIZE_H = get_next_power_of_2(H)
    assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE
    NUM_WARPS = 8

    sm_count = Accelerator.get_sm_count(x.device)
    NUM_BLOCKS = min(sm_count, ceil_divide(B, BLOCK_SIZE_B))

    norm_backward_triton_kernel[NUM_BLOCKS,](
        x_ptr=x,
        x_stride=x.stride(),
        y_ptr=y,
        y_stride=y.stride(),
        dy_ptr=dy,
        dy_stride=dy.stride(),
        dx_ptr=dx,
        dx_stride=dx.stride(),
        multiplier=multiplier,
        B=B,
        H=H,
        P=p,
        P_inv=1 / p,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        num_warps=NUM_WARPS,
    )
