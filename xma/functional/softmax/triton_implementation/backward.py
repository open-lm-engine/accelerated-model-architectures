# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import ceil_divide, get_next_power_of_2


@triton.jit
def softmax_backward_triton_kernel(
    y_ptr,
    y_stride,
    dy_ptr,
    dy_stride,
    dx_ptr,
    dx_stride,
    logits_multiplier,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID = tl.program_id(0)

    BLOCK_B = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    MASK_B = BLOCK_B < B

    accumulator = tl.zeros((BLOCK_SIZE_B, 1), dtype=tl.float32)
    NUM_BLOCKS_H = tl.cdiv(H, BLOCK_SIZE_H)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    y_ptrs = y_ptr + BLOCK_B[:, None] * y_stride[0] + BLOCK_H[None, :] * y_stride[1]
    dy_ptrs = dy_ptr + BLOCK_B[:, None] * dy_stride[0] + BLOCK_H[None, :] * dy_stride[1]

    for _ in range(NUM_BLOCKS_H):
        MASK_H = BLOCK_H < H
        MASK_BH = MASK_B[:, None] & MASK_H[None, :]

        y = tl.load(y_ptrs, mask=MASK_BH)
        dy = tl.load(dy_ptrs, mask=MASK_BH)

        acc = dy * y
        acc = acc.to(tl.float32)
        accumulator += tl.sum(acc, axis=1, keep_dims=True)

        BLOCK_H += BLOCK_SIZE_H
        y_ptrs += BLOCK_SIZE_H * y_stride[1]
        dy_ptrs += BLOCK_SIZE_H * dy_stride[1]

    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    y_ptrs = y_ptr + BLOCK_B[:, None] * y_stride[0] + BLOCK_H[None, :] * y_stride[1]
    dy_ptrs = dy_ptr + BLOCK_B[:, None] * dy_stride[0] + BLOCK_H[None, :] * dy_stride[1]
    dx_ptrs = dx_ptr + BLOCK_B[:, None] * dx_stride[0] + BLOCK_H[None, :] * dx_stride[1]

    for _ in range(NUM_BLOCKS_H):
        MASK_H = BLOCK_H < H
        MASK_BH = MASK_B[:, None] & MASK_H[None, :]

        y = tl.load(y_ptrs, mask=MASK_BH)
        dy = tl.load(dy_ptrs, mask=MASK_BH)

        dy -= accumulator
        y *= dy
        if logits_multiplier is not None:
            y *= logits_multiplier

        tl.store(dx_ptrs, y, mask=MASK_BH)

        BLOCK_H += BLOCK_SIZE_H
        y_ptrs += BLOCK_SIZE_H * y_stride[1]
        dy_ptrs += BLOCK_SIZE_H * dy_stride[1]
        dx_ptrs += BLOCK_SIZE_H * dx_stride[1]


@xma_op(mutates_args={"dx"})
def softmax_backward_triton(
    y: torch.Tensor, dy: torch.Tensor, dx: torch.Tensor, logits_multiplier: float | None
) -> None:
    B, H = dx.size()

    BLOCK_SIZE_B = 1
    BLOCK_SIZE_H = min(get_next_power_of_2(H), 4096 if y.dtype == torch.float32 else 8192)

    softmax_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B),](
        y_ptr=y,
        y_stride=y.stride(),
        dy_ptr=dy,
        dy_stride=dy.stride(),
        dx_ptr=dx,
        dx_stride=dx.stride(),
        logits_multiplier=logits_multiplier,
        B=B,
        H=H,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
