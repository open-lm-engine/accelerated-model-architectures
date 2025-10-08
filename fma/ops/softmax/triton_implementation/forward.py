# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....utils import get_num_elements_and_hidden_size


@triton.jit
def softmax_forward_triton_kernel(
    x_ptr,
    x_stride,
    y_ptr,
    y_stride,
    logits_multiplier,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID = tl.program_id(axis=0)

    BLOCK_B = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    MASK_B = BLOCK_B < B

    Z = tl.zeros((BLOCK_SIZE_B, 1), dtype=tl.float32)
    M = tl.full((BLOCK_SIZE_B, 1), -float("inf"), dtype=tl.float32)

    NUM_BLOCKS_H = tl.cdiv(H, BLOCK_SIZE_H)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)

    x_ptrs = x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_H[None, :] * x_stride[1]

    for _ in range(NUM_BLOCKS_H):
        MASK_H = BLOCK_H < H
        MASK_BH = MASK_B[:, None] & MASK_H[None, :]

        x = tl.load(x_ptrs, mask=MASK_BH, other=-float("inf"))

        x = x.to(tl.float32)
        if logits_multiplier is not None:
            x *= logits_multiplier

        prev_m = M
        m = tl.max(x, axis=1, keep_dims=True)
        M = max(M, m)

        x -= M
        x = tl.exp(x)
        Z = Z * tl.exp(prev_m - M) + tl.sum(x, axis=1, keep_dims=True)

        BLOCK_H += BLOCK_SIZE_H
        x_ptrs += BLOCK_SIZE_H * x_stride[1]

    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    x_ptrs = x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_H[None, :] * x_stride[1]
    y_ptrs = y_ptr + BLOCK_B[:, None] * y_stride[0] + BLOCK_H[None, :] * y_stride[1]

    for _ in range(NUM_BLOCKS_H):
        MASK_H = BLOCK_H < H
        MASK_BH = MASK_B[:, None] & MASK_H[None, :]

        x = tl.load(x_ptrs, mask=MASK_BH)

        x = x.to(tl.float32)
        if logits_multiplier is not None:
            x *= logits_multiplier

        x -= M
        x = tl.exp(x)
        x /= Z

        tl.store(y_ptrs, x, mask=MASK_BH)

        BLOCK_H += BLOCK_SIZE_H
        x_ptrs += BLOCK_SIZE_H * x_stride[1]
        y_ptrs += BLOCK_SIZE_H * y_stride[1]


@custom_op(f"{LIBRARY_NAME}::softmax_forward_triton", mutates_args={"output"})
def softmax_forward_triton(x: torch.Tensor, output: torch.Tensor, logits_multiplier: float | None) -> None:
    if x.dim() == 1:
        B = 1
        H = x.size(-1)
    else:
        B, H = get_num_elements_and_hidden_size(x)

    BLOCK_SIZE_B = 1
    BLOCK_SIZE_H = min(get_next_power_of_2(H), 4096 if x.dtype == torch.float32 else 8192)

    with torch.device(x.device):
        softmax_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B),](
            x_ptr=x,
            x_stride=x.stride(),
            y_ptr=output,
            y_stride=output.stride(),
            logits_multiplier=logits_multiplier,
            B=B,
            H=H,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
