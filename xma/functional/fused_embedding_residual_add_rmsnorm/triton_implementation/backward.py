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
def fused_embedding_residual_add_rmsnorm_backward_triton_kernel(
    # Inputs from forward
    x_ptr,  # token indices (B,)
    x_stride,
    W1_ptr,  # embedding table (V, H)
    W1_stride,
    W2_ptr,  # RMSNorm weight (H,)
    W2_stride,
    s_ptr,  # saved rsqrt scaling factor (B,) - can be None
    s_stride,
    # Upstream gradient
    dy_ptr,  # gradient from upstream (B, H)
    dy_stride,
    # Output gradients
    dW1_ptr,  # gradient for embedding table (V, H) - atomic add
    dW1_stride,
    dW2_ptr,  # gradient for RMSNorm weight (H,) or (num_blocks, H) if deterministic
    dW2_stride,
    # Params
    eps,
    multiplier,
    B,
    H,
    ATOMIC_ADD: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    """
    Backward kernel for fused embedding + RMSNorm.

    Forward was:
        xr = W1[x] * multiplier
        s = rsqrt(mean(xr²) + eps)
        y = xr * s * W2

    Backward computes:
        dW1[x] += dx * multiplier           -- gradient for embedding (scattered)
        dW2 = sum_b(dy * xr * s)            -- gradient for RMSNorm weight
        dx = s * dyW - (1/H) * s³ * xr * sum(dyW * xr)  -- gradient w.r.t. xr
             where dyW = dy * W2
    """
    BLOCK_ID = tl.program_id(0)
    NUM_BLOCKS = tl.num_programs(0)

    # Each block processes a chunk of the batch
    NUM_ELEMENTS_PER_BLOCK = tl.cdiv(B, NUM_BLOCKS)

    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    MASK_H = BLOCK_H < H

    start = BLOCK_ID * NUM_ELEMENTS_PER_BLOCK
    end = min(start + NUM_ELEMENTS_PER_BLOCK, B)
    NUM_ELEMENTS_IN_CURRENT_BLOCK = end - start

    NUM_LOOPS = tl.cdiv(NUM_ELEMENTS_IN_CURRENT_BLOCK, BLOCK_SIZE_B)

    # Load W2 (RMSNorm weight) if provided
    if W2_ptr is not None:
        W2 = tl.load(W2_ptr + BLOCK_H * W2_stride[0], mask=MASK_H)[None, :]
        dW2_acc = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)

    for i in range(NUM_LOOPS):
        BLOCK_B = start + i * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)

        MASK_B = BLOCK_B < end
        MASK_BH = MASK_B[:, None] & MASK_H[None, :]

        # ----- Load token indices -----
        x_indices = tl.load(x_ptr + BLOCK_B, mask=MASK_B)

        # ----- Recompute xr (pre-norm values) from embedding lookup -----
        xr = tl.load(W1_ptr + x_indices[:, None] * W1_stride[0] + BLOCK_H[None, :] * W1_stride[1], mask=MASK_BH).to(
            tl.float32
        )

        if multiplier is not None:
            xr = xr * multiplier

        # ----- Get or recompute scaling factor s -----
        if s_ptr is None:
            # Recompute: s = rsqrt(mean(xr²) + eps)
            s = tl.sum(xr * xr, axis=1)
            s = tl.rsqrt(s / H + eps)
        else:
            # Load saved scaling factor
            s = tl.load(s_ptr + BLOCK_B * s_stride[0], mask=MASK_B)

        # ----- Load upstream gradient dy -----
        dy = tl.load(dy_ptr + BLOCK_B[:, None] * dy_stride[0] + BLOCK_H[None, :] * dy_stride[1], mask=MASK_BH)

        # ----- Compute dyW = dy * W2 -----
        if W2_ptr is not None:
            dyW = (dy * W2).to(tl.float32)
        else:
            dyW = dy.to(tl.float32)

        # ----- Compute dx (gradient w.r.t. xr) -----
        # dx = s * dyW - (1/H) * s³ * xr * sum(dyW * xr)
        s_col = s[:, None]  # (BLOCK_SIZE_B, 1) for broadcasting

        # First term: s * dyW
        dx = s_col * dyW

        # Second term: (1/H) * s³ * xr * sum(dyW * xr)
        dot_product = tl.sum(dyW * xr, axis=1, keep_dims=True)  # (BLOCK_SIZE_B, 1)
        dx = dx - (1.0 / H) * s_col * s_col * s_col * xr * dot_product

        # ----- Compute dW1 gradient and scatter to embedding table -----
        # dW1[x] = dx * multiplier (chain rule: d(xr)/d(W1[x]) = multiplier)
        dW1_grad = dx
        if multiplier is not None:
            dW1_grad = dW1_grad * multiplier

        # Atomic add to dW1 at the token indices
        # Since BLOCK_SIZE_B=1, x_indices is a scalar per iteration
        tl.atomic_add(
            dW1_ptr + x_indices[:, None] * dW1_stride[0] + BLOCK_H[None, :] * dW1_stride[1],
            dW1_grad,
            mask=MASK_BH,
            sem="relaxed",
        )

        # ----- Accumulate dW2 = sum(dy * xr * s) -----
        if W2_ptr is not None:
            # dy * (xr * s) = dy * normalized_x
            dW2_acc += tl.sum(dy * (xr * s_col), axis=0)

    # ----- Store dW2 -----
    if W2_ptr is not None:
        if ATOMIC_ADD:
            tl.atomic_add(dW2_ptr + BLOCK_H * dW2_stride[0], dW2_acc, mask=MASK_H, sem="relaxed")
        else:
            tl.store(dW2_ptr + BLOCK_ID * dW2_stride[0] + BLOCK_H * dW2_stride[1], dW2_acc, mask=MASK_H)


@xma_op(mutates_args={"dW1", "dW2"})
def fused_embedding_residual_add_rmsnorm_backward_triton(
    x: torch.Tensor,  # token indices (B,)
    W1: torch.Tensor,  # embedding table (V, H)
    W2: torch.Tensor | None,  # RMSNorm weight (H,)
    dy: torch.Tensor,  # upstream gradient (B, H)
    s: torch.Tensor | None,  # saved scaling factor (B,) - can be None for memory_efficient
    dW1: torch.Tensor,  # output: gradient for embedding table (V, H)
    dW2: torch.Tensor | None,  # output: gradient for W2 (H,) or (num_blocks, H)
    eps: float,
    multiplier: float | None,
    deterministic: bool,
) -> None:
    B = x.numel()
    H = W1.size(-1)

    BLOCK_SIZE_B = 1
    BLOCK_SIZE_H = get_next_power_of_2(H)
    assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE
    NUM_WARPS = 8

    sm_count = Accelerator.get_sm_count(x.device)
    NUM_BLOCKS = min(sm_count, ceil_divide(B, BLOCK_SIZE_B))

    fused_embedding_residual_add_rmsnorm_backward_triton_kernel[NUM_BLOCKS,](
        x_ptr=x,
        x_stride=x.stride(),
        W1_ptr=W1,
        W1_stride=W1.stride(),
        W2_ptr=W2,
        W2_stride=None if W2 is None else W2.stride(),
        s_ptr=s,
        s_stride=None if s is None else s.stride(),
        dy_ptr=dy,
        dy_stride=dy.stride(),
        dW1_ptr=dW1,
        dW1_stride=dW1.stride(),
        dW2_ptr=dW2,
        dW2_stride=None if dW2 is None else dW2.stride(),
        eps=eps,
        multiplier=multiplier,
        B=B,
        H=H,
        ATOMIC_ADD=not deterministic,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        num_warps=NUM_WARPS,
    )
