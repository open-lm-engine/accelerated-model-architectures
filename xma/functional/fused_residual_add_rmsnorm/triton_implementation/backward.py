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
def fused_residual_add_rmsnorm_backward_triton_kernel(
    xr_ptr,
    xr_stride,
    W_ptr,
    W_stride,
    dy_ptr,
    dy_stride,
    dxr_ptr,
    dxr_stride,
    dx_ptr,
    dx_stride,
    dr_ptr,
    dr_stride,
    dW_ptr,
    dW_stride,
    s_ptr,
    s_stride,
    eps,
    multiplier,
    B,
    H,
    ATOMIC_ADD: tl.constexpr,
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

    if W_ptr is not None:
        W = tl.load(W_ptr + BLOCK_H * W_stride[0], mask=MASK_H)[None, :]
        dW = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)

    for i in range(NUM_LOOPS):
        BLOCK_B = start + i * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)

        MASK_B = BLOCK_B < end
        MASK_BH = MASK_B[:, None] & MASK_H[None, :]

        xr = tl.load(xr_ptr + BLOCK_B[:, None] * xr_stride[0] + BLOCK_H[None, :] * xr_stride[1], mask=MASK_BH).to(
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

    if W_ptr is not None:
        if ATOMIC_ADD:
            tl.atomic_add(dW_ptr + BLOCK_H * dW_stride[0], dW, mask=MASK_H, sem="relaxed")
        else:
            tl.store(dW_ptr + BLOCK_ID * dW_stride[0] + BLOCK_H * dW_stride[1], dW, mask=MASK_H)


@xma_op(mutates_args={"dx", "dr", "dW"})
def fused_residual_add_rmsnorm_backward_triton(
    xr: torch.Tensor,
    W: torch.Tensor | None,
    dy: torch.Tensor,
    dxr: torch.Tensor | None,
    s: torch.Tensor | None,
    dx: torch.Tensor,
    dr: torch.Tensor | None,
    dW: torch.Tensor | None,
    eps: float,
    multiplier: float | None,
    deterministic: bool,
) -> None:
    B, H = xr.size()

    BLOCK_SIZE_B = 1
    BLOCK_SIZE_H = get_next_power_of_2(H)
    assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE
    NUM_WARPS = 8

    sm_count = Accelerator.get_sm_count(xr.device)
    NUM_BLOCKS = min(sm_count, ceil_divide(B, BLOCK_SIZE_B))

    fused_residual_add_rmsnorm_backward_triton_kernel[NUM_BLOCKS,](
        xr_ptr=xr,
        xr_stride=None if xr is None else xr.stride(),
        W_ptr=W,
        W_stride=None if W is None else W.stride(),
        dy_ptr=dy,
        dy_stride=dy.stride(),
        dxr_ptr=dxr,
        dxr_stride=None if dxr is None else dxr.stride(),
        dx_ptr=dx,
        dx_stride=dx.stride(),
        dr_ptr=dr,
        dr_stride=None if dr is None else dr.stride(),
        dW_ptr=dW,
        dW_stride=None if dW is None else dW.stride(),
        s_ptr=s,
        s_stride=None if s is None else s.stride(),
        eps=eps,
        multiplier=multiplier,
        B=B,
        H=H,
        ATOMIC_ADD=not deterministic,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        num_warps=NUM_WARPS,
    )
