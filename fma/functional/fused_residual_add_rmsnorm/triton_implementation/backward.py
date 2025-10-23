# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ....constants import LIBRARY_NAME, MAX_TRITON_BLOCK_SIZE
from ....math import ceil_divide, get_next_power_of_2
from ....utils import get_num_elements_and_hidden_size, get_sm_count


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
    BLOCK_ID = tl.program_id(axis=0)
    NUM_BLOCKS = tl.num_programs(axis=0)

    NUM_ELEMENTS_PER_BLOCK = tl.cdiv(B, NUM_BLOCKS)

    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    MASK_H = BLOCK_H < H

    start = BLOCK_ID * NUM_ELEMENTS_PER_BLOCK
    end = min(start + NUM_ELEMENTS_PER_BLOCK, B)
    NUM_ELEMENTS_IN_CURRENT_BLOCK = end - start

    NUM_LOOPS = tl.cdiv(NUM_ELEMENTS_IN_CURRENT_BLOCK, BLOCK_SIZE_B)

    x_dtype = xr_ptr.dtype.element_ty

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

        dx = dx.to(x_dtype)

        if dxr_ptr is not None:
            dx += tl.load(dxr_ptr + BLOCK_B[:, None] * dxr_stride[0] + BLOCK_H[None, :] * dxr_stride[1], mask=MASK_BH)

        if dr_ptr is not None:
            tl.store(dr_ptr + BLOCK_B[:, None] * dr_stride[0] + BLOCK_H[None, :] * dr_stride[1], dx, mask=MASK_BH)

        if multiplier is not None:
            dx *= multiplier

        tl.store(dx_ptr + BLOCK_B[:, None] * dx_stride[0] + BLOCK_H[None, :] * dx_stride[1], dx, mask=MASK_BH)

        if W_ptr is not None:
            dW += tl.sum(dy * (xr * r[:, None]).to(x_dtype), axis=0)

    if W_ptr is not None:
        if ATOMIC_ADD:
            tl.atomic_add(dW_ptr + BLOCK_H * dW_stride[0], dW, mask=MASK_H, sem="relaxed")
        else:
            tl.store(dW_ptr + BLOCK_ID * dW_stride[0] + BLOCK_H * dW_stride[1], dW, mask=MASK_H)


@custom_op(
    f"{LIBRARY_NAME}::fused_residual_add_rmsnorm_backward_triton",
    mutates_args={"x_grad", "residual_grad", "weight_grad"},
)
def fused_residual_add_rmsnorm_backward_triton(
    added_x_residual: torch.Tensor,
    weight: torch.Tensor | None,
    output_grad: torch.Tensor,
    added_x_residual_grad: torch.Tensor | None,
    rmsnorm_denominator: torch.Tensor | None,
    x_grad: torch.Tensor,
    residual_grad: torch.Tensor | None,
    weight_grad: torch.Tensor | None,
    eps: float,
    multiplier: float | None,
    deterministic: bool,
) -> None:
    B, H = get_num_elements_and_hidden_size(added_x_residual)

    BLOCK_SIZE_B = 1
    BLOCK_SIZE_H = get_next_power_of_2(H)
    assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE
    NUM_WARPS = 8

    sm_count = get_sm_count(added_x_residual.device)
    NUM_BLOCKS = min(sm_count, ceil_divide(B, BLOCK_SIZE_B))

    with torch.device(added_x_residual.device):
        fused_residual_add_rmsnorm_backward_triton_kernel[NUM_BLOCKS,](
            xr_ptr=added_x_residual,
            xr_stride=None if added_x_residual is None else added_x_residual.stride(),
            W_ptr=weight,
            W_stride=None if weight is None else weight.stride(),
            dy_ptr=output_grad,
            dy_stride=output_grad.stride(),
            dxr_ptr=added_x_residual_grad,
            dxr_stride=None if added_x_residual_grad is None else added_x_residual_grad.stride(),
            dx_ptr=x_grad,
            dx_stride=x_grad.stride(),
            dr_ptr=residual_grad,
            dr_stride=None if residual_grad is None else residual_grad.stride(),
            dW_ptr=weight_grad,
            dW_stride=None if weight_grad is None else weight_grad.stride(),
            s_ptr=rmsnorm_denominator,
            s_stride=None if rmsnorm_denominator is None else rmsnorm_denominator.stride(),
            eps=eps,
            multiplier=multiplier,
            B=B,
            H=H,
            ATOMIC_ADD=not deterministic,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            num_warps=NUM_WARPS,
        )
