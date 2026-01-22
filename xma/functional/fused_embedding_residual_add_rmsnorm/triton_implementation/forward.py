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
def fused_embedding_residual_add_rmsnorm_forward_triton_kernel(
    x_ptr,
    x_stride,
    r_ptr,
    r_stride,
    W1_ptr,
    W1_stride,
    W2_ptr,
    W2_stride,
    y_ptr,
    y_stride,
    xr_ptr,
    xr_stride,
    s_ptr,
    s_stride,
    eps,
    multiplier,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(0)
    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    MASK_B = BLOCK_B < B
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)
    MASK_H = BLOCK_H < H

    MASK_BH = MASK_B[:, None] & MASK_H[None, :]

    # Loading x as a single vector of size BLOCK_SIZE_B
    x = tl.load(x_ptr + BLOCK_B, mask=BLOCK_B < B)

    # Loading the corresponding rows of W1 as a single vector of size BLOCK_SIZE_B
    W1_rows = tl.load(W1_ptr + x[:, None] * W1_stride[0] + BLOCK_H[None, :] * W1_stride[1], mask=MASK_BH)
    x = W1_rows

    # Calculating RMSNorm for each row

    if multiplier is not None:
        x *= multiplier

    # assuming r is always none
    # if r_ptr is not None:
    #     r = tl.load(r_ptr + BLOCK_B[:, None] * r_stride[0] + BLOCK_H[None, :] * r_stride[1], mask=MASK_BH)
    #     x += r

    # if xr_ptr is not None:
    #     tl.store(xr_ptr + BLOCK_B[:, None] * xr_stride[0] + BLOCK_H[None, :] * xr_stride[1], x, mask=MASK_BH)

    r = tl.sum(x * x, axis=1)
    r = tl.rsqrt((r / H) + eps)

    if s_ptr is not None:
        tl.store(s_ptr + BLOCK_B * s_stride[0], r, mask=MASK_B)

    x *= r[:, None]

    if W2_ptr is not None:
        W2 = tl.load(W2_ptr + BLOCK_H * W2_stride[0], mask=MASK_H)
        x = x.to(W1_ptr.dtype.element_ty) * W2[None, :]

    tl.store(y_ptr + BLOCK_B[:, None] * y_stride[0] + BLOCK_H[None, :] * y_stride[1], x, mask=MASK_BH)


@xma_op(mutates_args={"y", "xr", "s"})
def fused_embedding_residual_add_rmsnorm_forward_triton(
    x: torch.Tensor,
    r: torch.Tensor | None,
    W1: torch.Tensor | None,
    W2: torch.Tensor | None,
    y: torch.Tensor,
    eps: float,
    multiplier: float | None,
    xr: torch.Tensor | None,
    s: torch.Tensor | None,
) -> None:
    B = x.numel()  # total number of tokens (x is indices)
    H = W1.size(-1)  # hidden dim from embedding table

    BLOCK_SIZE_B = 1
    BLOCK_SIZE_H = get_next_power_of_2(H)
    assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE
    NUM_WARPS = 8

    fused_embedding_residual_add_rmsnorm_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B),](
        x_ptr=x,
        x_stride=x.stride(),
        r_ptr=r,
        r_stride=None if r is None else r.stride(),
        W1_ptr=W1,
        W1_stride=None if W1 is None else W1.stride(),
        W2_ptr=W2,
        W2_stride=None if W2 is None else W2.stride(),
        y_ptr=y,
        y_stride=y.stride(),
        xr_ptr=xr,
        xr_stride=None if xr is None else xr.stride(),
        s_ptr=s,
        s_stride=None if s is None else s.stride(),
        eps=eps,
        multiplier=multiplier,
        B=B,
        H=H,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        num_warps=NUM_WARPS,
    )
