# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import ceil_divide
from ....triton_utils import sigmoid
from ....utils import get_num_elements_and_hidden_size
from .forward import _get_autotune_configs


@triton.autotune(configs=_get_autotune_configs(), key=[])
@triton.jit
def swiglu_backward_triton_kernel(
    g_ptr,
    g_stride,
    u_ptr,
    u_stride,
    dy_ptr,
    dy_stride,
    dg_ptr,
    dg_stride,
    du_ptr,
    du_stride,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(0)
    BLOCK_ID_H = tl.program_id(1)

    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = BLOCK_ID_H * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    MASK_B = BLOCK_B < B
    MASK_H = BLOCK_H < H

    MASK = MASK_B[:, None] & MASK_H[None, :]

    g = tl.load(g_ptr + BLOCK_B[:, None] * g_stride[0] + BLOCK_H[None, :] * g_stride[1], mask=MASK).to(tl.float32)
    u = tl.load(u_ptr + BLOCK_B[:, None] * u_stride[0] + BLOCK_H[None, :] * u_stride[1], mask=MASK)
    dy = tl.load(dy_ptr + BLOCK_B[:, None] * dy_stride[0] + BLOCK_H[None, :] * dy_stride[1], mask=MASK)

    g_sigmoid = sigmoid(g)
    g_silu = g * g_sigmoid

    dg = dy * u * (g_sigmoid + g_silu * (1 - g_sigmoid))
    du = dy * g_silu

    tl.store(dg_ptr + BLOCK_B[:, None] * dg_stride[0] + BLOCK_H[None, :] * dg_stride[1], dg, mask=MASK)
    tl.store(du_ptr + BLOCK_B[:, None] * du_stride[0] + BLOCK_H[None, :] * du_stride[1], du, mask=MASK)


@xma_op(mutates_args={"dg", "du"})
def swiglu_backward_triton(
    g: torch.Tensor, u: torch.Tensor, dy: torch.Tensor, dg: torch.Tensor, du: torch.Tensor
) -> None:
    B, H = get_num_elements_and_hidden_size(g)
    GRID = lambda kwargs: (ceil_divide(B, kwargs["BLOCK_SIZE_B"]), ceil_divide(H, kwargs["BLOCK_SIZE_H"]))

    swiglu_backward_triton_kernel[GRID](
        g_ptr=g,
        g_stride=g.stride(),
        u_ptr=u,
        u_stride=u.stride(),
        dy_ptr=dy,
        dy_stride=dy.stride(),
        dg_ptr=dg,
        dg_stride=dg.stride(),
        du_ptr=du,
        du_stride=du.stride(),
        B=B,
        H=H,
    )
