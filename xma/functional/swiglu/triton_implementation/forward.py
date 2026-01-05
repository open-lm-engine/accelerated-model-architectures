# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import ceil_divide, get_powers_of_2
from ....triton_utils import sigmoid
from ....utils import get_num_elements_and_hidden_size


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for BLOCK_SIZE_B in get_powers_of_2(16, 64):
        for BLOCK_SIZE_H in get_powers_of_2(16, 64):
            for num_warps in get_powers_of_2(4, 8):
                configs.append(
                    triton.Config({"BLOCK_SIZE_B": BLOCK_SIZE_B, "BLOCK_SIZE_H": BLOCK_SIZE_H}, num_warps=num_warps)
                )

    return configs


@triton.autotune(configs=_get_autotune_configs(), key=[])
@triton.jit
def swiglu_forward_triton_kernel(
    g_ptr,
    g_stride,
    u_ptr,
    u_stride,
    y_ptr,
    y_stride,
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

    y = u * g * sigmoid(g)

    tl.store(y_ptr + BLOCK_B[:, None] * y_stride[0] + BLOCK_H[None, :] * y_stride[1], y, mask=MASK)


@xma_op(mutates_args={"y"})
def swiglu_forward_triton(g: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> None:
    B, H = get_num_elements_and_hidden_size(g)
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), ceil_divide(H, meta["BLOCK_SIZE_H"]))

    # second last stride can be used to iterate the token dimension
    swiglu_forward_triton_kernel[GRID](
        g_ptr=g, g_stride=g.stride(), u_ptr=u, u_stride=u.stride(), y_ptr=y, y_stride=y.stride(), B=B, H=H
    )
