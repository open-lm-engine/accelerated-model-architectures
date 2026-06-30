# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import ceil_divide
from ....triton_utils import elementwise_2in_1out_kernel, sigmoid


@triton.jit
def _swiglu_fwd_compute(g, u):
    g = g.to(tl.float32)
    return u * g * sigmoid(g)


@xma_op(mutates_args={"y"})
def _swiglu_forward_triton(g: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> None:
    B, H = g.size()
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), ceil_divide(H, meta["BLOCK_SIZE_H"]))

    elementwise_2in_1out_kernel[GRID](
        x0_ptr=g,
        x0_stride=g.stride(),
        x1_ptr=u,
        x1_stride=u.stride(),
        y_ptr=y,
        y_stride=y.stride(),
        B=B,
        H=H,
        COMPUTE_FN=_swiglu_fwd_compute,
    )
