# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton

from ....custom_op import xma_op


@triton.jit
def short_convolution_forward_triton_kernel(): ...


@xma_op(mutates_args={"y", "h"})
def short_convolution_forward_triton(
    x: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None,
    y: torch.Tensor,
    stride: int,
    groups: int,
    h0: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
    activation_function: str,
) -> None:
    K = W.size(0)

    if cu_seqlens is None:
        B, S, H = x.size()
    else:
        T, H = x.size()

    short_convolution_forward_triton_kernel[GRID]()
