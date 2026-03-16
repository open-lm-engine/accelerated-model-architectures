# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op


def causal_convolution_triton_kernel(
    x_ptr, x_stride, h0_ptr, h0_stride, W_ptr, W_stride, b_ptr, b_stride, G, S, ACTIVATION: tl.constexpr
):
    tl.load(x)


@xma_op(mutates_args={"y"})
def causal_convolution_triton(
    x: torch.Tensor,
    h0: torch.Tensor | None,
    W: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
    G: int,
    S: int,
    activation_function: str,
) -> None:
    causal_convolution_triton_kernel[1,](
        x_ptr=x,
        x_stride=x.stride(),
        h0_ptr=h0,
        h0_stride=h0.stride(),
        W_ptr=W,
        W_stride=W.stride(),
        b_ptr=b,
        b_stride=b.stride(),
        G=G,
        S=S,
        ACTIVATION=activation_function,
    )
