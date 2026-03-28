# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import ceil_divide


@triton.jit
def _sgd_step(W, dW, lr, MAXIMIZE):
    if MAXIMIZE:
        dW = -dW

    W -= lr * dW

    return W


@triton.jit
def sgd_triton_kernel(W_ptr, dW_ptr, N, lr, BLOCK_SIZE: tl.constexpr, MAXIMIZE: tl.constexpr):
    BLOCK_ID = tl.program_id(0)

    BLOCK = BLOCK_ID * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    MASK = BLOCK < N

    W = tl.load(W_ptr + BLOCK, mask=MASK)
    dW = tl.load(dW_ptr + BLOCK, mask=MASK)

    W = _sgd_step(W=W, dW=dW, lr=lr, MAXIMIZE=MAXIMIZE)
    tl.store(W_ptr + BLOCK, W, mask=MASK)


@xma_op(mutates_args={"W"})
def sgd_triton(W: torch.Tensor, dW: torch.Tensor, lr: float, maximize: bool) -> None:
    N = W.numel()
    GRID = lambda kwargs: (ceil_divide(N, kwargs["BLOCK_SIZE"]),)

    sgd_triton_kernel[GRID](W_ptr=W, dW_ptr=dW, N=N, lr=lr, MAXIMIZE=maximize)
