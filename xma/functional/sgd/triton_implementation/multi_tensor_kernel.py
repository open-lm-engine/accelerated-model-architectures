# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from .single_tensor_kernel import _sgd_step


@triton.jit
def multi_tensor_sgd_triton_kernel(
    W_ptr_ptr, dW_ptr_ptr, N_ptr, lr, BLOCK_SIZE: tl.constexpr, MAXIMIZE: tl.constexpr, DTYPE: tl.constexpr
):
    i = tl.program_id(0)

    W_ptr = tl.load(W_ptr_ptr + i).to(tl.pointer_type(DTYPE))
    dW_ptr = tl.load(dW_ptr_ptr + i).to(tl.pointer_type(DTYPE))
    N = tl.load(N_ptr + i)

    for START in range(0, N, BLOCK_SIZE):
        BLOCK = START + tl.arange(0, BLOCK_SIZE)
        MASK = BLOCK < N

        W = tl.load(W_ptr + BLOCK, mask=MASK)
        dW = tl.load(dW_ptr + BLOCK, mask=MASK)

        W = _sgd_step(W=W, dW=dW, lr=lr, MAXIMIZE=MAXIMIZE)
        tl.store(W_ptr + BLOCK, W, mask=MASK)
