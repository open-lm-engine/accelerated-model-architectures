# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import triton
import triton.language as tl

from ....math import get_powers_of_2
from .single_tensor_kernel import _sgd_step


# @triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for num_warps in get_powers_of_2(2, 32)], key=[])
@triton.jit
def _multi_tensor_sgd_triton_kernel(
    W_ptr_ptr,
    dW_ptr_ptr,
    N_ptr,
    lr,
    weight_decay,
    BLOCK_SIZE: tl.constexpr,
    MAXIMIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    BLOCK_ID = tl.program_id(0)

    W_ptr = tl.load(W_ptr_ptr + BLOCK_ID).to(tl.pointer_type(DTYPE))
    dW_ptr = tl.load(dW_ptr_ptr + BLOCK_ID).to(tl.pointer_type(DTYPE))
    N = tl.load(N_ptr + BLOCK_ID)

    for START in range(0, N, BLOCK_SIZE):
        BLOCK = START + tl.arange(0, BLOCK_SIZE)
        MASK = BLOCK < N

        W = tl.load(W_ptr + BLOCK, mask=MASK)
        dW = tl.load(dW_ptr + BLOCK, mask=MASK)

        W = _sgd_step(W=W, dW=dW, lr=lr, weight_decay=weight_decay, MAXIMIZE=MAXIMIZE)
        tl.store(W_ptr + BLOCK, W, mask=MASK)
