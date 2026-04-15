# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import triton
import triton.language as tl

from ....math import get_powers_of_2
from .single_tensor_kernel import _sgd_step


@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for num_warps in get_powers_of_2(2, 16)], key=[])
@triton.jit
def _multi_tensor_sgd_triton_kernel(
    W_ptr_ptr,
    W_dtype: tl.constexpr,
    dW_ptr_ptr,
    dW_dtype: tl.constexpr,
    M_ptr_ptr,
    M_dtype: tl.constexpr,
    N_ptr,
    lr,
    weight_decay,
    momentum,
    dampening,
    NESTEROV: tl.constexpr,
    MAXIMIZE: tl.constexpr,
    IS_FIRST_STEP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    BLOCK_ID = tl.program_id(0)

    W_ptr = tl.load(W_ptr_ptr + BLOCK_ID).to(tl.pointer_type(W_dtype))
    dW_ptr = tl.load(dW_ptr_ptr + BLOCK_ID).to(tl.pointer_type(dW_dtype))

    if momentum is None:
        tl.static_assert(M_ptr_ptr is None)
        M_ptr = None
    else:
        M_ptr = tl.load(M_ptr_ptr + BLOCK_ID).to(tl.pointer_type(M_dtype))

    N = tl.load(N_ptr + BLOCK_ID)

    for START in range(0, N, BLOCK_SIZE):
        BLOCK = START + tl.arange(0, BLOCK_SIZE)
        MASK = BLOCK < N

        W = tl.load(W_ptr + BLOCK, mask=MASK)
        dW = tl.load(dW_ptr + BLOCK, mask=MASK)

        if M_ptr is not None and not IS_FIRST_STEP:
            M = tl.load(M_ptr + BLOCK, mask=MASK)
        else:
            M = None

        output = _sgd_step(
            W=W,
            dW=dW,
            M=M,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            dampening=dampening,
            NESTEROV=NESTEROV,
            MAXIMIZE=MAXIMIZE,
            IS_FIRST_STEP=IS_FIRST_STEP,
        )

        if M_ptr is None:
            W = output
        else:
            W, M = output
            tl.store(M_ptr + BLOCK, M, mask=MASK)

        tl.store(W_ptr + BLOCK, W, mask=MASK)
