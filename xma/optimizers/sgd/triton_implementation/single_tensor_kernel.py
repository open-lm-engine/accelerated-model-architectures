# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import ceil_divide, get_powers_of_2


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for num_warps in get_powers_of_2(4, 32):
        for BLOCK_SIZE in get_powers_of_2(64, 8192):
            configs.append(triton.Config({"BLOCK_SIZE": BLOCK_SIZE}, num_warps=num_warps))

    return configs


@triton.jit
def _sgd_step(W, dW, M, lr, weight_decay, momentum, dampening, MAXIMIZE):
    W = W.to(tl.float32)
    dW = dW.to(tl.float32)

    if MAXIMIZE:
        dW = -dW

    if weight_decay is not None:
        dW += weight_decay * W

    if momentum is not None:
        if M is None:
            M = dW
        else:
            M = M.to(tl.float32)
            M = momentum * M + (1 - dampening) * dW

        dW = M

    W -= lr * dW

    if M is None:
        return W
    else:
        return W, M


@triton.autotune(configs=_get_autotune_configs(), key=[], restore_value=["W_ptr"])
@triton.jit
def _single_tensor_sgd_triton_kernel(
    W_ptr,
    dW_ptr,
    M_ptr,
    N,
    lr,
    weight_decay,
    momentum,
    dampening,
    BLOCK_SIZE: tl.constexpr,
    MAXIMIZE: tl.constexpr,
):
    BLOCK_ID = tl.program_id(0)

    BLOCK = BLOCK_ID * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    MASK = BLOCK < N

    W = tl.load(W_ptr + BLOCK, mask=MASK)
    dW = tl.load(dW_ptr + BLOCK, mask=MASK)

    if momentum is None:
        tl.static_assert(M_ptr is None)
        W = _sgd_step(
            W=W,
            dW=dW,
            M=None,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            dampening=dampening,
            MAXIMIZE=MAXIMIZE,
        )
    else:
        M = tl.load(M_ptr + BLOCK, mask=MASK)

        W, M = _sgd_step(
            W=W,
            dW=dW,
            M=M,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            dampening=dampening,
            MAXIMIZE=MAXIMIZE,
        )

        tl.store(M_ptr + BLOCK, M, mask=MASK)

    tl.store(W_ptr + BLOCK, W, mask=MASK)


@xma_op(mutates_args={"W", "M"})
def _single_tensor_sgd_triton(
    W: torch.Tensor,
    dW: torch.Tensor,
    M: torch.Tensor | None,
    lr: float,
    weight_decay: float,
    momentum: float,
    dampening: float,
    maximize: bool,
) -> None:
    N = W.numel()
    GRID = lambda kwargs: (ceil_divide(N, kwargs["BLOCK_SIZE"]),)

    _single_tensor_sgd_triton_kernel[GRID](
        W_ptr=W,
        dW_ptr=dW,
        M_ptr=M,
        N=N,
        lr=lr,
        weight_decay=None if weight_decay == 0 else weight_decay,
        dampening=None if dampening == 0 else dampening,
        momentum=None if momentum == 0 else momentum,
        MAXIMIZE=maximize,
    )
