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

    if momentum is None or M is None:
        M = dW
    else:
        if dampening is not None:
            dW *= 1 - dampening

        M = M.to(tl.float32)
        M *= momentum
        M += dW

    W -= lr * M

    if momentum is None:
        return W
    else:
        return W, M


@triton.autotune(configs=_get_autotune_configs(), key=[], restore_value=["W_ptr"])
@triton.jit
def _single_tensor_sgd_triton_kernel_no_momentum(
    W_ptr,
    dW_ptr,
    N,
    lr,
    weight_decay,
    BLOCK_SIZE: tl.constexpr,
    MAXIMIZE: tl.constexpr,
):
    BLOCK_ID = tl.program_id(0)

    BLOCK = BLOCK_ID * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    MASK = BLOCK < N

    W = tl.load(W_ptr + BLOCK, mask=MASK)
    dW = tl.load(dW_ptr + BLOCK, mask=MASK)

    W = _sgd_step(
        W=W, dW=dW, M=None, lr=lr, weight_decay=weight_decay, momentum=None, dampening=None, MAXIMIZE=MAXIMIZE
    )

    tl.store(W_ptr + BLOCK, W, mask=MASK)


@triton.autotune(configs=_get_autotune_configs(), key=[], restore_value=["W_ptr", "M_ptr"])
@triton.jit
def _single_tensor_sgd_triton_kernel_with_momentum(
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
    M = tl.load(M_ptr + BLOCK, mask=MASK)

    W, M = _sgd_step(
        W=W, dW=dW, M=M, lr=lr, weight_decay=weight_decay, momentum=momentum, dampening=dampening, MAXIMIZE=MAXIMIZE
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

    kwargs = {
        "W_ptr": W,
        "dW_ptr": dW,
        "N": N,
        "lr": lr,
        "weight_decay": None if weight_decay == 0 else weight_decay,
        "MAXIMIZE": maximize,
    }

    if M is None:
        _single_tensor_sgd_triton_kernel_no_momentum[GRID](**kwargs)
    else:
        _single_tensor_sgd_triton_kernel_with_momentum[GRID](
            **kwargs, M_ptr=M, momentum=momentum, dampening=None if dampening == 0 else dampening
        )
