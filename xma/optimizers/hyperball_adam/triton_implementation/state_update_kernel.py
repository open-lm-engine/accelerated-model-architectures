# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import ceil_divide, get_powers_of_2


# from torch.distributed.tensor._ops import pointwise_strategy, register_op_strategy


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for BLOCK_SIZE_X in get_powers_of_2(4, 1024):
        for BLOCK_SIZE_Y in get_powers_of_2(4, 1024):
            for BLOCK_SIZE_Z in get_powers_of_2(4, 1024):
                total = BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z
                if BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z > 8192:
                    continue

                for num_warps in get_powers_of_2(4, max(4, total >> 5)):
                    configs.append(
                        triton.Config(
                            {"BLOCK_SIZE_X": BLOCK_SIZE_X, "BLOCK_SIZE_Y": BLOCK_SIZE_Y, "BLOCK_SIZE_Z": BLOCK_SIZE_Z},
                            num_warps=num_warps,
                        )
                    )

    return configs


def _early_config_prune(configs: list[triton.Config], named_args: dict, **_kwargs) -> list[triton.Config]:
    Y = named_args.get("Y")
    Z = named_args.get("Z")

    pruned = []
    for config in configs:
        BLOCK_SIZE_Y = config.kwargs["BLOCK_SIZE_Y"]
        BLOCK_SIZE_Z = config.kwargs["BLOCK_SIZE_Z"]

        if Y is None:
            if BLOCK_SIZE_Y != 4 or BLOCK_SIZE_Z != 4:
                continue
        elif Z is None:
            if BLOCK_SIZE_Z != 4:
                continue

        pruned.append(config)

    return pruned


@triton.autotune(
    configs=_get_autotune_configs(),
    key=[],
    prune_configs_by={"early_config_prune": _early_config_prune},
    reset_to_zero=["u_norm_ptr"],
)
@triton.jit
def _single_tensor_hyperball_state_update_triton_kernel(
    exp_avg_ptr,
    exp_avg_stride,
    exp_avg_sq_ptr,
    exp_avg_sq_stride,
    dW_ptr,
    dW_stride,
    u_ptr,
    u_stride,
    u_norm_ptr,
    beta1,
    beta2,
    bc1,
    bc2,
    eps,
    X: tl.constexpr,
    Y: tl.constexpr,
    Z: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_Z: tl.constexpr,
):
    BLOCK_ID_X = tl.program_id(0)
    BLOCK_X = BLOCK_ID_X * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    MASK = BLOCK_X[:, None, None] < X

    exp_avg_ptrs = exp_avg_ptr + BLOCK_X[:, None, None] * exp_avg_stride[0]
    exp_avg_sq_ptrs = exp_avg_sq_ptr + BLOCK_X[:, None, None] * exp_avg_sq_stride[0]
    dW_ptrs = dW_ptr + BLOCK_X[:, None, None] * dW_stride[0]
    u_ptrs = u_ptr + BLOCK_X[:, None, None] * u_stride[0]

    if Y is not None:
        BLOCK_ID_Y = tl.program_id(1)
        BLOCK_Y = BLOCK_ID_Y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
        MASK = MASK & (BLOCK_Y[None, :, None] < Y)

        exp_avg_ptrs += BLOCK_Y[None, :, None] * exp_avg_stride[1]
        exp_avg_sq_ptrs += BLOCK_Y[None, :, None] * exp_avg_sq_stride[1]
        dW_ptrs += BLOCK_Y[None, :, None] * dW_stride[1]
        u_ptrs += BLOCK_Y[None, :, None] * u_stride[1]

    if Z is not None:
        BLOCK_ID_Z = tl.program_id(2)
        BLOCK_Z = BLOCK_ID_Z * BLOCK_SIZE_Z + tl.arange(0, BLOCK_SIZE_Z)
        MASK = MASK & (BLOCK_Z[None, None, :] < Z)

        exp_avg_ptrs += BLOCK_Z[None, None, :] * exp_avg_stride[2]
        exp_avg_sq_ptrs += BLOCK_Z[None, None, :] * exp_avg_sq_stride[2]
        dW_ptrs += BLOCK_Z[None, None, :] * dW_stride[2]
        u_ptrs += BLOCK_Z[None, None, :] * u_stride[2]

    dW = tl.load(dW_ptrs, mask=MASK).to(tl.float32)

    exp_avg = tl.load(exp_avg_ptrs, mask=MASK).to(tl.float32)
    exp_avg *= beta1
    exp_avg += dW * (1 - beta1)
    tl.store(exp_avg_ptrs, exp_avg, mask=MASK)

    exp_avg_sq = tl.load(exp_avg_sq_ptrs, mask=MASK).to(tl.float32)
    exp_avg_sq *= beta2
    exp_avg_sq += dW * dW * (1 - beta2)
    tl.store(exp_avg_sq_ptrs, exp_avg_sq, mask=MASK)

    u = exp_avg * bc1 / (tl.sqrt(exp_avg_sq * bc2) + eps)
    tl.store(u_ptrs, u, mask=MASK)

    u_norm = tl.sum(u * u)
    tl.atomic_add(u_norm_ptr, u_norm, sem="relaxed")


def _get_3d_grid(DIMS: int) -> Callable:
    if DIMS == 1:
        GRID = lambda kwargs: (ceil_divide(kwargs["X"], kwargs["BLOCK_SIZE_X"]),)
    elif DIMS == 2:
        GRID = lambda kwargs: (
            ceil_divide(kwargs["X"], kwargs["BLOCK_SIZE_X"]),
            ceil_divide(kwargs["Y"], kwargs["BLOCK_SIZE_Y"]),
        )
    elif DIMS == 3:
        GRID = lambda kwargs: (
            ceil_divide(kwargs["X"], kwargs["BLOCK_SIZE_X"]),
            ceil_divide(kwargs["Y"], kwargs["BLOCK_SIZE_Y"]),
            ceil_divide(kwargs["Z"], kwargs["BLOCK_SIZE_Z"]),
        )
    else:
        raise ValueError

    return GRID


@xma_op(mutates_args={"exp_avg", "exp_avg_sq", "u", "u_norm"})
def _single_tensor_hyperball_state_update_triton(
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    dW: torch.Tensor,
    u: torch.Tensor,
    u_norm: torch.Tensor,
    beta1: float,
    beta2: float,
    t: int,
    eps: float,
) -> None:
    DIMS = dW.dim()
    GRID = _get_3d_grid(DIMS)

    bc1 = 1 / (1 - beta1**t)
    bc2 = 1 / (1 - beta2**t)

    _single_tensor_hyperball_state_update_triton_kernel[GRID](
        exp_avg_ptr=exp_avg,
        exp_avg_stride=exp_avg.stride(),
        exp_avg_sq_ptr=exp_avg_sq,
        exp_avg_sq_stride=exp_avg_sq.stride(),
        dW_ptr=dW,
        dW_stride=dW.stride(),
        u_ptr=u,
        u_stride=u.stride(),
        u_norm_ptr=u_norm,
        beta1=beta1,
        beta2=beta2,
        bc1=bc1,
        bc2=bc2,
        eps=eps,
        X=dW.size(0),
        Y=dW.size(1) if DIMS >= 2 else None,
        Z=dW.size(2) if DIMS == 3 else None,
    )


# register_op_strategy(torch.ops.xma._single_tensor_hyperball_state_update_triton.default)(pointwise_strategy)
