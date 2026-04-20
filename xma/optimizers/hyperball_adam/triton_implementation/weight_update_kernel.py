# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from .state_update_kernel import _early_config_prune, _get_3d_grid, _get_autotune_configs


# from torch.distributed.tensor import Partial, Replicate
# from torch.distributed.tensor._op_schema import DTensorSpec, OpSchema, OpStrategy, PlacementStrategy
# from torch.distributed.tensor._ops import pointwise_strategy, register_op_strategy


@triton.autotune(
    configs=_get_autotune_configs(),
    key=[],
    prune_configs_by={"early_config_prune": _early_config_prune},
    reset_to_zero=["W_norm_ptr"],
)
@triton.jit
def _single_tensor_hyperball_weight_update_triton_kernel(
    u_ptr,
    u_stride,
    u_norm_ptr,
    lr,
    R_ptr,
    W_ptr,
    W_stride,
    W_norm_ptr,
    eps,
    X: tl.constexpr,
    Y: tl.constexpr,
    Z: tl.constexpr,
    COMPUTE_W_NORM: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_Z: tl.constexpr,
):
    BLOCK_ID_X = tl.program_id(0)
    BLOCK_X = BLOCK_ID_X * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    MASK = BLOCK_X[:, None, None] < X

    W_ptrs = W_ptr + BLOCK_X[:, None, None] * W_stride[0]
    u_ptrs = u_ptr + BLOCK_X[:, None, None] * u_stride[0]

    if Y is not None:
        BLOCK_ID_Y = tl.program_id(1)
        BLOCK_Y = BLOCK_ID_Y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
        MASK = MASK & (BLOCK_Y[None, :, None] < Y)

        W_ptrs += BLOCK_Y[None, :, None] * W_stride[1]
        u_ptrs += BLOCK_Y[None, :, None] * u_stride[1]

    if Z is not None:
        BLOCK_ID_Z = tl.program_id(2)
        BLOCK_Z = BLOCK_ID_Z * BLOCK_SIZE_Z + tl.arange(0, BLOCK_SIZE_Z)
        MASK = MASK & (BLOCK_Z[None, None, :] < Z)

        W_ptrs += BLOCK_Z[None, None, :] * W_stride[2]
        u_ptrs += BLOCK_Z[None, None, :] * u_stride[2]

    R = tl.load(R_ptr).to(tl.float32)

    u_norm = tl.load(u_norm_ptr).to(tl.float32)
    u_norm = tl.sqrt(u_norm)

    u = tl.load(u_ptrs, mask=MASK).to(tl.float32)
    u /= u_norm + eps

    u *= lr * R

    W = tl.load(W_ptrs, mask=MASK).to(tl.float32)
    W -= u

    if COMPUTE_W_NORM:
        W_norm = tl.sum(W * W)
        tl.atomic_add(W_norm_ptr, W_norm, sem="relaxed")
    else:
        W_norm = tl.load(W_norm_ptr).to(tl.float32)
        W_norm = tl.sqrt(W_norm)
        W /= W_norm + eps
        W *= R
        tl.store(W_ptrs, W, mask=MASK)


@xma_op(mutates_args={"W_norm"})
def _single_tensor_hyperball_weight_norm_triton(
    u: torch.Tensor,
    u_norm: torch.Tensor,
    lr: float,
    R: torch.Tensor,
    W: torch.Tensor,
    W_norm: torch.Tensor,
    eps: float,
) -> None:
    DIMS = W.dim()
    GRID = _get_3d_grid(DIMS)

    _single_tensor_hyperball_weight_update_triton_kernel[GRID](
        u_ptr=u,
        u_stride=u.stride(),
        u_norm_ptr=u_norm,
        lr=lr,
        R_ptr=R,
        W_ptr=W,
        W_stride=W.stride(),
        W_norm_ptr=W_norm,
        eps=eps,
        COMPUTE_W_NORM=True,
        X=W.size(0),
        Y=W.size(1) if DIMS >= 2 else None,
        Z=W.size(2) if DIMS == 3 else None,
    )


@xma_op(mutates_args={"W"})
def _single_tensor_hyperball_weight_update_triton(
    u: torch.Tensor,
    u_norm: torch.Tensor,
    lr: float,
    R: torch.Tensor,
    W: torch.Tensor,
    W_norm: torch.Tensor,
    eps: float,
) -> None:
    DIMS = W.dim()
    GRID = _get_3d_grid(DIMS)

    _single_tensor_hyperball_weight_update_triton_kernel[GRID](
        u_ptr=u,
        u_stride=u.stride(),
        u_norm_ptr=u_norm,
        lr=lr,
        R_ptr=R,
        W_ptr=W,
        W_stride=W.stride(),
        W_norm_ptr=W_norm,
        eps=eps,
        COMPUTE_W_NORM=False,
        X=W.size(0),
        Y=W.size(1) if DIMS >= 2 else None,
        Z=W.size(2) if DIMS == 3 else None,
    )


# @register_op_strategy(torch.ops.xma._single_tensor_hyperball_weight_norm_triton.default)
# def _weight_norm_strategy(mesh, op_schema: OpSchema) -> OpStrategy:
#     # Args: u(0), u_norm(1), lr(2), R(3), W(4), W_norm(5), eps(6)
#     u_spec = op_schema.args_schema[0]
#     if not isinstance(u_spec, DTensorSpec):
#         return pointwise_strategy(mesh, op_schema)

#     shard_placements = u_spec.placements
#     ndim = len(shard_placements)

#     return OpStrategy(
#         [
#             PlacementStrategy(
#                 output_specs=DTensorSpec(mesh, (Partial(),) * ndim),  # W_norm accumulates a partial sum
#                 input_specs=(
#                     DTensorSpec(mesh, shard_placements),  # u: keep Shard
#                     DTensorSpec(mesh, (Replicate(),) * ndim),  # u_norm: Partial→Replicate (triggers all-reduce)
#                     None,  # lr
#                     None,  # R
#                     DTensorSpec(mesh, shard_placements),  # W: keep Shard
#                     None,  # W_norm (local scalar, mutated in-place)
#                     None,  # eps
#                 ),
#             )
#         ]
#     )


# @register_op_strategy(torch.ops.xma._single_tensor_hyperball_weight_update_triton.default)
# def _weight_update_strategy(mesh, op_schema: OpSchema) -> OpStrategy:
#     # Args: u(0), u_norm(1), lr(2), R(3), W(4), W_norm(5), eps(6)
#     u_spec = op_schema.args_schema[0]
#     if not isinstance(u_spec, DTensorSpec):
#         return pointwise_strategy(mesh, op_schema)

#     shard_placements = u_spec.placements
#     ndim = len(shard_placements)

#     return OpStrategy(
#         [
#             PlacementStrategy(
#                 output_specs=DTensorSpec(mesh, shard_placements),  # W stays Shard
#                 input_specs=(
#                     DTensorSpec(mesh, shard_placements),  # u: keep Shard
#                     DTensorSpec(mesh, (Replicate(),) * ndim),  # u_norm: ensure Replicate
#                     None,  # lr
#                     None,  # R
#                     DTensorSpec(mesh, shard_placements),  # W: keep Shard
#                     DTensorSpec(mesh, (Replicate(),) * ndim),  # W_norm: Partial→Replicate (triggers all-reduce)
#                     None,  # eps
#                 ),
#             )
#         ]
#     )
