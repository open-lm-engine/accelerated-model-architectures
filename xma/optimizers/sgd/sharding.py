# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from torch.distributed.tensor._op_schema import OpSchema, OpStrategy, RuntimeSchemaInfo
from torch.distributed.tensor._ops import common_pointwise_strategy, register_op_strategy


# The SGD parameter update is element-wise:
#
#   grad = dW  (optionally negated when maximize=True)
#   if weight_decay != 0:  grad += weight_decay * W
#   if momentum != 0:      M = momentum * M + (1 - dampening) * grad
#   if nesterov:           grad = grad + momentum * M
#   else:                  grad = M
#   W -= lr * grad
#
# Every element of W is updated independently, with no cross-element
# communication.  This means any sharding of (W, dW, M) that is consistent
# across those three tensors is valid: the kernel can run locally on each
# device's shard without any inter-device communication.
#
# DTensor's pointwise strategy models exactly this constraint: all tensor
# inputs must carry the same placement (Replicate or Shard(d) for any d),
# and no all-reduce is needed before or after the kernel.
#
# Argument layout of _single_tensor_sgd_triton:
#   0: W             – Tensor   (mutated, parameter being updated)
#   1: dW            – Tensor   (gradient)
#   2: M             – Tensor?  (momentum buffer, mutated; None when momentum=0)
#   3: lr            – float    ← first scalar; RuntimeSchemaInfo(3) marks
#   4: weight_decay  – float      positions 3+ as static for cache-key purposes
#   5: momentum      – float
#   6: dampening     – float
#   7: nesterov      – bool
#   8: maximize      – bool
#   9: is_first_step – bool


@register_op_strategy(
    op=torch.ops.xma._single_tensor_sgd_triton.default,
    # static_argnum=3: scalar args at index ≥ 3 do not affect sharding and
    # are excluded from the cache key used to decide whether to recompute
    # the strategy for a given call site.
    schema_info=RuntimeSchemaInfo(3),
)
def _sgd_dtensor_strategy(op_schema: OpSchema) -> OpStrategy:
    """DTensor sharding strategy for the XMA SGD triton kernel.

    The SGD update is purely element-wise (no reduction across device
    boundaries), so we follow W's (arg 0) placement and require dW (arg 1)
    and the optional momentum buffer M (arg 2) to carry the same placement.
    DTensor will automatically insert redistributions when inputs do not
    already match the required placement.

    Supported placements: Replicate, Shard(d) for any tensor dimension d.
    Partial placements are not supported (gradient accumulation must be
    finalised before the optimizer step).
    """
    # common_pointwise_strategy enumerates every candidate placement stored
    # in W's OpStrategy (Replicate, Shard(0), Shard(1), …) and, for each
    # one, produces an OpSpec that requires every other tensor arg to carry
    # the same placement.  Redistribute costs are computed automatically so
    # the DTensor planner can choose the cheapest resharding when dW or M
    # do not already match W.
    W_strategy: OpStrategy = op_schema.args_schema[0]
    return common_pointwise_strategy(
        op_schema.args_schema,
        followed_strategy=W_strategy,
        followed_strategy_index=0,
    )
