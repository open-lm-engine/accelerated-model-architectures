# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ....custom_op import xma_op
from ....jit import cpp_jit


@xma_op(mutates_args={"y"})
@cpp_jit(is_mps=True)
def _swiglu_forward_mps(g: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> None: ...


@xma_op(mutates_args={"dg", "du"})
@cpp_jit(is_mps=True)
def _swiglu_backward_mps(
    g: torch.Tensor, u: torch.Tensor, dy: torch.Tensor, dg: torch.Tensor, du: torch.Tensor
) -> None: ...
