# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ....custom_op import xma_op
from ....jit import mps_jit


@xma_op(mutates_args={"y"})
@mps_jit(extensions=[".mm"])
def swiglu_forward_mps(g: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> None: ...


@xma_op(mutates_args={"dg", "du"})
@mps_jit(extensions=[".mm"])
def swiglu_backward_mps(
    g: torch.Tensor, u: torch.Tensor, dy: torch.Tensor, dg: torch.Tensor, du: torch.Tensor
) -> None: ...
