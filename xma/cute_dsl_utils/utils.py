# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from ..utils import get_alignment


def torch_tensor_to_cute_tensor(x: torch.Tensor, leading_dim: int) -> cute.Tensor:
    if leading_dim < 0:
        leading_dim += x.dim()

    x = x.detach()
    x = from_dlpack(x, assumed_align=get_alignment(x))

    # not sure if there is a better way to check PyTorch's broadcasting
    if x.stride(leading_dim) == 0:
        leading_dim = None

    x = x.mark_layout_dynamic(leading_dim=leading_dim)

    return x
