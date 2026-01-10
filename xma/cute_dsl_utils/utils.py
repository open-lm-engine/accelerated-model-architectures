# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from ..cute_dsl_utils import get_cute_dtype_from_torch_dtype
from ..utils import get_alignment


def get_fake_cute_tensor(
    dtype: torch.dtype, shape: tuple[int], divisibility: int = 1, leading_dim: int = -1
) -> cute.Tensor:
    if leading_dim < 0:
        leading_dim = len(shape) + leading_dim

    stride = tuple(1 if i == leading_dim else cute.sym_int64(divisibility=divisibility) for i in range(len(shape)))

    tensor = cute.runtime.make_fake_tensor(
        get_cute_dtype_from_torch_dtype(dtype), shape, stride=stride, assumed_align=divisibility * dtype.itemsize
    )

    return tensor


def torch_tensor_to_cute_tensor(x: torch.Tensor, leading_dim: int) -> cute.Tensor:
    if leading_dim < 0:
        leading_dim += x.dim()

    x = x.detach()
    x = from_dlpack(x, assumed_align=get_alignment(x))

    # not sure if there is a better way to check PyTorch's broadcasting
    if x.stride[leading_dim] == 0:
        leading_dim = None

    x = x.mark_layout_dynamic(leading_dim=leading_dim)

    return x
