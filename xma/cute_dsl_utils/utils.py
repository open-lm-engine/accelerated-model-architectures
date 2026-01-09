# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

import cutlass.cute as cute

from ..cute_dsl_utils import get_cute_dtype_from_torch_dtype


def get_fake_cute_tensor(
    dtype: torch.dtype, shape: tuple[int], stride: tuple[int], divisibility: int = 1, leading_dim: int = -1
) -> cute.Tensor:
    if stride[leading_dim] != 1:
        raise ValueError("leading_dim doesn't have stride 1")

    if leading_dim < 0:
        leading_dim = len(shape) + leading_dim

    stride = tuple(1 if i == leading_dim else cute.sym_int64(divisibility=divisibility) for i in range(len(shape)))

    tensor = cute.runtime.make_fake_tensor(
        get_cute_dtype_from_torch_dtype(dtype), shape, stride=stride, assumed_align=divisibility * dtype.itemsize // 8
    )

    return tensor
