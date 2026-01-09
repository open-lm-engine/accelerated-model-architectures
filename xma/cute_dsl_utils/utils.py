# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

import cutlass.cute as cute

from ..cute_dsl_utils import get_cute_dtype_from_torch_dtype


def get_fake_cute_tensor(x: torch.Tensor, divisibility: int = 1, leading_dim: int = -1) -> cute.Tensor:
    if x.stride(leading_dim) != 1:
        raise ValueError("leading_dim doesn't have stride 1")

    if leading_dim < 0:
        leading_dim = x.dim() + leading_dim

    stride = tuple(1 if i == leading_dim else cute.sym_int64(divisibility=divisibility) for i in range(x.dim()))

    tensor = cute.runtime.make_fake_tensor(
        get_cute_dtype_from_torch_dtype(x.dtype),
        x.size(),
        stride=stride,
        assumed_align=divisibility * x.dtype.itemsize // 8,
    )

    return tensor
