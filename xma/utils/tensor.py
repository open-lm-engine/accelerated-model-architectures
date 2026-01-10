# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ..math import get_powers_of_2


def get_num_elements_and_hidden_size(x: torch.Tensor) -> tuple[int]:
    hidden_size = x.size(-1)
    num_elements = x.numel() // hidden_size
    return num_elements, hidden_size


def empty_like_contiguous(x: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    return torch.empty_like(x, dtype=dtype, memory_format=torch.contiguous_format)


def zeros_like_contiguous(x: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    return torch.zeros_like(x, dtype=dtype, memory_format=torch.contiguous_format)


def get_alignment(x: torch.Tensor) -> int:
    x = x.data_ptr()

    alignment = 4
    for i in get_powers_of_2(4, 16):
        if x % i != 0:
            break
        else:
            alignment = i

    return alignment
