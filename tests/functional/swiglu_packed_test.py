# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import pytest
import torch

from xma import KernelBackend, ceil_divide, swiglu_packed

from ..utils import assert_equal_tensors, get_random_duplicated_tensors, skip_if_incompatible_kernel_backend
from .swiglu_test import _generate_args


@pytest.mark.parametrize("size,dtype,kernel_backend,function", _generate_args(swiglu_packed, add_mps=False))
@torch._dynamo.config.patch(recompile_limit=1024)
def test_swiglu_packed(
    size: tuple[int],
    dtype: torch.dtype,
    kernel_backend: KernelBackend,
    function: Callable,
) -> None:
    skip_if_incompatible_kernel_backend(kernel_backend)
    device = kernel_backend.get_compatible_accelerator().get_current_device()

    multiple = 2
    if kernel_backend == KernelBackend.cuda:
        multiple *= 16 // dtype.itemsize
    size = (size[0], ceil_divide(size[-1], multiple) * multiple)

    x_kernel, x_expected = get_random_duplicated_tensors(size, device=device, dtype=dtype)

    z_kernel = function(x_kernel, kernel_backend=kernel_backend)
    z_expected = swiglu_packed(x_expected, kernel_backend=KernelBackend.torch)

    assert_equal_tensors(z_kernel, z_expected, False)

    z_kernel.mean().backward()
    z_expected.mean().backward()

    assert_equal_tensors(x_kernel.grad, x_expected.grad, False)
