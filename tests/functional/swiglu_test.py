# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from itertools import product
from typing import Callable

import pytest
import torch

from xma import KernelBackend, ceil_divide, swiglu

from ..utils import (
    assert_equal_tensors,
    get_2d_tensor_sizes,
    get_random_duplicated_tensors,
    skip_if_incompatible_kernel_backend,
)


def _generate_args(function: Callable, add_mps: bool) -> list:
    args = list(
        product(
            get_2d_tensor_sizes(),
            [torch.float32, torch.float16, torch.bfloat16],
            [KernelBackend.cuda, KernelBackend.triton],
            [function, torch.compile(function, fullgraph=True)],
        )
    )

    args += list(
        product(
            [(4100, 3700)],
            [torch.float32, torch.float16, torch.bfloat16],
            [KernelBackend.nki],
            [function],
        )
    )

    args += list(
        product(
            [(4100, 3700)],
            [torch.float32, torch.bfloat16],
            [KernelBackend.pallas],
            [function],
        )
    )

    if add_mps:
        args += list(
            product(
                [(4100, 3700)],
                [torch.float32, torch.float16, torch.bfloat16],
                [KernelBackend.mps],
                [function, torch.compile(function, fullgraph=True)],
            )
        )

    return args


@pytest.mark.parametrize("size,dtype,kernel_backend,function", _generate_args(swiglu, add_mps=True))
@torch._dynamo.config.patch(recompile_limit=1024)
def test_swiglu(size: tuple[int], dtype: torch.dtype, kernel_backend: KernelBackend, function: Callable) -> None:
    skip_if_incompatible_kernel_backend(kernel_backend)
    device = kernel_backend.get_compatible_accelerator().get_current_device()

    if kernel_backend == KernelBackend.cuda:
        multiple = 16 // dtype.itemsize
        size = (size[0], ceil_divide(size[1], multiple) * multiple)

    x_kernel, x_expected = get_random_duplicated_tensors(size, device=device, dtype=dtype)
    y_kernel, y_expected = get_random_duplicated_tensors(size, device=device, dtype=dtype)

    z_kernel = function(x_kernel, y_kernel, kernel_backend=kernel_backend)
    z_expected = swiglu(x_expected, y_expected, kernel_backend=KernelBackend.torch)

    assert_equal_tensors(
        z_kernel, z_expected, False, atol_float32=7e-5, rtol_float32=0, atol_float16=1e-3, rtol_float16=0
    )

    z_kernel.mean().backward()
    z_expected.mean().backward()

    assert_equal_tensors(x_kernel.grad, x_expected.grad, False)
    assert_equal_tensors(y_kernel.grad, y_expected.grad, False)
