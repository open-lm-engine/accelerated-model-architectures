# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import torch
from parameterized import parameterized

from xma import KernelBackend, ceil_divide, swiglu

from ..test_commons import TestCommons


class SwiGLUTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_2d_tensor_sizes(),  # size
            TestCommons.get_dtypes(),  # dtype
            [KernelBackend.cuda, KernelBackend.triton],  # kernel_backend
            [swiglu, torch.compile(swiglu, fullgraph=True)],  # function
        )
        + TestCommons.make_args_matrix(
            [(4100, 3700)],  # size
            TestCommons.get_dtypes(),  # dtype
            [KernelBackend.nki],  # kernel_backend
            [swiglu],  # function
        )
        + TestCommons.make_args_matrix(
            [(4100, 3700)],  # size
            [torch.float32, torch.bfloat16],  # dtype
            [KernelBackend.pallas],  # kernel_backend
            [swiglu],  # function
        )
    )
    def test_swiglu(
        self,
        size: tuple[int],
        dtype: torch.dtype,
        kernel_backend: KernelBackend,
        function: Callable,
    ) -> None:
        self.skip_if_incompatible_kernel_backend(kernel_backend)
        device = kernel_backend.get_compatible_accelerator().get_current_device()

        if kernel_backend == KernelBackend.cuda:
            multiple = 16 // dtype.itemsize
            size = (size[0], ceil_divide(size[1], multiple) * multiple)

        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        y_kernel, y_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)

        z_kernel = function(x_kernel, y_kernel, kernel_backend=kernel_backend)
        z_expected = swiglu(x_expected, y_expected, kernel_backend=KernelBackend.torch)

        self.assert_equal_tensors(z_kernel, z_expected, False, atol_float32=5.4e-5, rtol_float32=0)

        z_kernel.mean().backward()
        z_expected.mean().backward()

        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, False, atol_float32=5e-6, rtol_float32=0)
        self.assert_equal_tensors(y_kernel.grad, y_expected.grad, False, atol_float32=5e-6, rtol_float32=0)
