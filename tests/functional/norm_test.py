# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Callable

import torch
from parameterized import parameterized

from xma import KernelBackend, norm, set_seed

from ..test_commons import TestCommons
from .fused_residual_add_rmsnorm_test import _get_sizes


_SEED = 42


class NormTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            _get_sizes(),  # size
            [KernelBackend.triton],  # KernelBackend
            [torch.float32],  # dtype
            [1, 2, 3, "inf"],  # p
            [None, 0.9],  # multiplier
            [norm, torch.compile(norm, fullgraph=True)],  # function
        )
    )
    def test_p_norm(
        self,
        size: tuple[int],
        kernel_backend: KernelBackend,
        dtype: torch.dtype,
        p: int | str,
        multiplier: float | None,
        function: Callable,
    ) -> None:
        self.skip_if_incompatible_kernel_backend(kernel_backend)
        device = kernel_backend.get_compatible_accelerator().get_current_device()

        set_seed(_SEED)

        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)

        z_kernel = function(x=x_kernel, multiplier=multiplier, p=p)
        z_expected = norm(x=x_expected, multiplier=multiplier, p=p, kernel_backend=KernelBackend.torch)

        self.assert_equal_tensors(
            z_kernel, z_expected, False, atol_float32=3.1e-3 if p == 3 else None, rtol_float32=0 if p == 3 else None
        )
