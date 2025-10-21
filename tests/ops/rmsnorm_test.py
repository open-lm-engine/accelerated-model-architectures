# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Callable

import torch
from parameterized import parameterized

from fma import KernelBackend, force_kernel_backend, rmsnorm, set_seed

from ..test_commons import TestCommons
from .fused_residual_add_rmsnorm_test import _get_sizes


_EPSILON = 1e-5
_SEED = 42


class RMSNormTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            _get_sizes(),  # size
            [torch.device("cuda")],  # device
            [torch.float32, torch.float16],  # dtype
            [True, False],  # memory_efficient
            [True, False],  # has_weight
            [False],  # deterministic
            [rmsnorm, torch.compile(rmsnorm, fullgraph=True)],  # function
        )
        + TestCommons.make_args_matrix(
            [(400, 77)],  # size
            [torch.device("cuda")],  # device
            [torch.float32, torch.float16],  # dtype
            [True, False],  # memory_efficient
            [True, False],  # has_weight
            [True],  # deterministic
            [rmsnorm, torch.compile(rmsnorm, fullgraph=True)],  # function
        )
    )
    def test_rmsnorm(
        self,
        size: tuple[int],
        device: torch.device,
        dtype: torch.dtype,
        memory_efficient: bool,
        has_weight: bool,
        deterministic: bool,
        function: Callable,
    ) -> None:
        set_seed(_SEED)

        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)

        if has_weight:
            weight_kernel, weight_expected = self.get_random_duplicated_tensors(size[-1], device=device, dtype=dtype)
        else:
            weight_kernel = None
            weight_expected = None

        z_kernel = function(
            x=x_kernel,
            weight=weight_kernel,
            eps=_EPSILON,
            memory_efficient=memory_efficient,
            deterministic=deterministic,
        )

        with force_kernel_backend(KernelBackend.torch):
            z_expected = rmsnorm(x=x_expected, weight=weight_expected, eps=_EPSILON)

        z_kernel.sum().backward()
        z_expected.sum().backward()

        self.assert_equal_tensors(z_kernel, z_expected, False, atol_float16=1.6e-2, rtol_float16=0)
        self.assert_equal_tensors(
            x_kernel.grad,
            x_expected.grad,
            False,
            atol_float32=1.2e-5,
            rtol_float32=0,
            atol_float16=9e-2,
            rtol_float16=0,
        )

        if has_weight:
            self.assert_equal_tensors(
                weight_kernel.grad,
                weight_expected.grad,
                False,
                atol_float32=6.5e-5,
                rtol_float32=0,
                atol_float16=0.1,
                rtol_float16=0.01,
            )

    # def test_rmsnorm_kernel_replacement(self) -> None:
    #     class Model(nn.Module):
    #         def __init__(self) -> Model:
    #             super().__init__()
    #             self.norm = nn.RMSNorm(size[-1])
    #             self.l1 = nn.Linear(size[-1], size[-1])
    #             self.l2 = nn.Linear(size[-1], size[-1])

    #         def forward(self, x: torch.Tensor) -> torch.Tensor:
    #             x = self.l1(x)
    #             x = self.norm(x)
    #             x = self.l2(x)
    #             return x

    #     device = torch.cuda.current_device()
    #     enable_kernels([rmsnorm.__name__])

    #     for size in [(4, 4), (4, 4, 4)]:
    #         for dtype in TestCommons.get_dtypes():
    #             with torch.device(device):
    #                 model = Model().to(dtype)

    #             x = torch.randn(size, device=device, dtype=dtype, requires_grad=True)

    #             reset_counters()
    #             model = torch.compile(model, fullgraph=True)

    #             with enable_counters():
    #                 model(x)

    #             assert get_counter_value(rmsnorm) == 2
