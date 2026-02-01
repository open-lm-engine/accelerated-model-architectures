# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Callable

import torch
from parameterized import parameterized

from xma import KernelBackend, fused_residual_add_rmsnorm, set_seed

from ..test_commons import TestCommons


_EPSILON = 1e-5
_SEED = 42


def _get_sizes() -> list[tuple]:
    sizes = []
    for size in TestCommons.get_1d_tensor_sizes(max_offset=5):
        sizes.append((400, size))

    return sizes


class FusedResdidualAddRMSNormTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            _get_sizes(),  # size
            [KernelBackend.triton],  # KernelBackend
            [torch.float32],  # dtype
            [True, False],  # memory_efficient
            [True, False],  # has_weight
            [None, 0.9],  # multiplier
            [
                fused_residual_add_rmsnorm,
                torch.compile(fused_residual_add_rmsnorm, fullgraph=True),
            ],  # function
        )
    )
    def test_fused_residual_add_rmsnorm(
        self,
        size: tuple[int],
        kernel_backend: KernelBackend,
        dtype: torch.dtype,
        memory_efficient: bool,
        has_weight: bool,
        multiplier: float | None,
        function: Callable,
    ) -> None:
        self.skip_if_incompatible_kernel_backend(kernel_backend)
        device = kernel_backend.get_compatible_accelerator().get_current_device()

        set_seed(_SEED)

        x_kernel, x_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)
        residual_kernel, residual_expected = self.get_random_duplicated_tensors(size, device=device, dtype=dtype)

        if has_weight:
            weight_kernel, weight_expected = self.get_random_duplicated_tensors(size[-1], device=device, dtype=dtype)
        else:
            weight_kernel = None
            weight_expected = None

        z_kernel, r_kernel = function(
            x=x_kernel,
            residual=residual_kernel,
            weight=weight_kernel,
            eps=_EPSILON,
            multiplier=multiplier,
            memory_efficient=memory_efficient,
            kernel_backend=KernelBackend.triton,
        )
        z_kernel = z_kernel * 2 + r_kernel * 3

        z_expected, r_expected = fused_residual_add_rmsnorm(
            x=x_expected,
            residual=residual_expected,
            weight=weight_expected,
            eps=_EPSILON,
            multiplier=multiplier,
            kernel_backend=KernelBackend.torch,
        )
        z_expected = z_expected * 2 + r_expected * 3

        self.assert_equal_tensors(z_kernel, z_expected, False, atol_float32=1.4e-4, rtol_float32=0)

        z_kernel.sum().backward()
        z_expected.sum().backward()

        self.assert_equal_tensors(x_kernel.grad, x_expected.grad, False, atol_float32=1.5e-4, rtol_float32=0)
        self.assert_equal_tensors(
            residual_kernel.grad, residual_expected.grad, False, atol_float32=1.6e-4, rtol_float32=0
        )

        if has_weight:
            self.assert_equal_tensors(
                weight_kernel.grad,
                weight_expected.grad,
                False,
                atol_float32=1.1e-4,
                rtol_float32=0,
            )

    # def test_fused_residual_add_rmsnorm_kernel_replacement(self) -> None:
    #     class Model(nn.Module):
    #         def __init__(self, shape: int) -> Model:
    #             super().__init__()
    #             self.norm = nn.RMSNorm(shape)
    #             self.l1 = nn.Linear(shape, shape)
    #             self.l2 = nn.Linear(shape, shape)

    #         def forward(self, x: torch.Tensor) -> torch.Tensor:
    #             x = self.l1(x)
    #             r = x
    #             x = x + r
    #             x = self.norm(x)
    #             x = self.l2(x)
    #             return x

    #     size = (4, 7)

    #     device = torch.cuda.current_device()
    #     dtype = torch.float32

    #     with torch.device(device):
    #         model = Model(size[-1])

    #     x = torch.randn(size, device=device, dtype=dtype, requires_grad=True)

    #     reset_counters()

    #     enable_kernels([fused_residual_add_rmsnorm.__name__])
    #     model = torch.compile(model)

    #     with enable_counters():
    #         model(x)

    #     assert get_counter_value(fused_residual_add_rmsnorm) == 2
