# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Callable

import pytest
import torch
import torch._inductor.config as config
import torch.nn as nn
from parameterized import parameterized

from xma import KernelBackend, enable_counters, enable_kernels, get_counter_value, reset_counters, rmsnorm, set_seed
from xma.inductor import _CallablePatternMatcherPass

from ..test_commons import (
    TestCommons,
    assert_equal_tensors,
    get_random_duplicated_tensors,
    skip_if_incompatible_kernel_backend,
)
from .fused_residual_add_rmsnorm_test import _get_sizes


_EPSILON = 1e-5
_SEED = 42


@pytest.mark.parametrize("size", _get_sizes())
@pytest.mark.parametrize("kernel_backend", [KernelBackend.triton])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("memory_efficient", [False, True])
@pytest.mark.parametrize("function", [rmsnorm, torch.compile(rmsnorm, fullgraph=True)])
def test_rmsnorm(
    size: tuple[int],
    kernel_backend: KernelBackend,
    dtype: torch.dtype,
    memory_efficient: bool,
    has_weight: bool,
    function: Callable,
) -> None:
    skip_if_incompatible_kernel_backend(kernel_backend)
    device = kernel_backend.get_compatible_accelerator().get_current_device()

    set_seed(_SEED)

    x_kernel, x_expected = get_random_duplicated_tensors(size, device=device, dtype=dtype)

    if has_weight:
        weight_kernel, weight_expected = get_random_duplicated_tensors(size[-1], device=device, dtype=dtype)
    else:
        weight_kernel = None
        weight_expected = None

    z_kernel = function(
        x=x_kernel,
        weight=weight_kernel,
        eps=_EPSILON,
        memory_efficient=memory_efficient,
        kernel_backend=KernelBackend.triton,
    )

    z_expected = rmsnorm(x=x_expected, weight=weight_expected, eps=_EPSILON, kernel_backend=KernelBackend.torch)

    z_kernel.sum().backward()
    z_expected.sum().backward()

    assert_equal_tensors(z_kernel, z_expected, False, atol_float16=1.6e-2, rtol_float16=0)
    assert_equal_tensors(
        x_kernel.grad,
        x_expected.grad,
        False,
        atol_float32=1.2e-5,
        rtol_float32=0,
        atol_float16=9e-2,
        rtol_float16=0,
    )

    if has_weight:
        assert_equal_tensors(
            weight_kernel.grad,
            weight_expected.grad,
            False,
            atol_float32=6.5e-5,
            rtol_float32=0,
            atol_float16=0.1,
            rtol_float16=0.01,
        )


class RMSNormTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [(4, 4)],  # size
            [KernelBackend.triton],  # KernelBackend
            TestCommons.get_dtypes(),  # dtype
        )
    )
    def test_rmsnorm_kernel_replacement(
        self, size: tuple[int], kernel_backend: KernelBackend, dtype: torch.dtype
    ) -> None:
        class Model(nn.Module):
            def __init__(self) -> Model:
                super().__init__()

                self.h = nn.Sequential(
                    nn.Linear(size[-1], size[-1]), nn.RMSNorm(size[-1]), nn.Linear(size[-1], size[-1])
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.h(x)

        device = torch.cuda.current_device()
        x = torch.randn(size, device=device, dtype=dtype, requires_grad=True)

        with torch.device(device):
            model = Model().to(dtype)

        with torch._inductor.config.patch(
            pattern_matcher=False,
            post_grad_custom_pre_pass=None,
            post_grad_custom_post_pass=_CallablePatternMatcherPass(),
        ):
            enable_kernels([rmsnorm.__name__], config.post_grad_custom_post_pass)

            reset_counters()
            model = torch.compile(model, fullgraph=True)

            with enable_counters():
                model(x)

            assert get_counter_value(f"_FusedResidualAddRMSNorm-{kernel_backend.value}") == 1
