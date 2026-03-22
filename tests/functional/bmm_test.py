# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import pytest
import torch

from xma import KernelBackend, bmm, set_seed

from ..test_commons import assert_equal_tensors, get_2d_tensor_sizes, skip_if_incompatible_kernel_backend


_SEED = 42


@pytest.mark.parametrize("size", get_2d_tensor_sizes())
@pytest.mark.parametrize("is_A_transposed", [False, True])
@pytest.mark.parametrize("is_B_transposed", [False, True])
@pytest.mark.parametrize("has_C", [False, True])
@pytest.mark.parametrize("kernel_backend", [KernelBackend.triton])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("function", [bmm, torch.compile(bmm, fullgraph=True)])
def test_bmm(
    size: tuple[int],
    is_A_transposed: bool,
    is_B_transposed: bool,
    has_C: bool,
    kernel_backend: KernelBackend,
    dtype: torch.dtype,
    function: Callable,
) -> None:
    skip_if_incompatible_kernel_backend(kernel_backend)
    device = kernel_backend.get_compatible_accelerator().get_current_device()

    set_seed(_SEED)
    std = 0.02

    L = 7
    M = 417

    A = (
        torch.randn(
            (L, size[0], M) if is_A_transposed else (L, M, size[0]),
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        * std
    )
    B = (
        torch.randn(
            (L, size[1], size[0]) if is_B_transposed else (L, *size),
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        * std
    )
    C = torch.randn(L, M, size[1], device=device, dtype=dtype, requires_grad=False) * std if has_C else None

    alpha = 0.3
    beta = 0.7 if has_C else 0

    output_kernel = function(
        A=A,
        B=B,
        C=C,
        is_A_transposed=is_A_transposed,
        is_B_transposed=is_B_transposed,
        alpha=alpha,
        beta=beta,
        kernel_backend=KernelBackend.triton,
    )

    output_expected = bmm(
        A=A,
        B=B,
        C=C,
        alpha=alpha,
        beta=beta,
        is_A_transposed=is_A_transposed,
        is_B_transposed=is_B_transposed,
        kernel_backend=KernelBackend.torch,
    )

    assert_equal_tensors(
        output_kernel,
        output_expected,
        False,
        atol_float32=7e-5,
        rtol_float32=1e-4,
        atol_float16=1e-4,
        rtol_float16=5e-3,
        atol_bfloat16=1e-3,
        rtol_bfloat16=7e-3,
    )
