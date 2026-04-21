# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import pytest
import torch

from xma import KernelBackend, bmm, set_seed

from ..utils import assert_equal_tensors, get_2d_tensor_sizes, skip_if_incompatible_kernel_backend


_SEED = 42


@pytest.mark.parametrize("size", get_2d_tensor_sizes())
@pytest.mark.parametrize("is_A_transposed", [False, True])
@pytest.mark.parametrize("is_B_transposed", [False, True])
@pytest.mark.parametrize("has_C", [False, True])
@pytest.mark.parametrize("kernel_backend", [KernelBackend.triton])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("function", [bmm, torch.compile(bmm, fullgraph=True)])
@torch._dynamo.config.patch(recompile_limit=1024)
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

    L = 7
    M = 417

    A = torch.randint(
        -8,
        8,
        (L, size[0], M) if is_A_transposed else (L, M, size[0]),
        device=device,
        dtype=dtype,
        requires_grad=False,
    )

    B = torch.randint(
        -8,
        8,
        (L, size[1], size[0]) if is_B_transposed else (L, *size),
        device=device,
        dtype=dtype,
        requires_grad=False,
    )

    C = torch.randint(-8, 8, (L, M, size[1]), device=device, dtype=dtype, requires_grad=False) if has_C else None

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
        True,
    )
