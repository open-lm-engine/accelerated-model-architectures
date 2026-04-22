# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import pytest
import torch

from xma import KernelBackend, set_seed, softmax

from ..utils import assert_equal_tensors, get_random_duplicated_tensors, skip_if_incompatible_kernel_backend
from .rmsnorm_test import _get_sizes


_SEED = 42


@pytest.mark.parametrize("size", _get_sizes())
@pytest.mark.parametrize("kernel_backend", [KernelBackend.triton])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("logits_multiplier", [None, 0.7])
@pytest.mark.parametrize("function", [softmax, torch.compile(softmax, fullgraph=True)])
@torch._dynamo.config.patch(recompile_limit=1024)
def test_softmax(
    size: tuple[int],
    kernel_backend: KernelBackend,
    dtype: torch.dtype,
    logits_multiplier: float | None,
    function: Callable,
) -> None:
    skip_if_incompatible_kernel_backend(kernel_backend)
    device = kernel_backend.get_compatible_accelerator().get_current_device()

    set_seed(_SEED)

    if isinstance(size, int):
        size = (size,)

    x_kernel, x_expected = get_random_duplicated_tensors(size, device=device, dtype=dtype)

    z_kernel = function(x_kernel, logits_multiplier, kernel_backend=kernel_backend)
    z_expected = softmax(x_expected, logits_multiplier, kernel_backend=KernelBackend.torch)

    assert_equal_tensors(z_kernel, z_expected, False)

    z_kernel.sum().backward()
    z_expected.sum().backward()

    assert_equal_tensors(x_kernel.grad, x_expected.grad, False, atol_bfloat16=9e-4, rtol_bfloat16=0)
