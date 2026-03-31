# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import pytest
import torch

from xma import KernelBackend, cross_entropy, set_seed

from ..utils import (
    assert_equal_tensors,
    get_2d_tensor_sizes,
    get_random_duplicated_tensors,
    skip_if_incompatible_kernel_backend,
)


_SEED = 42


@pytest.mark.parametrize("size", get_2d_tensor_sizes())
@pytest.mark.parametrize("kernel_backend", [KernelBackend.triton])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("logits_multiplier", [None, 0.7])
@pytest.mark.parametrize("function", [cross_entropy, torch.compile(cross_entropy, fullgraph=True)])
@torch._dynamo.config.patch(recompile_limit=1024)
def test_cross_entropy(
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

    x_kernel, x_expected = get_random_duplicated_tensors(size, device=device, dtype=dtype, std=0.02)
    labels = torch.randint(0, x_kernel.size(-1), (x_kernel.size(0),), device=x_kernel.device)

    loss_kernel = function(
        x=x_kernel, labels=labels, logits_multiplier=logits_multiplier, kernel_backend=KernelBackend.triton
    )

    loss_expected = cross_entropy(
        x=x_expected, labels=labels, logits_multiplier=logits_multiplier, kernel_backend=KernelBackend.torch
    )

    assert_equal_tensors(loss_kernel, loss_expected, False, atol_float32=6.3e-5, rtol_float32=0)

    loss_kernel.backward()
    loss_expected.backward()

    assert_equal_tensors(x_kernel.grad, x_expected.grad, False)
