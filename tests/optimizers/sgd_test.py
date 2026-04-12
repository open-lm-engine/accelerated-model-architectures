# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import pytest
import torch

from xma import SGD, KernelBackend

from ..utils import assert_equal_tensors, get_1d_tensor_sizes, skip_if_incompatible_kernel_backend


_LEARNING_RATE = 1e-3


@pytest.mark.parametrize("size", get_1d_tensor_sizes())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("horizontal_fusion", [True, False])
@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("weight_decay", [2, 0])
@pytest.mark.parametrize("kernel_backend", [KernelBackend.triton])
def test_sgd(
    size: int,
    dtype: torch.dtype,
    horizontal_fusion: bool,
    maximize: bool,
    weight_decay: float,
    kernel_backend: KernelBackend,
) -> None:
    skip_if_incompatible_kernel_backend(kernel_backend)
    device = kernel_backend.get_compatible_accelerator().get_current_device()

    params_kernel = [torch.randn(size, device=device, dtype=dtype) for _ in range(3)]
    params_torch = [p.clone() for p in params_kernel]

    grads = [torch.randn(size, device=device, dtype=dtype) for _ in range(3)]

    for pk, pt, g in zip(params_kernel, params_torch, grads):
        pk.grad = g
        pt.grad = g

    sgd_kernel = SGD(
        params=params_kernel,
        lr=_LEARNING_RATE,
        weight_decay=weight_decay,
        maximize=maximize,
        foreach=horizontal_fusion,
    )

    sgd_torch = SGD(
        params=params_torch, lr=_LEARNING_RATE, weight_decay=weight_decay, maximize=maximize, foreach=horizontal_fusion
    )

    sgd_kernel.step(kernel_backend=kernel_backend)
    sgd_torch.step(kernel_backend=KernelBackend.torch)

    for p_triton, p_torch in zip(params_kernel, params_torch):
        assert_equal_tensors(p_triton, p_torch, exact_match=True)
