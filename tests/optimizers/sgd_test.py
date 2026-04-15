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
@pytest.mark.parametrize("horizontal_fusion", [False])
@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("weight_decay", [0, 0.7])
@pytest.mark.parametrize("momentum", [0, 0.7])
@pytest.mark.parametrize("dampening", [0, 0.7])
@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("kernel_backend", [KernelBackend.triton])
def test_sgd(
    size: int,
    dtype: torch.dtype,
    horizontal_fusion: bool,
    maximize: bool,
    weight_decay: float,
    momentum: float,
    dampening: float,
    nesterov: bool,
    kernel_backend: KernelBackend,
) -> None:
    skip_if_incompatible_kernel_backend(kernel_backend)
    device = kernel_backend.get_compatible_accelerator().get_current_device()

    if nesterov and (dampening != 0 or momentum == 0):
        pytest.skip(f"invalid config")

    params_kernel = [torch.randint(-8, 8, (size,), device=device, dtype=dtype) for _ in range(3)]
    params_torch = [p.clone() for p in params_kernel]

    grads = [torch.randint(-8, 8, (size,), device=device, dtype=dtype) for _ in range(3)]

    for pk, pt, g in zip(params_kernel, params_torch, grads):
        pk.grad = g
        pt.grad = g

    sgd_kernel = SGD(
        params=params_kernel,
        lr=_LEARNING_RATE,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        maximize=maximize,
        nesterov=nesterov,
        foreach=horizontal_fusion,
    )

    sgd_torch = SGD(
        params=params_torch,
        lr=_LEARNING_RATE,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        maximize=maximize,
        nesterov=nesterov,
        foreach=horizontal_fusion,
    )

    sgd_kernel.step(kernel_backend=kernel_backend)
    sgd_torch.step(kernel_backend=KernelBackend.torch)

    for param_kernel, param_torch in zip(params_kernel, params_torch):
        assert_equal_tensors(param_kernel, param_torch, exact_match=False)

        m_kernel = sgd_kernel.state[param_kernel].get("momentum_buffer")
        m_torch = sgd_torch.state[param_torch].get("momentum_buffer")

        if momentum == 0:
            assert m_kernel is None
            assert m_torch is None
