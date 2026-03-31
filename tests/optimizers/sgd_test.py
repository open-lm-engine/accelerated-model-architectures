# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Callable

import pytest
import torch

from xma import KernelBackend
from xma.functional import sgd

from ..utils import assert_equal_tensors, get_1d_tensor_sizes, skip_if_incompatible_kernel_backend


@pytest.mark.parametrize("size", get_1d_tensor_sizes())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("horizontal_fusion", [True, False])
@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize("kernel_backend", [KernelBackend.triton])
@pytest.mark.parametrize("function", [sgd, torch.compile(sgd, fullgraph=True)])
def test_sgd(
    size: int,
    dtype: torch.dtype,
    horizontal_fusion: bool,
    maximize: bool,
    kernel_backend: KernelBackend,
    function: Callable,
) -> None:
    skip_if_incompatible_kernel_backend(kernel_backend)
    device = kernel_backend.get_compatible_accelerator().get_current_device()

    params_triton = [torch.randn(size, device=device, dtype=dtype) for _ in range(3)]
    grads = [torch.randn(size, device=device, dtype=dtype) for _ in range(3)]
    params_torch = [p.clone() for p in params_triton]

    function(
        parameters=params_triton,
        gradients=grads,
        lr=1e-3,
        maximize=maximize,
        horizontal_fusion=horizontal_fusion,
        kernel_backend=kernel_backend,
    )

    sgd(
        parameters=params_torch,
        gradients=[g.clone() for g in grads],
        lr=1e-3,
        maximize=maximize,
        horizontal_fusion=horizontal_fusion,
        kernel_backend=KernelBackend.torch,
    )

    for p_triton, p_torch in zip(params_triton, params_torch):
        assert_equal_tensors(
            p_triton,
            p_torch,
            exact_match=False,
            atol_float32=0,
            rtol_float32=0,
            atol_float16=0,
            rtol_float16=0,
            atol_bfloat16=0,
            rtol_bfloat16=0,
        )
