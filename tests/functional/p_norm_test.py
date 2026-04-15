# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import pytest
import torch

from xma import KernelBackend, p_norm, set_seed

from ..utils import assert_equal_tensors, get_random_duplicated_tensors, skip_if_incompatible_kernel_backend
from .fused_residual_add_rmsnorm_test import _get_sizes


_SEED = 42


@pytest.mark.parametrize("size", _get_sizes())
@pytest.mark.parametrize("kernel_backend", [KernelBackend.triton])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("p", [1, 2, 3, "inf"])
@pytest.mark.parametrize("multiplier", [None, 0.9])
@torch._dynamo.config.patch(recompile_limit=1024)
def test_p_norm(
    size: tuple[int],
    kernel_backend: KernelBackend,
    dtype: torch.dtype,
    p: int | str,
    multiplier: float | None,
) -> None:
    skip_if_incompatible_kernel_backend(kernel_backend)
    device = kernel_backend.get_compatible_accelerator().get_current_device()

    set_seed(_SEED)

    x_kernel, x_expected = get_random_duplicated_tensors(size, device=device, dtype=dtype)

    z_kernel = p_norm(x=x_kernel, multiplier=multiplier, p=p, kernel_backend=kernel_backend)
    z_expected = p_norm(x=x_expected, multiplier=multiplier, p=p, kernel_backend=KernelBackend.torch)

    assert_equal_tensors(
        z_kernel, z_expected, False, atol_float32=3.1e-3 if p == 3 else None, rtol_float32=0 if p == 3 else None
    )
