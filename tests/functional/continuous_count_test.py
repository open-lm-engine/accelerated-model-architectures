# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Callable

import pytest
import torch

from xma import KernelBackend, continuous_count, set_seed

from ..utils import assert_equal_tensors, get_1d_tensor_sizes, skip_if_incompatible_kernel_backend


_MAX_EXPERTS = 72
_SEED = 42


@pytest.mark.parametrize("bins", get_1d_tensor_sizes())
@pytest.mark.parametrize("kernel_backend", [KernelBackend.cuda])
@pytest.mark.parametrize("dtype", [torch.long, torch.int])
def test_continuous_count(bins: int, kernel_backend: KernelBackend, dtype: torch.dtype) -> None:
    skip_if_incompatible_kernel_backend(kernel_backend)
    device = kernel_backend.get_compatible_accelerator().get_current_device()

    set_seed(_SEED)
    x = torch.randint(0, _MAX_EXPERTS, (bins,), device=device, dtype=dtype)

    z_kernel = continuous_count(x=x, bins=_MAX_EXPERTS, kernel_backend=KernelBackend.cuda)
    z_expected = continuous_count(x.view(-1), bins=_MAX_EXPERTS, kernel_backend=KernelBackend.torch)

    assert_equal_tensors(z_kernel, z_expected, True)
