# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import random

import pytest

from xma import KernelBackend
from xma.utils import is_jax_available, is_torch_available


if is_torch_available():
    import torch
    import torch.nn as nn
    from torch.testing import assert_close


# markers for tests that only need a lazy, function-local `import torch`/`import jax` (no module-level use of
# torch/jax objects, e.g. in `@pytest.mark.parametrize(...)` argument lists computed at collection time). Modules
# that build parametrize arguments out of `torch.float32` etc. at module level can't use these markers, since the
# module-level import already crashes before the marker gets a chance to run — use
# `torch = pytest.importorskip("torch")` at the top of the file instead.
torch_test = pytest.mark.skipif(not is_torch_available(), reason="torch not available")
jax_test = pytest.mark.skipif(not is_jax_available(), reason="jax not available")


def skip_if_incompatible_kernel_backend(kernel_backend: KernelBackend) -> None:
    if not kernel_backend.verify_accelerator():
        pytest.skip(f"device incompatible with kernel_backend ({kernel_backend})")


def get_1d_tensor_sizes(log_max_power_of_2: int = 15, max_offset: int = 10, num_not_powers_of_2: int = 50) -> set[int]:
    rng = random.Random(0)
    sizes = set()
    # powers of 2
    for i in range(log_max_power_of_2):
        start = 2**i
        for j in range(max_offset):
            sizes.add(start + j)
    # not powers of 2
    for _ in range(num_not_powers_of_2):
        sizes.add(3000 + rng.randint(-1000, 1000))
    return list(sizes)


def get_2d_tensor_sizes(log_max_power_of_2: int = 15, max_offset: int = 10, num_not_powers_of_2: int = 50) -> set[int]:
    rng = random.Random(0)
    sizes = set()
    # powers of 2
    for i in range(log_max_power_of_2):
        start = 2**i
        for j in range(max_offset):
            sizes.add((start + j, start + j))
    # not powers of 2
    for _ in range(num_not_powers_of_2):
        sizes.add((3000 + rng.randint(-1000, 1000), 3000 + rng.randint(-1000, 1000)))
    return list(sizes)


def get_random_duplicated_tensors(
    size: tuple[int], device: torch.device, dtype: torch.dtype, std: float | None = None
) -> tuple[torch.Tensor]:
    if isinstance(size, int):
        size = (size,)

    if std is None:
        x = torch.randint(-8, 8, size, device=device, dtype=dtype, requires_grad=False)
    else:
        x = torch.randn(size, device=device, dtype=dtype, requires_grad=False) * std

    x.requires_grad_()
    x_clone = x.clone().detach().requires_grad_()

    return x, x_clone


def assert_equal_tensors(
    x: torch.Tensor,
    y: torch.Tensor,
    exact_match: bool,
    rtol_float32: float = None,
    atol_float32: float = None,
    rtol_float16: float = None,
    atol_float16: float = None,
    rtol_bfloat16: float = None,
    atol_bfloat16: float = None,
) -> None:
    if exact_match:
        assert x.equal(y)
    else:
        assert x.dtype == y.dtype
        dtype = x.dtype

        if dtype == torch.float32:
            assert_close(x, y, rtol=rtol_float32, atol=atol_float32)
        elif dtype == torch.float16:
            assert_close(x, y, rtol=rtol_float16, atol=atol_float16)
        elif dtype == torch.bfloat16:
            assert_close(x, y, rtol=rtol_bfloat16, atol=atol_bfloat16)
        else:
            raise ValueError(f"unexpected dtype ({dtype})")


def collect_gradients_from_module_and_zero_grads(model: nn.Module) -> dict[str, torch.Tensor]:
    grads = {}
    for weight_name, weight in model.named_parameters():
        grads[weight_name] = weight.grad

    model.zero_grad()

    return grads


def get_activation_function(is_glu: bool) -> nn.Module:
    return nn.GLU() if is_glu else nn.GELU(approximate="tanh")
