# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import pytest
import torch

from xma import KernelBackend, fused_residual_add_rmsnorm, set_seed

from ..utils import (
    assert_equal_tensors,
    get_1d_tensor_sizes,
    get_random_duplicated_tensors,
    skip_if_incompatible_kernel_backend,
)


_EPSILON = 1e-5
_SEED = 42


def _get_sizes() -> list[tuple]:
    sizes = []
    for size in get_1d_tensor_sizes(max_offset=5):
        sizes.append((400, size))

    return sizes


@pytest.mark.parametrize("size", _get_sizes())
@pytest.mark.parametrize("kernel_backend", [KernelBackend.triton])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("memory_efficient", [False, True])
@pytest.mark.parametrize("has_weight", [False, True])
@pytest.mark.parametrize("multiplier", [None, 0.9])
def test_fused_residual_add_rmsnorm(
    size: tuple[int],
    kernel_backend: KernelBackend,
    dtype: torch.dtype,
    memory_efficient: bool,
    has_weight: bool,
    multiplier: float | None,
) -> None:
    skip_if_incompatible_kernel_backend(kernel_backend)
    device = kernel_backend.get_compatible_accelerator().get_current_device()

    set_seed(_SEED)

    x_kernel, x_expected = get_random_duplicated_tensors(size, device=device, dtype=dtype)
    residual_kernel, residual_expected = get_random_duplicated_tensors(size, device=device, dtype=dtype)

    if has_weight:
        weight_kernel, weight_expected = get_random_duplicated_tensors(size[-1], device=device, dtype=dtype)
    else:
        weight_kernel = None
        weight_expected = None

    z_kernel, r_kernel = fused_residual_add_rmsnorm(
        x=x_kernel,
        residual=residual_kernel,
        weight=weight_kernel,
        eps=_EPSILON,
        multiplier=multiplier,
        memory_efficient=memory_efficient,
        kernel_backend=KernelBackend.triton,
    )
    z_kernel = z_kernel * 2 + r_kernel * 3

    z_expected, r_expected = fused_residual_add_rmsnorm(
        x=x_expected,
        residual=residual_expected,
        weight=weight_expected,
        eps=_EPSILON,
        multiplier=multiplier,
        kernel_backend=KernelBackend.torch,
    )
    z_expected = z_expected * 2 + r_expected * 3

    assert_equal_tensors(z_kernel, z_expected, False, atol_float32=1.4e-4, rtol_float32=0)

    z_kernel.sum().backward()
    z_expected.sum().backward()

    assert_equal_tensors(x_kernel.grad, x_expected.grad, False, atol_float32=1.5e-4, rtol_float32=0)
    assert_equal_tensors(residual_kernel.grad, residual_expected.grad, False, atol_float32=1.6e-4, rtol_float32=0)

    if has_weight:
        assert_equal_tensors(
            weight_kernel.grad,
            weight_expected.grad,
            False,
            atol_float32=1.1e-4,
            rtol_float32=0,
        )


# def test_fused_residual_add_rmsnorm_kernel_replacement() -> None:
#     class Model(nn.Module):
#         def __init__(self, shape: int) -> Model:
#             super().__init__()
#             self.norm = nn.RMSNorm(shape)
#             self.l1 = nn.Linear(shape, shape)
#             self.l2 = nn.Linear(shape, shape)

#         def forward(self, x: torch.Tensor) -> torch.Tensor:
#             x = self.l1(x)
#             r = x
#             x = x + r
#             x = self.norm(x)
#             x = self.l2(x)
#             return x

#     size = (4, 7)

#     device = torch.cuda.current_device()
#     dtype = torch.float32

#     with torch.device(device):
#         model = Model(size[-1])

#     x = torch.randn(size, device=device, dtype=dtype, requires_grad=True)

#     reset_counters()

#     enable_kernels([fused_residual_add_rmsnorm.__name__])
#     model = torch.compile(model)

#     with enable_counters():
#         model(x)

#     assert get_counter_value(fused_residual_add_rmsnorm) == 2
