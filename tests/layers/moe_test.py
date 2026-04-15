# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from itertools import product

import pytest
import torch

from xma import KernelBackend, MoE, set_seed

from ..utils import (
    assert_equal_tensors,
    collect_gradients_from_module_and_zero_grads,
    get_activation_function,
    skip_if_incompatible_kernel_backend,
)


_SEED = 42


def _generate_args() -> list:
    args = list(
        product(
            [torch.float32, torch.float16, torch.bfloat16],  # dtype
            [2, 4, 6, 8],  # num_experts
            [2, 4],  # num_experts_per_tok
            [2048],  # hidden_size
            [8192],  # intermediate_size
            [True, False],  # is_glu
            [KernelBackend.triton],  # kernel_backend
        )
    )

    args += list(
        product(
            [torch.float32, torch.float16, torch.bfloat16],  # dtype
            [128],  # num_experts
            [8],  # num_experts_per_tok
            [576],  # hidden_size
            [256],  # intermediate_size
            [True, False],  # is_glu
            [KernelBackend.triton],  # kernel_backend
        )
    )

    args += list(
        product(
            [torch.bfloat16],
            [2, 4, 6, 8],  # num_experts
            [2, 4],  # num_experts_per_tok
            [2048],  # hidden_size
            [8192],  # intermediate_size
            [True, False],  # is_glu
            [KernelBackend.cuda],  # kernel_backend
        )
    )

    args += list(
        product(
            [torch.bfloat16],
            [128],  # num_experts
            [8],  # num_experts_per_tok
            [576],  # hidden_size
            [256],  # intermediate_size
            [True, False],  # is_glu
            [KernelBackend.cuda],  # kernel_backend
        )
    )

    return args


@pytest.mark.parametrize(
    "dtype,num_experts,num_experts_per_tok,hidden_size,intermediate_size,is_glu,kernel_backend",
    _generate_args(),
)
def test_moe(
    dtype: torch.dtype,
    num_experts: int,
    num_experts_per_tok: int,
    hidden_size: int,
    intermediate_size: int,
    is_glu: bool,
    kernel_backend: KernelBackend,
) -> None:
    skip_if_incompatible_kernel_backend(kernel_backend)
    device = kernel_backend.get_compatible_accelerator().get_current_device()

    if kernel_backend == KernelBackend.cuda and torch.cuda.get_device_capability(
        kernel_backend.get_compatible_accelerator().get_current_device()
    ) < (
        10,
        0,
    ):
        pytest.skip("skipping Blackwell Grouped Gemm since Blackwell GPU is not found")

    set_seed(_SEED)

    if num_experts_per_tok > num_experts:
        pytest.skip(
            f"skipping test since number of experts per token ({num_experts_per_tok}) is more than number of experts ({num_experts})"
        )

    with torch.device(device):
        moe = MoE(
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_function=get_activation_function(is_glu=is_glu),
            is_glu=is_glu,
            add_bias=False,
            std=0.02,
        ).to(dtype=dtype)

    x_torch = torch.randn(7, hidden_size, device=device, dtype=dtype, requires_grad=True)
    x_kernel = x_torch.clone().detach().requires_grad_()

    y_torch = moe(x_torch, kernel_backend=KernelBackend.torch)[0]
    y_kernel = moe(x_kernel, kernel_backend=kernel_backend)[0]

    assert_equal_tensors(
        y_kernel,
        y_torch,
        False,
        atol_float32=6e-3,
        rtol_float32=0,
        atol_float16=4e-3,
        rtol_float16=0,
        atol_bfloat16=2.35e-2,
        rtol_bfloat16=0,
    )

    y_torch.sum().backward()
    weight_torch_grads = collect_gradients_from_module_and_zero_grads(moe)

    y_kernel.sum().backward()
    weight_kernel_grads = collect_gradients_from_module_and_zero_grads(moe)

    assert_equal_tensors(
        x_kernel.grad,
        x_torch.grad,
        False,
        atol_float32=6.5e-3,
        rtol_float32=0,
        atol_float16=4e-3,
        rtol_float16=0,
        atol_bfloat16=4e-2,
        rtol_bfloat16=0,
    )

    for weight_name in weight_torch_grads:
        if "gate" in weight_name:
            continue

        assert_equal_tensors(
            weight_kernel_grads[weight_name],
            weight_torch_grads[weight_name],
            False,
            atol_float32=3e-2,
            rtol_float32=0,
            atol_float16=4e-3,
            rtol_float16=0,
            atol_bfloat16=4e-2,
            rtol_bfloat16=0,
        )
