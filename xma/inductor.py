# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import inspect
from contextlib import contextmanager
from functools import partial
from typing import Callable, Generator

import torch
from torch._inductor.fx_passes.joint_graph import patterns
from torch._inductor.pattern_matcher import fwd_only, joint_fwd_bwd, register_replacement

from .enums import KernelBackend
from .functional import fused_residual_add_rmsnorm, rmsnorm


_ALL_TRACE_FUNCTIONS = [joint_fwd_bwd, fwd_only]
_ALL_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def init_inductor(cache_size_limit: int) -> None:
    torch._dynamo.config.cache_size_limit = cache_size_limit
    torch._dynamo.config.accumulated_cache_size_limit = cache_size_limit


def partialize_and_update_signature(func: Callable, **kwargs) -> Callable:
    original_sig = inspect.signature(func)
    parameters = original_sig.parameters

    new_parameters = {key: value for key, value in parameters.items() if key not in kwargs}
    new_signature = inspect.Signature(parameters=list(new_parameters.values()))

    partial_func = partial(func, **kwargs)

    def wrapper(*args, **kwargs):
        return partial_func(*args, **kwargs)

    wrapper.__signature__ = new_signature
    wrapper.__name__ = func.__name__

    return wrapper


_DIM_TO_SIZE = {1: (4,), 2: (4, 4), 3: (4, 4, 4), 4: (4, 4, 4, 4)}


def _get_example_input(dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(_DIM_TO_SIZE[dim], device=device, dtype=dtype, requires_grad=True)


def get_rmsnorm_replacer(
    device: torch.device,
) -> Generator[tuple[Callable, Callable, tuple[torch.Tensor, torch.Tensor]]]:
    for dtype in _ALL_DTYPES:
        example_inputs = (
            _get_example_input(2, device=device, dtype=dtype),
            _get_example_input(1, device=device, dtype=dtype),
        )

        search_function = partialize_and_update_signature(
            rmsnorm, eps=None, memory_efficient=False, kernel_backend=KernelBackend.torch
        )

        replacement_function = partialize_and_update_signature(
            rmsnorm, eps=None, memory_efficient=False, kernel_backend=KernelBackend.triton
        )

        yield search_function, replacement_function, example_inputs


def get_fused_residual_add_rmsnorm_replacer(
    device: torch.device,
) -> Generator[tuple[Callable, Callable, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    for dtype in _ALL_DTYPES:
        for dim in range(1, 5):
            example_inputs = (
                torch.empty((1,) * dim, device=device, dtype=dtype, requires_grad=True),
                torch.empty((1,) * dim, device=device, dtype=dtype, requires_grad=True),
                torch.empty(1, device=device, dtype=dtype, requires_grad=True),
            )

            search_function = partialize_and_update_signature(
                fused_residual_add_rmsnorm,
                eps=None,
                multiplier=None,
                memory_efficient=False,
                kernel_backend=KernelBackend.torch,
            )

            replacement_function = partialize_and_update_signature(
                fused_residual_add_rmsnorm,
                eps=None,
                multiplier=None,
                memory_efficient=False,
                kernel_backend=KernelBackend.triton,
            )

            yield search_function, replacement_function, example_inputs


_MAPPING = {
    rmsnorm.__name__: get_rmsnorm_replacer,
    fused_residual_add_rmsnorm.__name__: get_fused_residual_add_rmsnorm_replacer,
}


# @contextmanager
def enable_kernels(kernels: list[str]):
    device = torch.cuda.current_device()

    for kernel in kernels:
        for search_function, replacement_function, example_inputs in _MAPPING[kernel](device):
            for trace_function in _ALL_TRACE_FUNCTIONS:
                print("hi")
                register_replacement(
                    search_fn=search_function,
                    replace_fn=replacement_function,
                    example_inputs=example_inputs,
                    trace_fn=trace_function,
                    pass_dicts=patterns,
                )
