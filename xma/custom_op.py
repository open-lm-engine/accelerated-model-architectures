# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import inspect
from typing import Any, Callable, Iterable, Sequence

import torch

from .accelerator import Accelerator, KernelBackend
from .constants import LIBRARY_NAME
from .counters import increment_counter


def ctx_needs_gradients(ctx) -> bool:
    return any(ctx.needs_input_grad)


def ctx_save_for_backward(ctx, *args) -> None:
    if ctx_needs_gradients(ctx):
        ctx.save_for_backward(*args)


class CustomOp(torch.autograd.Function):
    @classmethod
    def run(cls, kernel_backend: KernelBackend | None = None, **kwargs) -> Any:
        if kernel_backend is None:
            kernel_backend = Accelerator.get_kernel_backend()
        else:
            assert kernel_backend.verify_accelerator()

        if kernel_backend is None:
            raise ValueError("code is not supposed to reach here! kernel_backend was not inferrable")

        increment_counter(cls._get_key(kernel_backend))

        output = (
            cls.forward_backward_torch(**kwargs)
            if kernel_backend == KernelBackend.torch
            else cls.apply(*tuple(kwargs.values()), kernel_backend)
        )

        return output

    @staticmethod
    def forward(ctx, *args, kernel_backend: KernelBackend) -> Any:
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *grad_outputs) -> Any:
        raise NotImplementedError

    @staticmethod
    def forward_backward_torch(*args, **kwargs) -> Any:
        raise NotImplementedError

    @classmethod
    def _get_key(cls, kernel_backend: KernelBackend) -> str:
        return f"{cls.__name__}-{kernel_backend.value}"


def xma_op(
    mutates_args: str | Iterable[str] = None,
    device_types: str | Sequence[str] | None = None,
    schema: str | None = None,
    fake_func: Callable | None = None,
) -> Callable:
    def _inner(func: Callable):
        custom_op = torch.library.custom_op(
            f"{LIBRARY_NAME}::{func.__name__}",
            func,
            mutates_args=mutates_args,
            device_types=device_types,
            schema=schema,
        )

        if fake_func is not None:
            custom_op.register_fake(fake_func)

        def _run(*args, **kwargs):
            return custom_op(*args, **kwargs)

        _run.__signature__ = inspect.signature(func)
        _run.__name__ = func.__name__

        return _run

    return _inner
