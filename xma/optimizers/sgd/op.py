# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from torch.optim.sgd import _multi_tensor_sgd, _single_tensor_sgd

from ...accelerator import Accelerator, KernelBackend
from ...utils import is_triton_available


if is_triton_available():
    from .triton_implementation import _sgd_triton


@torch.no_grad()
def sgd(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    momentum_buffer_list: list[torch.Tensor],
    lr: float,
    maximize: bool,
    horizontal_fusion: bool,
    weight_decay: float,
    momentum: float,
    dampening: float,
    nesterov: bool,
    *,
    kernel_backend: KernelBackend | None = None,
) -> None:
    if kernel_backend is None:
        kernel_backend = Accelerator.get_kernel_backend()
    else:
        assert kernel_backend.verify_accelerator()

    if kernel_backend in [KernelBackend.cuda, KernelBackend.triton]:
        assert not horizontal_fusion

        for W, M in zip(params, momentum_buffer_list):
            assert W.is_contiguous()

            if M is not None:
                assert M.is_contiguous()

        is_first_step = False
        if momentum == 0:
            assert len(momentum_buffer_list) == 0
            momentum_buffer_list = None
        elif momentum_buffer_list[0] is None:
            assert all([m is None for m in momentum_buffer_list])
            is_first_step = True

            for i, p in enumerate(params):
                momentum_buffer_list[i] = torch.empty_like(p, dtype=torch.float32)

        if momentum_buffer_list is None:
            momentum_buffer_list = [None] * len(params)

        for W, dW, M in zip(params, grads, momentum_buffer_list):
            dW = dW.contiguous()

            _sgd_triton(
                W=W,
                dW=dW,
                M=M,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                dampening=dampening,
                nesterov=nesterov,
                maximize=maximize,
                is_first_step=is_first_step,
            )
    elif kernel_backend == KernelBackend.torch:
        (_multi_tensor_sgd if horizontal_fusion else _single_tensor_sgd)(
            params=params,
            grads=grads,
            momentum_buffer_list=momentum_buffer_list,
            grad_scale=None,
            found_inf=None,
            weight_decay=weight_decay,
            momentum=momentum,
            lr=lr,
            dampening=dampening,
            nesterov=nesterov,
            maximize=maximize,
            has_sparse_grad=False,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")
