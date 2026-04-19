# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from torch.distributed.tensor import DTensor
from torch.optim.sgd import _multi_tensor_sgd, _single_tensor_sgd

from ...accelerator import Accelerator, KernelBackend
from ...utils import is_triton_available


if is_triton_available():
    import triton.language as tl

    from .triton_implementation import _single_tensor_sgd_triton


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

        is_first_step = False
        if momentum == 0:
            assert len(momentum_buffer_list) == 0
            momentum_buffer_list = [None] * len(params)
        elif momentum_buffer_list[0] is None:
            assert all([m is None for m in momentum_buffer_list])
            is_first_step = True

            for i, p in enumerate(params):
                momentum_buffer_list[i] = torch.empty_like(p, dtype=torch.float32)

        is_dtensor = isinstance(params[0], DTensor)

        if is_dtensor:
            for W, dW, M in zip(params, grads, momentum_buffer_list):
                assert isinstance(dW, DTensor)
                assert W.placements == dW.placements

                if M is not None:
                    assert isinstance(M, DTensor)
                    assert W.placements == M.placements

        for W, dW, M in zip(params, grads, momentum_buffer_list):
            assert W.is_contiguous()
            dW = dW.contiguous()

            if M is not None:
                assert M.is_contiguous()

            if is_dtensor:
                W = W.to_local()
                dW = dW.to_local()

                if M is not None:
                    M = M.to_local()

            _single_tensor_sgd_triton(
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
