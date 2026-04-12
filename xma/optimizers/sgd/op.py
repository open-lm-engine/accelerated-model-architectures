# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from torch.optim.sgd import _multi_tensor_sgd, _single_tensor_sgd

from ...accelerator import Accelerator, KernelBackend
from ...constants import LOG_WARP_SIZE
from ...utils import empty_like_contiguous, is_triton_available


if is_triton_available():
    import triton.language as tl

    from .triton_implementation import _multi_tensor_sgd_triton_kernel, _single_tensor_sgd_triton

    _TORCH_TO_TRITON_DTYPE = {torch.float32: tl.float32, torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}


@torch.no_grad()
def sgd(
    parameters: list[torch.Tensor],
    gradients: list[torch.Tensor],
    momentum_buffer: list[torch.Tensor],
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
        assert dampening == 0
        assert not nesterov

        if momentum == 0:
            assert len(momentum_buffer) == 0
            momentum_buffer = None
        elif momentum_buffer[0] is None:
            assert all([m is None for m in momentum_buffer])
            for i, p in enumerate(parameters):
                momentum_buffer[i] = empty_like_contiguous(p, dtype=torch.float32)

        if horizontal_fusion:
            device = parameters[0].device
            NUM_WARPS = 8

            _multi_tensor_sgd_triton_kernel[len(parameters),](
                W_ptr_ptr=torch.tensor([W.data_ptr() for W in parameters], dtype=torch.int64, device=device),
                dW_ptr_ptr=torch.tensor([dW.data_ptr() for dW in gradients], dtype=torch.int64, device=device),
                M_ptr_ptr=(
                    None
                    if momentum == 0
                    else torch.tensor([M.data_ptr() for M in momentum_buffer], dtype=torch.int64, device=device)
                ),
                N_ptr=torch.tensor([W.numel() for W in parameters], dtype=torch.int64, device=device),
                lr=lr,
                weight_decay=None if weight_decay == 0 else weight_decay,
                momentum=None if momentum == 0 else momentum,
                BLOCK_SIZE=(NUM_WARPS << LOG_WARP_SIZE) * (16 // parameters[0].dtype.itemsize),
                MAXIMIZE=maximize,
                DTYPE=_TORCH_TO_TRITON_DTYPE[parameters[0].dtype],
                num_warps=NUM_WARPS,
            )
        else:
            if len(momentum_buffer) == 0:
                momentum_buffer = [None] * len(parameters)

            for W, dW, M in zip(parameters, gradients, momentum_buffer):
                assert W.is_contiguous()
                dW = dW.contiguous()

                _single_tensor_sgd_triton(
                    W=W, dW=dW, M=M, lr=lr, weight_decay=weight_decay, momentum=momentum, maximize=maximize
                )
    elif kernel_backend == KernelBackend.torch:
        (_multi_tensor_sgd if horizontal_fusion else _single_tensor_sgd)(
            params=parameters,
            grads=gradients,
            momentum_buffer_list=momentum_buffer,
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
