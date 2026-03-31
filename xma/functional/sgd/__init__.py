# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from torch.optim.sgd import _multi_tensor_sgd, _single_tensor_sgd

from ...accelerator import Accelerator, KernelBackend
from .triton_implementation import sgd_horizontally_fused_triton, sgd_triton


@torch.no_grad()
def sgd(
    parameters: list[torch.Tensor],
    gradients: list[torch.Tensor],
    lr: float,
    maximize: bool,
    horizontal_fusion: bool,
    *,
    kernel_backend: KernelBackend | None = None,
) -> None:
    if kernel_backend is None:
        kernel_backend = Accelerator.get_kernel_backend()
    else:
        assert kernel_backend.verify_accelerator()

    if kernel_backend in [KernelBackend.cuda, KernelBackend.triton]:
        if horizontal_fusion:
            sgd_horizontally_fused_triton(Ws=parameters, dWs=gradients, lr=lr, maximize=maximize)
        else:
            for W, dW in zip(parameters, gradients):
                assert W.is_contiguous()
                dW = dW.contiguous()

                sgd_triton(W=W, dW=dW, lr=lr, maximize=maximize)
    elif kernel_backend == KernelBackend.torch:
        (_multi_tensor_sgd if horizontal_fusion else _single_tensor_sgd)(
            params=parameters,
            grads=gradients,
            momentum_buffer_list=[None] * len(parameters),
            grad_scale=None,
            found_inf=None,
            weight_decay=0,
            momentum=0,
            lr=lr,
            dampening=0,
            nesterov=False,
            maximize=maximize,
            has_sparse_grad=False,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")
