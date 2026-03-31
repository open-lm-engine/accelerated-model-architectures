# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...accelerator import Accelerator, KernelBackend
from .triton_implementation import sgd_horizontally_fused_triton, sgd_triton


@torch.no_grad()
def sgd(
    parameters: list[torch.Tensor],
    gradients: list[torch.Tensor],
    lr: float = 1e-3,
    maximize: bool = False,
    horizontal_fusion: bool = True,
    *,
    kernel_backend: KernelBackend | None = None,
) -> None:
    if kernel_backend is None:
        kernel_backend = Accelerator.get_kernel_backend()
    else:
        assert kernel_backend.verify_accelerator()

    if horizontal_fusion:
        sgd_horizontally_fused_triton(Ws=parameters, dWs=gradients, lr=lr, maximize=maximize)
    else:
        for W, dW in zip(parameters, gradients):
            assert W.is_contiguous()
            dW = dW.contiguous()

            sgd_triton(W=W, dW=dW, lr=lr, maximize=maximize)
