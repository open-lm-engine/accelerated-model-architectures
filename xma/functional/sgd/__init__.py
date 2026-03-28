# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from .triton_implementation import sgd_horizontally_fused_triton, sgd_triton


@torch.no_grad()
def sgd(
    parameters: list[torch.Tensor],
    gradients: list[torch.Tensor],
    lr: float = 1e-3,
    maximize: bool = False,
    horizontal_fusion: bool = True,
) -> None:
    if horizontal_fusion:
        sgd_horizontally_fused_triton(Ws=parameters, dWs=gradients, lr=lr, maximize=maximize)
    else:
        for W, dW in zip(parameters, gradients):
            assert W.is_contiguous()
            dW = dW.contiguous()

            sgd_triton(W=W, dW=dW, lr=lr, maximize=maximize)
