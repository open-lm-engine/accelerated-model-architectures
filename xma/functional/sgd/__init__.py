# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from .triton_implementation import sgd_triton


@torch.no_grad()
def sgd(
    parameters: list[torch.Tensor], gradients: list[torch.Tensor], lr: float = 1e-3, maximize: bool = False
) -> None:
    for W, dW in zip(parameters, gradients):
        assert W.is_contiguous()
        dW = dW.contiguous()

        sgd_triton(W=W, dW=dW, lr=lr, maximize=maximize)
