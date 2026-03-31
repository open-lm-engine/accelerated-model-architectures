# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Callable

import torch
from torch.optim import SGD as _TorchSGD

from ..accelerator import KernelBackend
from ..functional import sgd


class SGD(_TorchSGD):
    @torch.no_grad()
    def step(self, closure: Callable | None = None, *, kernel_backend: KernelBackend | None = None) -> None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            parameters = []
            gradients = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                parameters.append(p)
                gradients.append(p.grad)

            sgd(
                parameters=parameters,
                gradients=gradients,
                lr=group["lr"],
                maximize=False,
                horizontal_fusion=group["foreach"],
                kernel_backend=kernel_backend,
            )

        return loss
