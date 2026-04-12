# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Callable

import torch
from torch.optim import SGD as _TorchSGD

from ...accelerator import KernelBackend
from .op import sgd


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
            momentum_buffer = []
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                parameters.append(p)
                gradients.append(p.grad)

                if momentum != 0:
                    momentum_buffer.append(self.state[p].get("momentum_buffer"))

            sgd(
                parameters=parameters,
                gradients=gradients,
                momentum_buffer=momentum_buffer,
                lr=group["lr"],
                maximize=False,
                horizontal_fusion=group["foreach"],
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                kernel_backend=kernel_backend,
            )

        return loss
