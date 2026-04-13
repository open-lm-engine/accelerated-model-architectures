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

            has_sparse_grad = self._init_group(
                group=group, params=parameters, grads=gradients, momentum_buffer_list=momentum_buffer
            )

            assert not has_sparse_grad

            sgd(
                parameters=parameters,
                gradients=gradients,
                momentum_buffer=momentum_buffer,
                lr=group["lr"],
                maximize=group["maximize"],
                horizontal_fusion=group["foreach"],
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                kernel_backend=kernel_backend,
            )

            if group["momentum"] != 0:
                for p, buf in zip(parameters, momentum_buffer):
                    self.state[p]["momentum_buffer"] = buf

        return loss
