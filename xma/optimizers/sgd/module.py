# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from torch.optim import SGD, Optimizer
from torch.optim.optimizer import ParamsT

from ...accelerator import KernelBackend
from .op import sgd


class SGD(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float | torch.Tensor = 0,
        nesterov: bool = False,
        *,
        maximize: bool = False,
    ) -> SGD:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, *, kernel_backend: KernelBackend | None = None) -> None:
        for group in self.param_groups:
            params, grads, momentum_buffer_list = self._init_group(
                group=group, params=params, grads=grads, momentum_buffer_list=momentum_buffer_list
            )

            sgd(
                params=params,
                grads=grads,
                momentum_buffer_list=momentum_buffer_list,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                kernel_backend=kernel_backend,
            )

            if group["momentum"] != 0:
                for p, m in zip(params, momentum_buffer_list, strict=True):
                    self.state[p]["momentum_buffer"] = m

    def _init_group(self, group, params, grads, momentum_buffer_list):
        params = []
        grads = []
        momentum_buffer_list = []

        for p in group["params"]:
            if p.grad is None:
                continue

            params.append(p)
            grads.append(p.grad)

            if group["momentum"] != 0:
                state = self.state[p]
                momentum_buffer_list.append(state.get("momentum_buffer"))
