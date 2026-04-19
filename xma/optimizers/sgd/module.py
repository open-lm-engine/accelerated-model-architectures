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
    def step(
        self,
        closure: Callable | None = None,
        *,
        chunk_size: int | None = None,
        kernel_backend: KernelBackend | None = None,
    ) -> None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = []
            grads = []
            momentum_buffer_list = []

            has_sparse_grad = self._init_group(
                group=group, params=params, grads=grads, momentum_buffer_list=momentum_buffer_list
            )

            assert not has_sparse_grad

            sgd(
                params=params,
                grads=grads,
                momentum_buffer_list=momentum_buffer_list,
                lr=group["lr"],
                maximize=group["maximize"],
                horizontal_fusion=group["foreach"],
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                chunk_size=chunk_size,
                kernel_backend=kernel_backend,
            )

            if group["momentum"] != 0:
                for p, m in zip(params, momentum_buffer_list, strict=True):
                    self.state[p]["momentum_buffer"] = m

        return loss
