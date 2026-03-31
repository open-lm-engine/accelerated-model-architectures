# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Callable

import torch
from torch.optim import SGD as _TorchSGD

from ..functional import sgd


class SGD(_TorchSGD):
    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for param_group in self.param_groups:
            sgd(parameters=self.param_groups["params"], lr=param_group["lr"], maximize=False)

        return loss
