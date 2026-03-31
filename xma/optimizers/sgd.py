from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer

from ..functional import sgd


class SGD(Optimizer):
    def __init__(self, params: list[nn.Parameter], defaults) -> SGD:
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for param_group in self.param_groups:
            sgd(parameters=self.param_groups["params"], lr=param_group["lr"], maximize=False)

        return loss
