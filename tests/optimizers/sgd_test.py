# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from xma import SGD, Accelerator


class Model(nn.Module):
    def __init__(self) -> Model:
        super().__init__()
        self.linear1 = nn.Linear(3, 17)
        self.linear2 = nn.Linear(17, 61)
        self.linear3 = nn.Linear(61, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


def test_sgd() -> None:
    model = Model()
    optimizer = SGD(params=model.parameters())

    x = torch.randn(5, 3, device=Accelerator.get_current_device())
    y = model(x)

    torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.randn_like(y))

    optimizer.step()
