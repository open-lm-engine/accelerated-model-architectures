# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn


class HiPPO(nn.Module):
    def __init__(self, state_head_dim: int, measure: str = "legS") -> HiPPO:
        self.state_head_dim = state_head_dim
        self.measure = measure

        assert self.measure in ["legT", "lagT", "legS"]

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def reset_parameters(self) -> None:
        self.A
        self.B
