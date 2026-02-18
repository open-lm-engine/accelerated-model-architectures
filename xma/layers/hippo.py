# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from ..functional import hippo


class HiPPO(nn.Module):
    def __init__(self, state_head_dim: int, measure: str = "legS") -> HiPPO:
        super().__init__()

        self.state_head_dim = state_head_dim
        self.measure = measure

        assert self.measure in ["legT", "lagT", "legS"]

        self.A = nn.Parameter(torch.empty(self.state_head_dim, self.state_head_dim))
        self.B = nn.Parameter(torch.empty(self.state_head_dim))

        self.reset_parameters()

    def forward(
        self,
        input: torch.Tensor,
        input_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        return hippo(
            input=input, A=self.A, B=self.B, input_state=input_state, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )

    @torch.no_grad
    def reset_parameters(self) -> None:
        self.A.normal_()
        self.B.normal_()
