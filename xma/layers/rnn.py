# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from ..accelerator import KernelBackend
from ..functional import rnn
from ..math import divide_if_divisible


class RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        state_head_dim: int,
        output_size: int,
        num_input_heads: int,
        num_weight_heads: int,
        add_bias: bool,
        gradient_clipping: float | None,
    ) -> RNN:
        super().__init__()

        self.num_input_heads = num_input_heads
        self.num_weight_heads = num_weight_heads
        self.num_heads = max(num_input_heads, num_weight_heads)

        divide_if_divisible(self.num_heads, self.num_input_heads)
        divide_if_divisible(self.num_heads, self.num_weight_heads)

        self.gradient_clipping = gradient_clipping

        self.state_head_dim = state_head_dim
        self.state_size = self.num_heads * self.state_head_dim

        self.input_projection = nn.Linear(input_size, self.num_input_heads * self.state_head_dim, bias=add_bias)
        self.state_weight = nn.Parameter(torch.empty(self.num_weight_heads, self.state_head_dim, self.state_head_dim))
        self.output_projection = nn.Linear(self.state_size, output_size, bias=add_bias)

        self.reset_parameters()

    def forward(
        self,
        input: torch.Tensor,
        input_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        *,
        kernel_backend: KernelBackend | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input = self.input_projection(input)
        input = input.view(*input.size()[:-1], self.num_input_heads, self.state_head_dim)

        if input_state is not None:
            input_state = input_state.view(-1, self.num_heads, self.state_head_dim)

        input, input_state = rnn(
            input=input,
            weight=self.state_weight,
            input_state=input_state,
            gradient_clipping=self.gradient_clipping,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=kernel_backend,
        )

        input = input.flatten(-2, -1)
        input_state = input_state.flatten(-2, -1)

        input = self.output_projection(input)

        return input, input_state

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight)

    def extra_repr(self) -> str:
        output = super().extra_repr()
        return f"{output}\nstate size = {self.state_size} elements"
