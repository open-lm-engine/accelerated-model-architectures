# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from ..accelerator import KernelBackend
from ..functional import causal_convolution
from ..math import divide_if_divisible


class CausalConvolution(nn.Conv1d):
    def __init__(
        self,
        input_size: int | None,
        output_size: int | None,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
        stride: int = 1,
        activation_function: str = "silu",
        add_bias: bool = True,
    ) -> CausalConvolution:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.stride = stride
        self.activation_string = activation_function

        self.input_projection = (
            nn.Identity() if input_size is None else nn.Linear(input_size, in_channels, bias=add_bias)
        )

        self.weight = nn.Parameter(torch.empty(self.out_channels, self.in_channels // self.groups, self.kernel_size))

        self.bias = None
        if add_bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels))

        self.output_projection = (
            nn.Identity() if output_size is None else nn.Linear(self.out_channels, output_size, bias=add_bias)
        )

    def forward(
        self,
        x: torch.Tensor,
        input_state: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        *,
        kernel_backend: KernelBackend | None = None,
    ) -> torch.Tensor:
        x = self.input_projection(x)

        x, input_state = causal_convolution(
            x=x,
            input_state=input_state,
            attention_mask=attention_mask,
            return_cache_state=cache_params is not None,
            weight=self.weight,
            bias=self.bias,
            groups=self.groups,
            stride=self.stride,
            activation_string=self.activation_string,
            kernel_backend=kernel_backend,
        )

        x = self.output_projection(x)

        return x, input_state
