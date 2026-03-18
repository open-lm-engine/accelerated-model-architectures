# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from ..functional import causal_convolution


class CausalConvolution(nn.Conv1d):
    def forward(
        self, x: torch.Tensor, input_state: torch.Tensor | None, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return causal_convolution(
            x=x,
            input_state=input_state,
            attention_mask=attention_mask,
            return_cache_state=cache_params is not None,
            weight=self.conv1d.weight,
            bias=self.conv1d.bias,
            num_groups=self.num_groups,
            stride=self.stride,
            activation_string=self.activation_string,
            conv1d_padding=self.kernel_size - 1,
            conv1d_stride=1,
        )
