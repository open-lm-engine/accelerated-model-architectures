# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import divide_if_divisible, is_causal_conv1d_available
from ..activations import get_activation_function, is_glu
from ..convolution import ParameterizedConv1d
from ..linear import ParameterizedLinear
from ..mlp_blocks.mlp import _get_std_for_linear


if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update


def _apply_mask_to_padding_states(x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = x.dtype
        x = (x * attention_mask[:, :, None]).to(dtype)

    return x


def causal_convolution(
    x: torch.Tensor,
    input_state: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor | None,
    conv1d_num_groups: int,
    return_cache_state: bool,
    activation_string: str,
    conv1d_padding: int,
    conv1d_stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    casual_conv1d_compatible = conv1d_num_groups == conv1d_weight.size(0) and conv1d_weight.size(1) == 1
    sequence_length = x.size(1)
    kernel_size = conv1d_weight.size(-1)

    assert conv1d_stride == 1
    assert conv1d_padding == kernel_size - 1

    x = _apply_mask_to_padding_states(x, attention_mask)

    if is_kernel_allowed(Kernel.causal_conv1d) and casual_conv1d_compatible:
        use_activation_inside_kernel = activation_string in [None, "silu", "swish"]

        if input_state is None:
            x = x.transpose(-1, -2)

            if return_cache_state:
                # F.pad trims the x if sequence_length > kernel_size
                input_state = F.pad(x, (kernel_size - sequence_length, 0))

            x = causal_conv1d_fn(
                x=x,
                weight=conv1d_weight.squeeze(1),
                bias=conv1d_bias,
                activation=activation_string if use_activation_inside_kernel else None,
            )

            x = x.transpose(-1, -2)
        else:
            assert sequence_length == 1

            # we clone to prevent modification in-place
            # torch compile can remove the clone if its not needed
            # this is to prevent silent incorrectness down the line in the model
            input_state_buffer = input_state.clone()
            x = causal_conv1d_update(
                x=x,
                conv_state=input_state_buffer,
                weight=conv1d_weight.squeeze(1),
                bias=conv1d_bias,
                activation=activation_string if use_activation_inside_kernel else None,
            )
            input_state = input_state_buffer if return_cache_state else None

        if not use_activation_inside_kernel:
            x = get_activation_function(activation_string)(x)
    else:
        if input_state is None:
            x = x.transpose(-1, -2)

            if return_cache_state:
                # F.pad trims the x if sequence_length > kernel_size
                input_state = F.pad(x, (kernel_size - sequence_length, 0))

            x = F.conv1d(
                input=x,
                weight=conv1d_weight,
                bias=conv1d_bias,
                stride=conv1d_stride,
                padding=conv1d_padding,
                groups=conv1d_num_groups,
            )

            # removes padding on the right side of the sequence
            x = x[..., : 1 - kernel_size]
            x = x.transpose(-1, -2)
        else:
            assert sequence_length == 1

            input_state = input_state.roll(shifts=-1, dims=-1)
            input_state[..., -1] = x[:, 0]

            x = (input_state * conv1d_weight.squeeze(1)).sum(dim=-1)
            x = x[:, None, :]
            if conv1d_bias is not None:
                x = x + conv1d_bias

            if not return_cache_state:
                input_state = None

        x = get_activation_function(activation_string)(x)
        x = _apply_mask_to_padding_states(x, attention_mask)

    return x, input_state


class CausalConvolution(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_groups: int,
        activation_function: str,
        add_bias: bool,
        initializer_range: float | None,
        m_width: float,
        init_method: str,
        num_layers: int,
        layer_idx: int,
        use_padding_free_transformer: bool,
    ) -> CausalConvolution:
        super().__init__()

        if use_padding_free_transformer:
            raise NotImplementedError()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.layer_idx = layer_idx
        self.activation_string = activation_function

        std = _get_std_for_linear(initializer_range, init_method, m_width)
        self.input_projection = ParameterizedLinear(hidden_size, in_channels, bias=add_bias, std=std)

        divide_if_divisible(in_channels, num_groups, "")
        divide_if_divisible(out_channels, num_groups, "")

        if is_glu(self.activation_string):
            intermediate_size = divide_if_divisible(out_channels, 2, "")
        else:
            intermediate_size = out_channels

        self.conv1d = ParameterizedConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=add_bias,
            padding=kernel_size - 1,
            groups=num_groups,
            std=std,
        )

        self.activation_function = get_activation_function(self.activation_string)

        self.output_projection = ParameterizedLinear(
            intermediate_size, hidden_size, bias=add_bias, std=std / math.sqrt(2 * num_layers)
        )

        self.casual_conv1d_compatible = self.num_groups == self.in_channels == self.out_channels
        self.use_activation_inside_kernel = self.activation_string in [None, "silu", "swish"]

    def forward(
        self, x: torch.Tensor, input_state: torch.Tensor | None, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.input_projection(x)

        x, input_state = causal_convolution(
            x=x,
            input_state=input_state,
            attention_mask=attention_mask,
            conv1d_weight=self.conv1d.weight,
            conv1d_bias=self.conv1d.bias,
            conv1d_num_groups=self.num_groups,
            return_cache_state=cache_params is not None,
            activation_string=self.activation_string,
            conv1d_padding=self.kernel_size - 1,
            conv1d_stride=1,
        )

        x = self.output_projection(x)

        return x, input_state
