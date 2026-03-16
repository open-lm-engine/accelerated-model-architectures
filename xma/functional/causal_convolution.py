# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from ..utils import is_causal_conv1d_available


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
    conv1d_padding: int,
    conv1d_stride: int = 1,
    activation_function: str | Callable | None = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    S = x.size(1)
    K = conv1d_weight.size(-1)

    assert conv1d_stride == 1
    assert conv1d_padding == K - 1

    x = _apply_mask_to_padding_states(x, attention_mask)

    if is_causal_conv1d_available() and conv1d_num_groups == conv1d_weight.size(0) and conv1d_weight.size(1) == 1:
        use_activation_inside_kernel = activation_function in [None, "silu", "swish"]

        if input_state is None:
            x = x.transpose(-1, -2)

            if return_cache_state:
                # F.pad trims the x if sequence_length > kernel_size
                input_state = F.pad(x, (K - S, 0))

            x = causal_conv1d_fn(
                x=x,
                weight=conv1d_weight.squeeze(1),
                bias=conv1d_bias,
                activation=activation_function if use_activation_inside_kernel else None,
            )

            x = x.transpose(-1, -2)
        else:
            assert S == 1
            input_state = input_state.clone()

            x = causal_conv1d_update(
                x=x,
                conv_state=input_state,
                weight=conv1d_weight.squeeze(1),
                bias=conv1d_bias,
                activation=activation_function if use_activation_inside_kernel else None,
            )

            if not return_cache_state:
                input_state = None

        if not use_activation_inside_kernel:
            x = activation_function(x)
    else:
        if input_state is None:
            x = x.transpose(-1, -2)

            if return_cache_state:
                # F.pad trims the x if sequence_length > kernel_size
                input_state = F.pad(x, (K - S, 0))

            x = F.conv1d(
                input=x,
                weight=conv1d_weight,
                bias=conv1d_bias,
                stride=conv1d_stride,
                padding=conv1d_padding,
                groups=conv1d_num_groups,
            )

            # removes padding on the right side of the sequence
            x = x[..., : 1 - K]
            x = x.transpose(-1, -2)
        else:
            assert S == 1

            input_state = input_state.roll(shifts=-1, dims=-1)
            input_state[..., -1] = x[:, 0]

            x = (input_state * conv1d_weight.squeeze(1)).sum(dim=-1)
            x = x[:, None, :]
            if conv1d_bias is not None:
                x = x + conv1d_bias

            if not return_cache_state:
                input_state = None

        x = activation_function(x)
        x = _apply_mask_to_padding_states(x, attention_mask)

    return x, input_state
