# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from ..accelerator import Accelerator, KernelBackend
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
    return_cache_state: bool,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    groups: int,
    padding: int,
    stride: int = 1,
    activation_function: str | Callable | None = "silu",
    *,
    kernel_backend: KernelBackend | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    S = x.size(1)
    K = weight.size(-1)

    assert stride == 1
    assert padding == K - 1

    x = _apply_mask_to_padding_states(x, attention_mask)

    if kernel_backend is None:
        kernel_backend = Accelerator.get_kernel_backend()
    else:
        assert kernel_backend.verify_accelerator()

    if (
        is_causal_conv1d_available()
        and kernel_backend in [KernelBackend.cuda, KernelBackend.triton]
        and groups == weight.size(0)
        and weight.size(1) == 1
    ):
        use_activation_inside_kernel = activation_function in [None, "silu", "swish"]

        if input_state is None:
            x = x.transpose(-1, -2)

            if return_cache_state:
                # F.pad trims the x if sequence_length > kernel_size
                input_state = F.pad(x, (K - S, 0))

            x = causal_conv1d_fn(
                x=x,
                weight=weight.squeeze(1),
                bias=bias,
                activation=activation_function if use_activation_inside_kernel else None,
            )

            x = x.transpose(-1, -2)
        else:
            assert S == 1
            input_state = input_state.clone()

            x = causal_conv1d_update(
                x=x,
                conv_state=input_state,
                weight=weight.squeeze(1),
                bias=bias,
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

            x = F.conv1d(input=x, weight=weight, bias=bias, stride=stride, padding=padding, groups=groups)

            # removes padding on the right side of the sequence
            x = x[..., : 1 - K]
            x = x.transpose(-1, -2)
        else:
            assert S == 1

            input_state = input_state.roll(shifts=-1, dims=-1)
            input_state[..., -1] = x[:, 0]

            x = (input_state * weight.squeeze(1)).sum(dim=-1)
            x = x[:, None, :]
            if bias is not None:
                x = x + bias

            if not return_cache_state:
                input_state = None

        x = activation_function(x)
        x = _apply_mask_to_padding_states(x, attention_mask)

    return x, input_state
