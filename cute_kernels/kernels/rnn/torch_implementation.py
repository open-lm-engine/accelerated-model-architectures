# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...torch_math import sigmoid, tanh


class _GradientClipping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, gradient_clipping: float) -> torch.Tensor:
        ctx.gradient_clipping = gradient_clipping
        return x

    @staticmethod
    def backward(ctx, x_grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        gradient_clipping = ctx.gradient_clipping
        x_grad = x_grad.clip(-gradient_clipping, gradient_clipping)
        return x_grad, None


def _activation_with_clipped_gradients(
    x: torch.Tensor, activation_function: str, relu_negative_slope: float | None, gradient_clipping: float | None
) -> torch.Tensor:
    if activation_function == "leaky_relu":
        x = F.leaky_relu(x, negative_slope=relu_negative_slope)
    elif activation_function == "sigmoid":
        x = sigmoid(x)
    elif activation_function == "tanh":
        x = tanh(x)
    else:
        raise ValueError(f"unexpected activation_function ({activation_function})")

    if gradient_clipping is not None:
        x = _GradientClipping.apply(x, gradient_clipping)

    return x


def rnn_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_state: torch.Tensor | None = None,
    gradient_clipping: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | int | None = None,
    activation_function: str = "tanh",
    relu_negative_slope: float | None = None,
) -> torch.Tensor:
    assert activation_function in ["leaky_relu", "sigmoid", "tanh"]

    if activation_function == "leaky_relu":
        assert relu_negative_slope is not None
    else:
        assert relu_negative_slope is None

    if gradient_clipping is not None and gradient_clipping < 0:
        gradient_clipping = -gradient_clipping

    output = torch.empty_like(input)

    weight = weight.unsqueeze(0)
    input = input.unsqueeze(-2)

    if cu_seqlens is None:
        assert max_seqlen is None
        B, S, N, _, H = input.size()

        if input_state is None:
            input_state = torch.zeros(B, N, H, device=input.device, dtype=input.dtype)

        input_state = input_state.unsqueeze(-2)

        # input -> (B, S, N, 1, H)
        # weight -> (1, N, H, H)
        # input_state -> (B, N, 1, H)

        for s in range(S):
            # (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
            input_state = input_state @ weight + input[:, s]

            input_state = _activation_with_clipped_gradients(
                x=input_state,
                activation_function=activation_function,
                relu_negative_slope=relu_negative_slope,
                gradient_clipping=gradient_clipping,
            )

            output[:, s] = input_state.squeeze(-2)
    else:
        assert max_seqlen is not None
        B = cu_seqlens.numel() - 1
        _, N, _, H = input.size()

        if input_state is None:
            input_state = torch.zeros(B, N, H, device=input.device, dtype=input.dtype)
        else:
            input_state = input_state.clone()

        # input -> (cu_seqlens[-1], N, 1, H)
        # weight -> (1, N, H, H)
        # input_state -> (B, N, H)

        start = cu_seqlens[:-1]
        end = cu_seqlens[1:]

        for s in range(max_seqlen):
            offset = start + s
            unfinished = offset < end
            new_state = input_state.unsqueeze(-2)

            offset_unfinished = offset[unfinished]

            # don't update the finished sequences
            # (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
            new_state = new_state[unfinished] @ weight + input[offset_unfinished]

            new_state = _activation_with_clipped_gradients(
                x=new_state,
                activation_function=activation_function,
                relu_negative_slope=relu_negative_slope,
                gradient_clipping=gradient_clipping,
            )

            new_state = new_state.squeeze(-2)

            output[offset_unfinished] = new_state
            input_state[unfinished] = new_state

    return output
