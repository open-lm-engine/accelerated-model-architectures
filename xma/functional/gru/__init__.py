# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...custom_op import CustomOp, ctx_needs_gradients, ctx_save_for_backward
from ...enums import KernelBackend
from ...torch_utils import clip_gradients, sigmoid, tanh
from ...utils import (
    empty_like_contiguous,
    get_max_seqlen_and_max_seqlen_tensor,
    is_triton_available,
    zeros_like_contiguous,
)


if is_triton_available():
    from .triton_implementation import gru_backward_triton, gru_forward_triton


class _GRU(CustomOp):
    @staticmethod
    def forward_backward_torch(
        input: torch.Tensor,
        weight: torch.Tensor,
        forget_input: torch.Tensor,
        forget_weight: torch.Tensor,
        reset_input: torch.Tensor,
        reset_weight: torch.Tensor,
        input_state: torch.Tensor | None,
        gradient_clipping: float | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: torch.Tensor | int | None,
    ) -> torch.Tensor:
        input_shape = input.size()

        Nx = input_shape[-2]
        Nxf = forget_input.size(-2)
        Nxr = reset_input.size(-2)

        Nw = weight.size(0)
        Nwf = forget_input.size(0)
        Nwr = reset_weight.size(0)

        N = max(Nx, Nxf, Nxr, Nw, Nwf, Nwr)

        output_shape = list(input_shape)
        output_shape[-2] = N

        output = torch.empty(output_shape, device=input.device, dtype=input.dtype)

        if cu_seqlens is None:
            B, S, _, H = input.size()
        else:
            _, _, H = input.size()
            B = cu_seqlens.size(0) - 1
            S = max_seqlen.item() if isinstance(max_seqlen, torch.Tensor) else max_seqlen

        Gx = N // Nx
        Gxf = N // Nxf
        Gxr = N // Nxr

        Gw = N // Nw
        Gwf = N // Nwf
        Gwr = N // Nwr

        input = input.repeat_interleave(Gx, dim=-2)
        forget_input = forget_input.repeat_interleave(Gxf, dim=-2)
        reset_input = reset_input.repeat_interleave(Gxr, dim=-2)

        weight = weight.repeat_interleave(Gw, dim=0)
        forget_weight = forget_weight.repeat_interleave(Gwf, dim=-2)
        reset_weight = reset_weight.repeat_interleave(Gwr, dim=-2)

        W = weight[None, ...]
        Wf = forget_weight[None, ...]
        Wr = reset_weight[None, ...]

        if input_state is None:
            input_state = torch.zeros(B, N, H, device=input.device, dtype=input.dtype)

        if cu_seqlens is not None:
            input_state = input_state.clone()
            start = cu_seqlens[:-1]
            end = cu_seqlens[1:]

        for s in range(S):
            if cu_seqlens is None:
                new_state = input_state[..., None, :]

                forget_gate = new_state @ Wf + forget_input[:, s, :, None, :]
                reset_gate = new_state @ Wr + reset_input[:, s, :, None, :]

                forget_gate = sigmoid(forget_gate)
                reset_gate = sigmoid(reset_gate)

                # (B, N, 1, H) = [(B, N, 1, H) * (B, N, 1, H)] @ (1, N, H, H) + (B, N, 1, H)
                possible_new_state = (new_state * reset_gate) @ W + input[:, s, :, None, :]
            else:
                offset = start + s
                unfinished = offset < end

                new_state = input_state[unfinished, :, None, :]
                offset_unfinished = offset[unfinished]

                # don't update the finished sequences
                # (B, N, 1, H) = (B, N, 1, H) @ (1, N, H, H) + (B, N, 1, H)
                forget_gate = new_state @ Wf + forget_input[offset_unfinished, :, None, :]
                reset_gate = new_state @ Wr + reset_input[offset_unfinished, :, None, :]

                forget_gate = sigmoid(forget_gate)
                reset_gate = sigmoid(reset_gate)

                # (B, N, 1, H) = [(B, N, 1, H) * (B, N, 1, H)] @ (1, N, H, H) + (B, N, 1, H)
                possible_new_state = (new_state * reset_gate) @ weight.unsqueeze(0) + input[
                    offset_unfinished, :, None, :
                ]

            possible_new_state = tanh(possible_new_state)
            new_state = forget_gate * new_state + (1 - forget_gate) * possible_new_state

            new_state = clip_gradients(new_state, gradient_clipping)
            new_state = new_state.squeeze(-2)

            if cu_seqlens is None:
                output[:, s] = new_state
                input_state = new_state
            else:
                output[offset_unfinished] = new_state
                input_state[unfinished] = new_state

        return output

    @staticmethod
    def forward_triton(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        forget_input: torch.Tensor,
        forget_weight: torch.Tensor,
        reset_input: torch.Tensor,
        reset_weight: torch.Tensor,
        input_state: torch.Tensor | None,
        gradient_clipping: float | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: torch.Tensor | int | None,
    ) -> torch.Tensor:
        needs_grad = ctx_needs_gradients(ctx)

        output = empty_like_contiguous(input)
        max_seqlen_tensor, max_seqlen = get_max_seqlen_and_max_seqlen_tensor(max_seqlen)
        forget_gate = empty_like_contiguous(input) if needs_grad else None
        reset_gate = empty_like_contiguous(input) if needs_grad else None
        output_update = empty_like_contiguous(input) if needs_grad else None

        gru_forward_triton(
            input=input,
            weight=weight,
            forget_input=forget_input,
            forget_weight=forget_weight,
            forget_gate=forget_gate,
            reset_input=reset_input,
            reset_weight=reset_weight,
            reset_gate=reset_gate,
            output_update=output_update,
            input_state=input_state,
            output=output,
            cu_seqlens=cu_seqlens,
            max_seqlen_tensor=max_seqlen_tensor,
            max_seqlen=max_seqlen,
        )

        ctx_save_for_backward(
            ctx,
            weight,
            forget_weight,
            forget_gate,
            reset_weight,
            reset_gate,
            output_update,
            output,
            input_state,
            cu_seqlens,
            max_seqlen_tensor,
        )

        ctx.max_seqlen = max_seqlen
        ctx.gradient_clipping = gradient_clipping

        return output

    @staticmethod
    def backward_triton(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor | None]:
        (
            weight,
            forget_weight,
            forget_gate,
            reset_weight,
            reset_gate,
            output_update,
            output,
            input_state,
            cu_seqlens,
            max_seqlen_tensor,
        ) = ctx.saved_tensors

        input_grad = empty_like_contiguous(output)
        forget_input_grad = empty_like_contiguous(output)
        reset_input_grad = empty_like_contiguous(output)
        weight_grad = zeros_like_contiguous(weight, dtype=torch.float32)
        forget_weight_grad = zeros_like_contiguous(weight, dtype=torch.float32)
        reset_weight_grad = zeros_like_contiguous(weight, dtype=torch.float32)

        gru_backward_triton(
            weight=weight,
            output=output,
            forget_weight=forget_weight,
            forget_gate=forget_gate,
            forget_input_grad=forget_input_grad,
            forget_weight_grad=forget_weight_grad,
            reset_weight=reset_weight,
            reset_gate=reset_gate,
            reset_input_grad=reset_input_grad,
            reset_weight_grad=reset_weight_grad,
            output_update=output_update,
            input_state=input_state,
            output_grad=output_grad,
            input_grad=input_grad,
            weight_grad=weight_grad,
            cu_seqlens=cu_seqlens,
            max_seqlen_tensor=max_seqlen_tensor,
            max_seqlen=ctx.max_seqlen,
            gradient_clipping=ctx.gradient_clipping,
        )

        weight_grad = weight_grad.type_as(weight)
        forget_weight_grad = forget_weight_grad.type_as(forget_weight)
        reset_weight_grad = reset_weight_grad.type_as(reset_weight)

        return (
            input_grad,
            weight_grad,
            forget_input_grad,
            forget_weight_grad,
            reset_input_grad,
            reset_weight_grad,
            *[None] * 5,
        )


def gru(
    input: torch.Tensor,
    weight: torch.Tensor,
    forget_input: torch.Tensor,
    forget_weight: torch.Tensor,
    reset_input: torch.Tensor,
    reset_weight: torch.Tensor,
    input_state: torch.Tensor | None = None,
    gradient_clipping: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | int | None = None,
    *,
    kernel_backend: KernelBackend | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """computes multihead RNN: tanh(`input_state` @ `weight` + `input`)

    Args:
        input (torch.Tensor): input tensor of shape (B, S, N, H) where N is the number of heads and H is the head
            dimension. Should have shape (T, N, H) and `cu_seqlens` should be passed.
        weight (torch.Tensor): weight tensor of shape (N, H, H)
        input_state (torch.Tensor | None, optional): starting state of shape (B, N, H), None means starting state
            is 0 tensor. Defaults to None.
        gradient_clipping (float | None, optional): gradient clipping for the state gradient in backward, None
            implies no clipping. Defaults to None.
        cu_seqlens (torch.Tensor | None, optional): cumulative sequence length (must contain 0 as first element). Defaults to None.
        max_seqlen (torch.Tensor | int | None, optional): max sequence length in the batch. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: output tensor of shape (B, S, N, H) and output state tensor of shape (B, N, H)
    """

    assert all([i.dim() == (3 + (cu_seqlens is None)) for i in (input, forget_input, reset_input)])

    if cu_seqlens is None:
        assert max_seqlen is None
        B, _, Nx, H = input.size()
    else:
        assert max_seqlen is not None
        assert cu_seqlens.dim() == 1

        _, Nx, H = input.size()
        B = cu_seqlens.size(0) - 1

    Nxf = forget_input.size(-2)
    Nxr = reset_input.size(-2)

    Nw = weight.size(0)
    Nwf = forget_weight.size(0)
    Nwr = reset_weight.size(0)

    N = max(Nx, Nxf, Nxr, Nw, Nwf, Nwr)

    assert weight.size() == (Nw, H, H)
    assert all([N % i == 0 for i in (Nx, Nxf, Nxr, Nw, Nwf, Nwr)])

    if input_state is not None:
        assert input_state.size() == (B, N, H)

    if gradient_clipping is not None and gradient_clipping < 0:
        gradient_clipping = -gradient_clipping

    input = _GRU.run(
        input=input,
        weight=weight,
        forget_input=forget_input,
        forget_weight=forget_weight,
        reset_input=reset_input,
        reset_weight=reset_weight,
        input_state=input_state,
        gradient_clipping=gradient_clipping,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        kernel_backend=kernel_backend,
    )

    input_state = input[:, -1] if cu_seqlens is None else input[cu_seqlens[1:] - 1]

    return input, input_state
