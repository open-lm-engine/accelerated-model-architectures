# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from ...accelerator import KernelBackend
from ...custom_op import CustomOp, ctx_save_for_backward
from ...math import divide_if_divisible
from ...torch_utils import clip_gradients, tanh
from ...utils import empty_like_contiguous, is_triton_available, zeros_like_contiguous


if is_triton_available():
    from .triton_implementation import rnn_backward_triton, rnn_forward_triton


def _get_num_heads(x: torch.Tensor, W: torch.Tensor, run_check: bool) -> tuple[int, int, int]:
    Nx = x.size(-2)
    Nw = W.size(0)
    N = max(Nx, Nw)

    if run_check:
        assert N % Nx == 0
        assert N % Nw == 0

    return Nx, Nw, N


def _get_backward_tensor(y: torch.Tensor, Nx: int, N: int) -> torch.Tensor:
    if Nx == N:
        dx = empty_like_contiguous(y)
    else:
        x_shape = list(y.size())
        x_shape[-2] = Nx
        dx = torch.zeros(x_shape, device=y.device, dtype=torch.float32)

    return dx


class _RNN(CustomOp):
    @staticmethod
    def forward_backward_torch(
        x: torch.Tensor,
        W: torch.Tensor,
        h0: torch.Tensor | None,
        gradient_clipping: float | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Nx, Nw, N = _get_num_heads(x=x, W=W, run_check=False)

        y_shape = list(x.size())
        y_shape[-2] = N
        y = torch.empty(y_shape, device=x.device, dtype=x.dtype)

        if cu_seqlens is None:
            B, S, _, H = x.size()
        else:
            B = cu_seqlens.size(0) - 1
            S = max_seqlen
            H = x.size(-1)

        Gx = N // Nx
        Gw = N // Nw

        x = x.repeat_interleave(Gx, dim=-2)
        W = W.repeat_interleave(Gw, dim=0)[None, ...]

        if h0 is None:
            h0 = torch.zeros(B, N, H, device=x.device, dtype=x.dtype)

        if cu_seqlens is not None:
            h0 = h0.clone()
            start = cu_seqlens[:-1]
            end = cu_seqlens[1:]

        for s in range(S):
            if cu_seqlens is None:
                h = h0[..., None, :] @ W + x[:, s, :, None, :]
            else:
                offset = start + s
                unfinished = offset < end
                offset_unfinished = offset[unfinished]

                h = h0[unfinished, :, None, :] @ W + x[offset_unfinished, :, None, :]

            h = tanh(h)
            h = h.squeeze(-2)
            h = clip_gradients(h, gradient_clipping)

            if cu_seqlens is None:
                y[:, s] = h
                h0 = h
            else:
                y[offset_unfinished] = h
                h0[unfinished] = h

        return y, h0

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        W: torch.Tensor,
        h0: torch.Tensor | None,
        gradient_clipping: float | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
        kernel_backend: KernelBackend,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert kernel_backend in [KernelBackend.cuda, KernelBackend.triton]

        Nx, _, N = _get_num_heads(x=x, W=W, run_check=False)
        y_shape = list(x.size())
        y_shape[-2] = N

        y = torch.empty(y_shape, device=x.device, dtype=x.dtype)

        rnn_forward_triton(
            x=x,
            W=W,
            h0=h0,
            y=y,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        ctx_save_for_backward(ctx, W, y, h0, cu_seqlens)

        ctx.max_seqlen = max_seqlen
        ctx.gradient_clipping = gradient_clipping
        ctx.Nx = Nx

        ht = y[:, -1] if cu_seqlens is None else y[cu_seqlens[1:] - 1]
        ht = ht.detach()

        return y, ht

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dht: torch.Tensor | None) -> tuple[torch.Tensor]:
        W, y, h0, cu_seqlens = ctx.saved_tensors
        Nx = ctx.Nx
        N = y.size(-2)

        if Nx == N:
            dx = empty_like_contiguous(y)
        else:
            x_shape = list(y.size())
            x_shape[-2] = Nx
            dx = torch.zeros(x_shape, device=y.device, dtype=torch.float32)

        dW = zeros_like_contiguous(W, dtype=torch.float32)
        dh0 = empty_like_contiguous(h0) if h0 is not None and h0.requires_grad else None

        rnn_backward_triton(
            W=W,
            y=y,
            h0=h0,
            dy=dy,
            dht=dht,
            dx=dx,
            dW=dW,
            dh0=dh0,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            gradient_clipping=ctx.gradient_clipping,
        )

        dx = dx.type_as(y)
        dW = dW.type_as(W)

        return dx, dW, dh0, *[None] * 4


def rnn(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_state: torch.Tensor | None = None,
    gradient_clipping: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    *,
    kernel_backend: KernelBackend | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    computes multihead RNN recurrent update over the sequence length: `tanh(input_state @ weight + input)`

    :param input: input tensor of shape (B, S, Nx, H) where Nx is the number of input heads and H is the head
        dimension. Should have shape (T, Nx, H) and `cu_seqlens` should be passed.
    :type input: torch.Tensor
    :param weight: weight tensor of shape (Nw, H, H)
    :type weight: torch.Tensor
    :param input_state: starting state of shape (B, N, H), where N = max{Nx, Nw}. None means starting state is
        0 tensor. Defaults to None.
    :type input_state: torch.Tensor | None
    :param gradient_clipping: gradient clipping for the state gradient in backward, None implies no clipping.
        Defaults to None.
    :type gradient_clipping: float | None
    :param cu_seqlens: cumulative sequence length (must contain 0 as first element). Defaults to None.
    :type cu_seqlens: torch.Tensor | None
    :param max_seqlen: max sequence length in the batch. Defaults to None.
    :type max_seqlen: int | None
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output tensor of shape (B, S, N, H) if `cu_seqlens` is None else (T, N, H) and output state of
        shape (B, N, H).
    :rtype: tuple[Tensor, Tensor]
    """

    assert input.dim() == 3 + (cu_seqlens is None)

    if cu_seqlens is None:
        assert max_seqlen is None
        B, _, _, H = input.size()
    else:
        assert max_seqlen is not None
        assert cu_seqlens.dim() == 1

        B = cu_seqlens.size(0) - 1
        H = input.size(-1)

    _, Nw, N = _get_num_heads(x=input, W=weight, run_check=True)

    assert weight.size() == (Nw, H, H)

    if input_state is not None:
        assert input_state.size() == (B, N, H)

    if gradient_clipping is not None and gradient_clipping < 0:
        gradient_clipping = -gradient_clipping

    input, input_state = _RNN.run(
        x=input,
        W=weight,
        h0=input_state,
        gradient_clipping=gradient_clipping,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        kernel_backend=kernel_backend,
    )

    return input, input_state


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
