# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...accelerator import Accelerator, KernelBackend
from ...custom_op import CustomOp, ctx_save_for_backward
from ...utils import empty_like_contiguous, zeros_like_contiguous


class _HiPPO(CustomOp):
    @staticmethod
    def forward_backward_torch(
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        h0: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.empty_like(x)

        if cu_seqlens is None:
            B, S, H = x.size()
        else:
            B = cu_seqlens.size(0) - 1
            S = max_seqlen
            H = x.size(-1)

        N = B.size(0)

        if h0 is None:
            h0 = torch.zeros(B, H, N, device=x.device, dtype=x.dtype)

        if cu_seqlens is None:
            h0 = h0.clone()
            start = cu_seqlens[:-1]
            end = cu_seqlens[1:]

        B = B[None, :]

        for s in range(S):
            if cu_seqlens is None:
                h = h0.flatten(0, 1) @ A.T + x[:, s].flatten()[..., None] * B
                y[:, s] = h
                h0 = h
            else:
                offset = start + s
                unfinished = offset < end
                offset_unfinished = offset[unfinished]

                h = h0[unfinished].flatten(0, 1) @ A.T + x[offset_unfinished].flatten()[..., None] * B
                y[offset_unfinished] = h
                h0[unfinished] = h

        return y, h0

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        W: torch.Tensor,
        h0: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
        kernel_backend: KernelBackend,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert kernel_backend in [KernelBackend.cuda, KernelBackend.triton]

        y = torch.empty_like(x)

        hippo_forward_triton(
            x=x,
            W=W,
            h0=h0,
            y=y,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        ctx_save_for_backward(ctx, W, y, h0, cu_seqlens)

        ctx.max_seqlen = max_seqlen
        ctx.Nx = Nx

        ht = y[:, -1] if cu_seqlens is None else y[cu_seqlens[1:] - 1]
        ht = ht.detach()

        return y, ht

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dht: torch.Tensor | None) -> tuple[torch.Tensor]:
        W, y, h0, cu_seqlens = ctx.saved_tensors
        Nx = ctx.Nx
        N = y.size(-2)

        dx = _get_backward_tensor(y=y, Nx=Nx, N=N)
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


def hippo(
    input: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    input_state: torch.Tensor | None = None,
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
    :param A: weight tensor of shape (H, H)
    :type A: torch.Tensor
    :param B: weight tensor of shape (H,)
    :type B: torch.Tensor
    :param input_state: starting state of shape (B, N, H), where N = max{Nx, Nw}. None means starting state is
        0 tensor. Defaults to None.
    :type input_state: torch.Tensor | None
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

    assert A.size() == (H, H)
    assert B.size() == (H,)

    if input_state is not None:
        assert input_state.size() == (B, H)

    if kernel_backend is None:
        kernel_backend = Accelerator.get_kernel_backend()
    else:
        assert kernel_backend.verify_accelerator()

    x = input
    h0 = input_state

    if kernel_backend == KernelBackend.triton:
        pass
    elif kernel_backend == KernelBackend.torch:
        y = torch.empty_like(x)

        if cu_seqlens is None:
            B, S, H = x.size()
        else:
            B = cu_seqlens.size(0) - 1
            S = max_seqlen
            H = x.size(-1)

        N = B.size(0)

        if h0 is None:
            h0 = torch.zeros(B, H, N, device=x.device, dtype=x.dtype)

        if cu_seqlens is None:
            h0 = h0.clone()
            start = cu_seqlens[:-1]
            end = cu_seqlens[1:]

        B = B[None, :]

        for s in range(S):
            if cu_seqlens is None:
                h = h0.flatten(0, 1) @ A.T + x[:, s].flatten()[..., None] * B
            else:
                offset = start + s
                unfinished = offset < end
                offset_unfinished = offset[unfinished]

                h = h0[unfinished].flatten(0, 1) @ A.T + x[offset_unfinished].flatten()[..., None] * B

            if cu_seqlens is None:
                y[:, s] = h
                h0 = h
            else:
                y[offset_unfinished] = h
                h0[unfinished] = h
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return y, h0
