# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...accelerator import KernelBackend
from ...custom_op import CustomOp, ctx_needs_gradients, ctx_save_for_backward
from ...torch_utils import clip_gradients, sigmoid, tanh
from ...utils import empty_like_contiguous, is_triton_available, zeros_like_contiguous
from ..rnn import _get_backward_tensor
from .utils import _get_num_heads


if is_triton_available():
    from .triton_implementation import gru_backward_triton, gru_forward_triton


class _GRU(CustomOp):
    @staticmethod
    def forward_backward_torch(
        x: torch.Tensor,
        W: torch.Tensor,
        xf: torch.Tensor,
        Wf: torch.Tensor,
        xr: torch.Tensor,
        Wr: torch.Tensor,
        h0: torch.Tensor | None,
        gradient_clipping: float | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
    ) -> torch.Tensor:
        Nx, Nxf, Nxr, Nw, Nwf, Nwr, N = _get_num_heads(x=x, W=W, xf=xf, Wf=Wf, xr=xr, Wr=Wr, run_check=False)

        y_shape = list(x.size())
        y_shape[-2] = N
        y = torch.empty(y_shape, device=x.device, dtype=x.dtype)

        if cu_seqlens is None:
            B, S, _, H = x.size()
        else:
            B = cu_seqlens.size(0) - 1
            S = max_seqlen.item() if isinstance(max_seqlen, torch.Tensor) else max_seqlen
            H = x.size(-1)

        Gx = N // Nx
        Gxf = N // Nxf
        Gxr = N // Nxr

        Gw = N // Nw
        Gwf = N // Nwf
        Gwr = N // Nwr

        x = x.repeat_interleave(Gx, dim=-2)
        xf = xf.repeat_interleave(Gxf, dim=-2)
        xr = xr.repeat_interleave(Gxr, dim=-2)

        W = W.repeat_interleave(Gw, dim=0)[None, ...]
        Wf = Wf.repeat_interleave(Gwf, dim=0)[None, ...]
        Wr = Wr.repeat_interleave(Gwr, dim=0)[None, ...]

        if h0 is None:
            h0 = torch.zeros(B, N, H, device=x.device, dtype=x.dtype)

        if cu_seqlens is not None:
            h0 = h0.clone()
            start = cu_seqlens[:-1]
            end = cu_seqlens[1:]

        for s in range(S):
            if cu_seqlens is None:
                f = h0[..., None, :] @ Wf + xf[:, s, :, None, :]
                r = h0[..., None, :] @ Wr + xr[:, s, :, None, :]
            else:
                offset = start + s
                unfinished = offset < end
                offset_unfinished = offset[unfinished]

                f = h0[unfinished, :, None, :] @ Wf + xf[offset_unfinished, :, None, :]
                r = h0[unfinished, :, None, :] @ Wr + xr[offset_unfinished, :, None, :]

            f = sigmoid(f)
            r = sigmoid(r)

            if cu_seqlens is None:
                z = (h0[..., None, :] * r) @ W + x[:, s, :, None, :]
            else:
                z = (h0[unfinished, :, None, :] * r) @ W + x[offset_unfinished, :, None, :]

            z = tanh(z)

            if cu_seqlens is None:
                h = f * h0[..., None, :] + (1 - f) * z
            else:
                h = f * h0[unfinished, :, None, :] + (1 - f) * z

            h = h.squeeze(-2)
            h = clip_gradients(h, gradient_clipping)

            if cu_seqlens is None:
                y[:, s] = h
                h0 = h
            else:
                y[offset_unfinished] = h
                h0[unfinished] = h

        return y

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        W: torch.Tensor,
        xf: torch.Tensor,
        Wf: torch.Tensor,
        xr: torch.Tensor,
        Wr: torch.Tensor,
        h0: torch.Tensor | None,
        gradient_clipping: float | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
        kernel_backend: KernelBackend,
    ) -> torch.Tensor:
        assert kernel_backend in [KernelBackend.cuda, KernelBackend.triton]

        Nx, Nxf, Nxr, _, _, _, N = _get_num_heads(x=x, W=W, xf=xf, Wf=Wf, xr=xr, Wr=Wr, run_check=False)
        y_shape = list(x.size())
        y_shape[-2] = N

        needs_grad = ctx_needs_gradients(ctx)

        y = torch.empty(y_shape, device=x.device, dtype=x.dtype)
        f = torch.empty(y_shape, device=x.device, dtype=x.dtype) if needs_grad and Nxf == N else None
        r = torch.empty(y_shape, device=x.device, dtype=x.dtype) if needs_grad and Nxr == N else None
        z = torch.empty(y_shape, device=x.device, dtype=x.dtype) if needs_grad and Nx == N else None

        gru_forward_triton(
            x=x,
            W=W,
            xf=xf,
            Wf=Wf,
            f=f,
            xr=xr,
            Wr=Wr,
            r=r,
            z=z,
            h0=h0,
            y=y,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        ctx_save_for_backward(
            ctx,
            W,
            Wf,
            f,
            Wr,
            r,
            z,
            y,
            h0,
            cu_seqlens,
            x if z is None else None,
            xf if f is None else None,
            xr if r is None else None,
        )

        ctx.max_seqlen = max_seqlen
        ctx.gradient_clipping = gradient_clipping
        ctx.num_heads = Nx, Nxf, Nxr

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor | None]:
        W, Wf, f, Wr, r, z, y, h0, cu_seqlens, x, xf, xr = ctx.saved_tensors
        Nx, Nxf, Nxr = ctx.num_heads

        dx = _get_backward_tensor(y=y, Nx=Nx, N=y.size(-2))
        dxf = _get_backward_tensor(y=y, Nx=Nxf, N=y.size(-2))
        dxr = _get_backward_tensor(y=y, Nx=Nxr, N=y.size(-2))

        dW = zeros_like_contiguous(W, dtype=torch.float32)
        dWf = zeros_like_contiguous(Wf, dtype=torch.float32)
        dWr = zeros_like_contiguous(Wr, dtype=torch.float32)

        dh0 = empty_like_contiguous(h0) if h0 is not None and h0.requires_grad else None

        gru_backward_triton(
            x=x,
            W=W,
            y=y,
            xf=xf,
            Wf=Wf,
            f=f,
            dxf=dxf,
            dWf=dWf,
            xr=xr,
            Wr=Wr,
            r=r,
            dxr=dxr,
            dWr=dWr,
            z=z,
            h0=h0,
            dy=dy,
            dx=dx,
            dW=dW,
            dh0=dh0,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            gradient_clipping=ctx.gradient_clipping,
        )

        dx = dx.type_as(y)
        dxf = dxf.type_as(y)
        dxr = dxr.type_as(y)

        dW = dW.type_as(W)
        dWf = dWf.type_as(Wf)
        dWr = dWr.type_as(Wr)

        return dx, dW, dxf, dWf, dxr, dWr, dh0, *[None] * 4


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
    max_seqlen: int | None = None,
    *,
    kernel_backend: KernelBackend | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    computes multihead RNN: `tanh(input_state @ weight + input)`

    :param input: input tensor of shape (B, S, Nx, H) where Nx is the number of input heads and H is the head
        dimension. Should have shape (T, Nx, H) and `cu_seqlens` should be passed.
    :type input: torch.Tensor
    :param weight: weight tensor of shape (Nw, H, H)
    :type weight: torch.Tensor
    :param forget_input: forget input tensor of shape (B, S, Nxf, H) where Nxf is the number of input heads and H is the head
        dimension. Should have shape (T, Nxf, H) and `cu_seqlens` should be passed.
    :type forget_input: torch.Tensor
    :param forget_weight: forget weight tensor of shape (NWf, H, H)
    :type forget_weight: torch.Tensor
    :param reset_input: reset input tensor of shape (B, S, Nxr, H) where Nxr is the number of input heads and H is the head
        dimension. Should have shape (T, Nxr, H) and `cu_seqlens` should be passed.
    :type reset_input: torch.Tensor
    :param reset_weight: reset weight tensor of shape (Nwr, H, H)
    :type reset_weight: torch.Tensor
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

    if cu_seqlens is None:
        assert max_seqlen is None
        B, S, _, H = input.size()
    else:
        assert max_seqlen is not None
        assert cu_seqlens.dim() == 1

        B = cu_seqlens.size(0) - 1
        T, _, H = input.size()

    Nx, Nxf, Nxr, Nw, Nwf, Nwr, N = _get_num_heads(
        x=input, W=weight, xf=forget_input, Wf=forget_weight, xr=reset_input, Wr=reset_weight, run_check=True
    )

    if cu_seqlens is None:
        input.size() == (B, S, Nx, H)
        forget_input.size() == (B, S, Nxf, H)
        reset_input.size() == (B, S, Nxr, H)
    else:
        input.size() == (T, Nx, H)
        forget_input.size() == (T, Nxf, H)
        reset_input.size() == (T, Nxr, H)

    assert weight.size() == (Nw, H, H)
    assert forget_weight.size() == (Nwf, H, H)
    assert reset_weight.size() == (Nwr, H, H)

    if input_state is not None:
        assert input_state.size() == (B, N, H)

    if gradient_clipping is not None and gradient_clipping < 0:
        gradient_clipping = -gradient_clipping

    input = _GRU.run(
        x=input,
        W=weight,
        xf=forget_input,
        Wf=forget_weight,
        xr=reset_input,
        Wr=reset_weight,
        h0=input_state,
        gradient_clipping=gradient_clipping,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        kernel_backend=kernel_backend,
    )

    input_state = input[:, -1] if cu_seqlens is None else input[cu_seqlens[1:] - 1]

    return input, input_state
