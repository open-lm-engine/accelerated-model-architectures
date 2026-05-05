# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from functools import partial

import torch

from ...accelerator import KernelBackend
from ...custom_op import CustomOp, ctx_save_for_backward
from ...torch_utils import clip_gradients, tanh
from ...utils import empty_like_contiguous, is_triton_available, zeros_like_contiguous
from .utils import _get_num_heads


if is_triton_available():
    from .triton_implementation import _MAX_BLOCK_SIZE_K, _m2rnn_backward_triton, _m2rnn_forward_triton


class _M2RNN(CustomOp):
    @staticmethod
    def forward_backward_torch(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        W: torch.Tensor,
        xf: torch.Tensor,
        h0: torch.Tensor | None,
        gradient_clipping: float | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Nq, Nk, Nv, Nw, Nxf, N = _get_num_heads(q=q, k=k, v=v, W=W, xf=xf, run_check=False)

        V = v.size(-1)

        if cu_seqlens is None:
            B, S, _, K = q.size()
            y = torch.empty(B, S, N, K, V, device=q.device, dtype=q.dtype)
        else:
            B = cu_seqlens.size(0) - 1
            S = max_seqlen.item() if isinstance(max_seqlen, torch.Tensor) else max_seqlen
            T, _, K = q.size()

            y = torch.empty(T, N, K, V, device=q.device, dtype=q.dtype)

        if h0 is None:
            h0 = torch.zeros(B, N, K, V, device=k.device, dtype=k.dtype)

        Gq = N // Nq
        Gk = N // Nk
        Gv = N // Nv

        Gw = N // Nw
        Gxf = N // Nxf

        q = q.repeat_interleave(Gq, dim=-2)
        k = k.repeat_interleave(Gk, dim=-2)
        v = v.repeat_interleave(Gv, dim=-2)
        W = W.repeat_interleave(Gw, dim=0)
        xf = xf.repeat_interleave(Gxf, dim=-1)

        # (B, S, N, K, V) = (B, S, N, K, 1) * (B, S, N, 1, V)
        x = k[..., None] * v[..., None, :]
        W = W[None, ...]

        if cu_seqlens is not None:
            h0 = h0.clone()
            start = cu_seqlens[:-1]
            end = cu_seqlens[1:]

        for s in range(S):
            if cu_seqlens is None:
                f = xf[:, s, :, None, None]
                # (B, N, K, V) = (B, N, K, V) @ (1, N, V, V) + (B, N, K, V)
                h = h0 @ W + x[:, s]
            else:
                offset = start + s
                unfinished = offset < end
                offset_unfinished = offset[unfinished]

                f = xf[offset_unfinished, :, None, None]
                # (B, N, K, V) = (B, N, K, V) @ (1, N, V, V) + (B, N, K, V)
                h = h0[unfinished] @ W + x[offset_unfinished]

            h = tanh(h)

            if cu_seqlens is None:
                h = f * h0 + (1 - f) * h
            else:
                h = f * h0[unfinished] + (1 - f) * h

            h = clip_gradients(h, gradient_clipping)

            if cu_seqlens is None:
                y[:, s] = h
                h0 = h
            else:
                y[offset_unfinished] = h
                h0[unfinished] = h

        y = q[..., None, :] @ y
        y = y.squeeze(-2)

        return y, h0

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        W: torch.Tensor,
        xf: torch.Tensor,
        h0: torch.Tensor | None,
        gradient_clipping: float | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
        kernel_backend: KernelBackend,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert kernel_backend in [KernelBackend.cuda, KernelBackend.triton]

        Nq, Nk, Nv, Nw, Nxf, N = _get_num_heads(q=q, k=k, v=v, W=W, xf=xf, run_check=False)

        if cu_seqlens is None:
            B = k.size(0)
        else:
            B = cu_seqlens.size(0) - 1

        K = k.size(-1)
        V = v.size(-1)

        ht = torch.empty(B, N, K, V, device=k.device, dtype=k.dtype)

        y_shape = list(v.size())
        y_shape[-2] = N

        if K > _MAX_BLOCK_SIZE_K:
            y = torch.zeros(y_shape, device=q.device, dtype=torch.float32)
        else:
            y = torch.empty(y_shape, device=q.device, dtype=q.dtype)

        _m2rnn_forward_triton(
            q=q,
            k=k,
            v=v,
            W=W,
            xf=xf,
            h0=h0,
            h=None,
            ht=ht,
            y=y,
            cu_seqlens=cu_seqlens,
            Nq=Nq,
            Nk=Nk,
            Nv=Nv,
            Nw=Nw,
            Nxf=Nxf,
            N=N,
        )

        y = y.type_as(v)

        ctx_save_for_backward(ctx, q, k, v, W, xf, h0, cu_seqlens)
        ctx.gradient_clipping = gradient_clipping
        ctx.num_heads = Nq, Nk, Nv, Nw, Nxf, N

        return y, ht

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dht: torch.Tensor) -> tuple[torch.Tensor | None]:
        q, k, v, W, xf, h0, cu_seqlens = ctx.saved_tensors
        Nq, Nk, Nv, Nw, Nxf, N = ctx.num_heads

        V = v.size(-1)

        if cu_seqlens is None:
            B, S, _, K = q.size()
            h = torch.empty(B, S, N, K, V, dtype=q.dtype, device=q.device)
        else:
            T, _, K = q.size()
            h = torch.empty(T, N, K, V, dtype=q.dtype, device=q.device)

        _m2rnn_forward_triton(
            q=None,
            k=k,
            v=v,
            W=W,
            xf=xf,
            h0=h0,
            h=h,
            ht=None,
            y=None,
            cu_seqlens=cu_seqlens,
            Nq=Nq,
            Nk=Nk,
            Nv=Nv,
            Nw=Nw,
            Nxf=Nxf,
            N=N,
        )

        function = partial(zeros_like_contiguous, dtype=torch.float32)

        dq = (empty_like_contiguous if Nq == N else function)(q)
        dk = (empty_like_contiguous if Nk == N else function)(k)
        dW = zeros_like_contiguous(W, dtype=torch.float32)
        dh0 = empty_like_contiguous(h0) if h0 is not None and h0.requires_grad else None

        if K > _MAX_BLOCK_SIZE_K:
            dv = function(v)
            dxf = function(xf)
        else:
            dv = (empty_like_contiguous if Nv == N else function)(v)
            dxf = (empty_like_contiguous if Nxf == N else function)(xf)

        _m2rnn_backward_triton(
            q=q,
            k=k,
            v=v,
            W=W,
            xf=xf,
            h0=h0,
            dy=dy,
            dht=dht,
            h=h,
            dq=dq,
            dk=dk,
            dv=dv,
            dW=dW,
            dxf=dxf,
            dh0=dh0,
            cu_seqlens=cu_seqlens,
            gradient_clipping=ctx.gradient_clipping,
        )

        dq = dq.type_as(q)
        dk = dk.type_as(k)
        dv = dv.type_as(v)
        dW = dW.type_as(W)
        dxf = dxf.type_as(xf)

        return dq, dk, dv, dW, dxf, dh0, *[None] * 4


def m2rnn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    forget_input: torch.Tensor,
    input_state: torch.Tensor | None = None,
    gradient_clipping: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | int | None = None,
    *,
    kernel_backend: KernelBackend | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    computes M2RNN recurrence

    :param query: query tensor of shape (B, S, Nq, K) where Nq is the number of query heads and K is the key head
        dimension. Should have shape (T, Nq, K) and `cu_seqlens` should be passed.
    :type query: torch.Tensor
    :param key: key tensor of shape (B, S, Nk, K) where Nk is the number of key heads and K is the key head
        dimension. Should have shape (T, Nk, K) and `cu_seqlens` should be passed.
    :type key: torch.Tensor
    :param value: value tensor of shape (B, S, Nv, V) where Nv is the number of value heads and V is the value head
        dimension. Should have shape (T, Nv, V) and `cu_seqlens` should be passed.
    :type value: torch.Tensor
    :param weight: weight tensor of shape (Nw, V, V)
    :type weight: torch.Tensor
    :param forget_input: forget input tensor of shape (B, S, Nxf) where Nxf is the number of forget heads and H is the head
        dimension. Should have shape (T, Nxf) and `cu_seqlens` should be passed.
    :type forget_input: torch.Tensor
    :param input_state: starting state of shape (B, N, K, V), where N = max{Nq, Nk, Nv, Nxf, Nw}. None means starting state is
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
    :return: output tensor of shape (B, S, N, V) if `cu_seqlens` is None else (T, N, V) and output state of
        shape (B, N, K, V).
    :rtype: tuple[Tensor, Tensor]
    """

    if cu_seqlens is None:
        assert max_seqlen is None
        B, S, _, K = query.size()
    else:
        assert max_seqlen is not None
        assert cu_seqlens.dim() == 1

        B = cu_seqlens.size(0) - 1
        T, _, K = query.size()

    V = value.size(-1)

    Nq, Nk, Nv, Nw, Nxf, N = _get_num_heads(q=query, k=key, v=value, W=weight, xf=forget_input, run_check=True)

    if cu_seqlens is None:
        assert query.size() == (B, S, Nq, K)
        assert key.size() == (B, S, Nk, K)
        assert value.size() == (B, S, Nv, V)
        assert forget_input.size() == (B, S, Nxf)
    else:
        assert query.size() == (T, Nq, K)
        assert key.size() == (T, Nk, K)
        assert value.size() == (T, Nv, V)
        assert forget_input.size() == (T, Nxf)

    assert weight.size() == (Nw, V, V)

    if input_state is not None:
        assert input_state.size() == (B, N, K, V)

    if gradient_clipping is not None and gradient_clipping < 0:
        gradient_clipping = -gradient_clipping

    output, input_state = _M2RNN.run(
        q=query,
        k=key,
        v=value,
        W=weight,
        xf=forget_input,
        h0=input_state,
        gradient_clipping=gradient_clipping,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        kernel_backend=kernel_backend,
    )

    return output, input_state
