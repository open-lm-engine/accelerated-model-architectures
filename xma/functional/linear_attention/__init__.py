# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import math

import torch

from ...accelerator import KernelBackend
from ...custom_op import CustomOp, ctx_needs_gradients, ctx_save_for_backward
from ...math import ceil_divide
from ...utils import empty_like_contiguous
from .triton_implementation import autotuned_linear_attention_forward_triton, recurrent_state_forward_triton
from .utils import _get_num_heads


class _LinearAttention(CustomOp):
    @staticmethod
    def forward_backward_torch(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h0: torch.Tensor | None,
        attention_multiplier: float,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: torch.Tensor | int | None,
        CHUNK_SIZE: int,
        use_fused_kernel_in_forward: bool | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert use_fused_kernel_in_forward is None
        Nq, Nk, Nv, N = _get_num_heads(q=q, k=k, v=v, run_check=False)

        y_shape = list(v.size())
        y_shape[-2] = N
        y = torch.empty(y_shape, device=q.device, dtype=q.dtype)

        if cu_seqlens is None:
            B, S, _, K = q.size()
        else:
            B = cu_seqlens.size(0) - 1
            S = max_seqlen
            K = q.size(-1)

        V = v.size(-1)

        Gq = N // Nq
        Gk = N // Nk
        Gv = N // Nv

        q = q.repeat_interleave(Gq, dim=-2)
        k = k.repeat_interleave(Gk, dim=-2)
        v = v.repeat_interleave(Gv, dim=-2)

        h0 = torch.zeros(B, N, K, V, dtype=torch.float32, device=q.device) if h0 is None else h0.float()

        if cu_seqlens is not None:
            h0 = h0.clone()
            start = cu_seqlens[:-1]
            end = cu_seqlens[1:]

        for s in range(S):
            if cu_seqlens is None:
                y[:, s] = (q[:, s, :, None, :] @ h0.type_as(q)).squeeze(-2)
                h0 = h0 + k[:, s, ..., None] * v[:, s, :, None, :]
            else:
                offset = start + s
                unfinished = offset < end
                offset_unfinished = offset[unfinished]

                y[offset_unfinished] = (q[offset_unfinished, :, None, :] @ h0[unfinished].type_as(q)).squeeze(-2)
                h0[unfinished] = h0[unfinished] + k[offset_unfinished, ..., None] * v[offset_unfinished, :, None, :]

        y = y * attention_multiplier

        return y, h0

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h0: torch.Tensor | None,
        attention_multiplier: float,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: torch.Tensor | int | None,
        CHUNK_SIZE: int,
        use_fused_kernel_in_forward: bool | None,
        kernel_backend: KernelBackend | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert kernel_backend in [KernelBackend.cuda, KernelBackend.triton]

        Nq, Nk, Nv, N = _get_num_heads(q=q, k=k, v=v, run_check=False)

        B, S, _, K = k.size()
        V = v.size(-1)

        y = torch.empty(B, S, N, V, dtype=k.dtype, device=k.device)
        ht = torch.empty(B, N, K, V, dtype=torch.float32, device=k.device)

        NUM_CHUNKS = ceil_divide(S, CHUNK_SIZE)

        h = (
            torch.empty(B, NUM_CHUNKS - 1, N, K, V, dtype=k.dtype, device=k.device)
            if ctx_needs_gradients(ctx)
            else None
        )

        autotuned_linear_attention_forward_triton(
            q=q,
            k=k,
            v=v,
            h0=h0,
            h=h,
            ht=ht,
            y=y,
            attention_multiplier=attention_multiplier,
            cu_seqlens=cu_seqlens,
            CHUNK_SIZE=CHUNK_SIZE,
            use_fused_kernel_in_forward=use_fused_kernel_in_forward,
        )

        ctx_save_for_backward(ctx, q, k, v, h0, cu_seqlens)

        ctx.CHUNK_SIZE = CHUNK_SIZE
        ctx.num_heads = Nq, Nk, Nv, N
        ctx.attention_multiplier = attention_multiplier

        return y, ht

    @staticmethod
    def backward(
        ctx, dy: torch.Tensor, dht: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, None, None, None, None, None, None]:
        q, k, v, h0, cu_seqlens = ctx.saved_tensors

        Nq, Nk, Nv, N = ctx.num_heads
        CHUNK_SIZE = ctx.CHUNK_SIZE
        attention_multiplier = ctx.attention_multiplier

        B, S, _, K = k.size()
        V = v.size(-1)
        NUM_CHUNKS = ceil_divide(S, CHUNK_SIZE)

        h = (
            torch.empty(B, NUM_CHUNKS - 1, N, K, V, dtype=k.dtype, device=k.device)
            if ctx_needs_gradients(ctx)
            else None
        )

        recurrent_state_forward_triton(
            q=None,
            k=k,
            v=v,
            h0=h0,
            h=h,
            ht=None,
            y=None,
            attention_multiplier=attention_multiplier,
            cu_seqlens=cu_seqlens,
            CHUNK_SIZE=CHUNK_SIZE,
        )

        dq_triton()

        dq = empty_like_contiguous(q)
        dk = empty_like_contiguous(k)
        dv = empty_like_contiguous(v)
        dh0 = empty_like_contiguous(h0) if h0 is not None and h0.requires_grad else None

        return dq, dk, dv, dh0, *[None] * 5


def linear_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    input_state: torch.Tensor | None,
    attention_multiplier: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | int | None = None,
    CHUNK_SIZE: int = 64,
    use_fused_kernel_in_forward: bool | None = None,
    *,
    kernel_backend: KernelBackend | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cu_seqlens is None:
        assert max_seqlen is None
        B, S, _, K = query.size()
    else:
        assert max_seqlen is not None
        assert cu_seqlens.dim() == 1

        T, _, K = query.size()
        B = cu_seqlens.size(0) - 1

    V = value.size(-1)
    Nq, Nk, Nv, N = _get_num_heads(q=query, k=key, v=value, run_check=True)

    if cu_seqlens is None:
        assert query.size() == (B, S, Nq, K)
        assert key.size() == (B, S, Nk, K)
        assert value.size() == (B, S, Nv, V)
    else:
        assert query.size() == (T, Nq, K)
        assert key.size() == (T, Nk, K)
        assert value.size() == (T, Nv, V)

    if input_state is not None:
        assert input_state.size() == (B, N, K, V)

    if attention_multiplier is None:
        attention_multiplier = 1 / math.sqrt(K)

    output, input_state = _LinearAttention.run(
        q=query,
        k=key,
        v=value,
        h0=input_state,
        attention_multiplier=attention_multiplier,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        CHUNK_SIZE=CHUNK_SIZE,
        use_fused_kernel_in_forward=use_fused_kernel_in_forward,
        kernel_backend=kernel_backend,
    )

    return output, input_state
