# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import math

import torch

from ...accelerator import KernelBackend
from ...custom_op import CustomOp, ctx_needs_gradients
from ...math import ceil_divide
from ...utils import is_torch_xla_available, is_triton_available
from .utils import _get_num_heads


if is_triton_available():
    from .triton_implementation import _linear_attention_forward_triton

if is_torch_xla_available():
    from .pallas_implementation import _linear_attention_backward_pallas, _linear_attention_forward_pallas


class _LinearAttention(CustomOp):
    @staticmethod
    def forward_backward_torch(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h0: torch.Tensor | None,
        attention_multiplier: float,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
        CHUNK_SIZE: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Nq, Nk, Nv, N = _get_num_heads(q=q, k=k, v=v, run_check=False)

        y_shape = list(v.size())
        y_shape[-2] = N
        y = torch.empty(y_shape, device=q.device, dtype=q.dtype)

        if cu_seqlens is None:
            B, S, _, K = q.size()
        else:
            raise NotImplementedError

        V = v.size(-1)

        Gq = N // Nq
        Gk = N // Nk
        Gv = N // Nv

        q = q.repeat_interleave(Gq, dim=-2)
        k = k.repeat_interleave(Gk, dim=-2)
        v = v.repeat_interleave(Gv, dim=-2)

        h0 = torch.zeros(B, N, K, V, dtype=torch.float32, device=q.device) if h0 is None else h0.float()

        for s in range(S):
            y[:, s] = (q[:, s, :, None, :] @ h0.type_as(q)).squeeze(-2)
            h0 = h0 + k[:, s, ..., None] * v[:, s, :, None, :]

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
        max_seqlen: int | None,
        CHUNK_SIZE: int,
        kernel_backend: KernelBackend | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert kernel_backend in [KernelBackend.cuda, KernelBackend.triton, KernelBackend.pallas]

        ctx.kernel_backend = kernel_backend

        if kernel_backend == KernelBackend.pallas:
            assert cu_seqlens is None

            y, ht = _linear_attention_forward_pallas(
                q=q, k=k, v=v, h0=h0, attention_multiplier=attention_multiplier, BLOCK_SIZE_S=CHUNK_SIZE
            )

            ctx.h0_is_none = h0 is None
            ctx.attention_multiplier = attention_multiplier
            ctx.CHUNK_SIZE = CHUNK_SIZE
            ctx.save_for_backward(*((q, k, v) if h0 is None else (q, k, v, h0)))

            return y, ht

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

        _linear_attention_forward_triton(
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
        )

        return y, ht

    @staticmethod
    def backward(
        ctx, dy: torch.Tensor, dht: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, None, None, None, None, None]:
        if ctx.kernel_backend != KernelBackend.pallas:
            raise NotImplementedError(f"backward is not implemented for kernel_backend ({ctx.kernel_backend})")

        if ctx.h0_is_none:
            q, k, v = ctx.saved_tensors
            h0 = None
        else:
            q, k, v, h0 = ctx.saved_tensors

        dq, dk, dv, dh0 = _linear_attention_backward_pallas(
            q=q,
            k=k,
            v=v,
            dy=dy,
            h0=h0,
            dh=dht,
            attention_multiplier=ctx.attention_multiplier,
            BLOCK_SIZE_S=ctx.CHUNK_SIZE,
        )

        return dq, dk, dv, (dh0 if h0 is not None else None), None, None, None, None, None


def linear_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    input_state: torch.Tensor | None,
    attention_multiplier: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    CHUNK_SIZE: int = 64,
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
        kernel_backend=kernel_backend,
    )

    return output, input_state
