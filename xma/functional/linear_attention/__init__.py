# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...accelerator import KernelBackend
from ...custom_op import CustomOp
from ...math import ceil_divide
from .triton_implementation import linear_attention_forward_triton
from .utils import _get_num_heads


class _LinearAttention(CustomOp):
    @staticmethod
    def forward_backward_torch(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h0: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: torch.Tensor | int | None,
        CHUNK_SIZE: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Nq, Nk, Nv, N = _get_num_heads(q=q, k=k, v=v, run_check=False)

        y_shape = list(v.size())
        y_shape[-2] = N
        y = torch.empty(y_shape, device=q.device, dtype=q.dtype)

        if cu_seqlens is None:
            B, S, _, K = q.size()
        else:
            B = cu_seqlens.size(0) - 1
            S = max_seqlen.item() if isinstance(max_seqlen, torch.Tensor) else max_seqlen
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
                h = h0 + k[:, s, ..., None] * v[:, s, :, None, :]
                y[:, s] = (q[:, s, :, None, :] @ h.type_as(q)).squeeze(-2)

                h0 = h
            else:
                offset = start + s
                unfinished = offset < end
                offset_unfinished = offset[unfinished]

                h = h0[unfinished] + k[offset_unfinished, ..., None] * v[offset_unfinished, :, None, :]
                y[offset_unfinished] = (q[offset_unfinished, :, None, :] @ h.type_as(q)).squeeze(-2)

                h0[unfinished] = h

        return y, h0

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h0: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: torch.Tensor | int | None,
        CHUNK_SIZE: int,
        kernel_backend: KernelBackend | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert kernel_backend in [KernelBackend.cuda, KernelBackend.triton]

        Nq, Nk, Nv, N = _get_num_heads(q=q, k=k, v=v, run_check=False)

        B, S, _, K = k.size()
        V = v.size(-1)
        NUM_CHUNKS = ceil_divide(S, CHUNK_SIZE)

        y = torch.empty(B, S, N, V)
        h = torch.empty(B, NUM_CHUNKS, N, K, V, dtype=k.dtype, device=k.device)
        ht = torch.empty(B, N, K, V, dtype=torch.float32, device=k.device)

        linear_attention_forward_triton(q=q, k=k, v=v, h0=h0, h=h, ht=ht, cu_seqlens=cu_seqlens, CHUNK_SIZE=CHUNK_SIZE)

        return y, ht


def linear_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    input_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | int | None = None,
    CHUNK_SIZE: int = 64,
    *,
    kernel_backend: KernelBackend | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_dim = 3 + (cu_seqlens is None)

    assert query.dim() == expected_dim
    assert key.dim() == expected_dim
    assert value.dim() == expected_dim

    if cu_seqlens is None:
        assert max_seqlen is None
        B, S, _, K = query.size()
    else:
        assert max_seqlen is not None
        assert cu_seqlens.dim() == 1

        T, _, K = query.size()
        B = cu_seqlens.size(0) - 1

    V = value.size(-1)
    Nq, _, Nv, N = _get_num_heads(q=query, k=key, v=value, run_check=True)

    if cu_seqlens is None:
        assert query.size() == (B, S, Nq, K)
        assert value.size() == (B, S, Nv, V)
    else:
        assert query.size() == (T, Nq, K)
        assert value.size() == (T, Nv, V)

    if input_state is not None:
        assert input_state.size() == (B, N, K, V)

    output, input_state = _LinearAttention.run(
        q=query,
        k=key,
        v=value,
        h0=input_state,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        CHUNK_SIZE=CHUNK_SIZE,
        kernel_backend=kernel_backend,
    )

    return output, input_state
