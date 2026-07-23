# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

from ....custom_op import xma_op
from ....layers_jax.linear_attention.pallas_implementation.forward import (
    _linear_attention_forward_core as _forward_core_jax,
)


def _output_shape_dtype_fn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h0: torch.Tensor
) -> list[tuple[tuple[int, ...], torch.dtype]]:
    B, _, S, K = q.shape
    V = v.size(-1)
    N = h0.size(1)

    return [((B, N, S, V), q.dtype), ((B, N, K, V), torch.float32)]


def _fake_function(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h0: torch.Tensor,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, Nq, S, K = q.shape
    V = v.shape[-1]
    N = h0.shape[1]

    y = torch.empty(B, N, S, V, dtype=q.dtype, device=q.device)
    h = torch.empty(B, N, K, V, dtype=torch.float32, device=q.device)

    return y, h


_CACHE = None


@xma_op(mutates_args={}, fake_func=_fake_function)
def _linear_attention_forward_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h0: torch.Tensor,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    global _CACHE

    if _CACHE is None:
        _CACHE = make_kernel_from_pallas(_forward_core_jax, _output_shape_dtype_fn)

    return _CACHE(
        q,
        k,
        v,
        h0,
        static_argnames=("attention_multiplier", "BLOCK_SIZE_S"),
        attention_multiplier=attention_multiplier,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
    )


def _linear_attention_forward_pallas(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h0: torch.Tensor | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, S, Nq, K = q.shape
    Nk = k.shape[-2]
    Nv, V = v.shape[-2:]

    N = max(Nq, Nk, Nv)

    if h0 is None:
        h0 = torch.zeros(B, N, K, V, dtype=torch.float32, device=q.device)
    else:
        h0 = h0.float()

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    y, h = _linear_attention_forward_core(
        q, k, v, h0, attention_multiplier=attention_multiplier, BLOCK_SIZE_S=BLOCK_SIZE_S
    )
    y = y.transpose(1, 2)

    return y, h
