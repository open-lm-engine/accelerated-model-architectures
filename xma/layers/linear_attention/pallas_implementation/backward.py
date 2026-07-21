# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

from ....custom_op import xma_op
from ....layers_jax.linear_attention.pallas_implementation import (
    _linear_attention_backward_pallas as _linear_attention_backward_pallas_jit,
)


def _get_output_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[int, int, int, int, int, int, int]:
    B, S, Nq, K = q.shape
    Nk = k.shape[-2]
    Nv, V = v.shape[-2:]

    return B, S, Nq, Nk, Nv, K, V


def _output_shape_dtype_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dy: torch.Tensor,
    h0: torch.Tensor | None,
    dh: torch.Tensor | None,
) -> list[tuple[tuple[int, ...], torch.dtype]]:
    B, S, Nq, Nk, Nv, K, V = _get_output_shapes(q, k, v)
    N = max(Nq, Nk, Nv)

    return [
        ((B, S, Nq, K), q.dtype),
        ((B, S, Nk, K), k.dtype),
        ((B, S, Nv, V), v.dtype),
        ((B, N, K, V), torch.float32),
    ]


def _fake_function(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dy: torch.Tensor,
    h0: torch.Tensor | None,
    dh: torch.Tensor | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, S, Nq, Nk, Nv, K, V = _get_output_shapes(q, k, v)
    N = max(Nq, Nk, Nv)

    dq = torch.empty(B, S, Nq, K, dtype=q.dtype, device=q.device)
    dk = torch.empty(B, S, Nk, K, dtype=k.dtype, device=k.device)
    dv = torch.empty(B, S, Nv, V, dtype=v.dtype, device=v.device)
    dh0 = torch.empty(B, N, K, V, dtype=torch.float32, device=q.device)

    return dq, dk, dv, dh0


_CACHE = None


@xma_op(mutates_args={}, fake_func=_fake_function)
def _linear_attention_backward_pallas(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dy: torch.Tensor,
    h0: torch.Tensor | None,
    dh: torch.Tensor | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    global _CACHE

    if _CACHE is None:
        _CACHE = make_kernel_from_pallas(_linear_attention_backward_pallas_jit, _output_shape_dtype_fn)

    return _CACHE(
        q,
        k,
        v,
        dy,
        h0,
        dh,
        static_argnames=("attention_multiplier", "BLOCK_SIZE_S"),
        attention_multiplier=attention_multiplier,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
    )
