# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

from ....custom_op import xma_op
from ....layers_jax.linear_attention.pallas_implementation import (
    _linear_attention_forward_pallas as _linear_attention_forward_pallas_jit,
)


def _get_output_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[int, int, int, int, int]:
    B, S, Nq, K = q.shape
    Nk = k.shape[-2]
    Nv, V = v.shape[-2:]
    N = max(Nq, Nk, Nv)

    return B, S, K, V, N


def _output_shape_dtype_fn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h0: torch.Tensor | None
) -> list[tuple[tuple[int, ...], torch.dtype]]:
    B, S, K, V, N = _get_output_shapes(q, k, v)
    return [((B, S, N, V), q.dtype), ((B, N, K, V), torch.float32)]


def _fake_function(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h0: torch.Tensor | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, S, K, V, N = _get_output_shapes(q, k, v)

    y = torch.empty(B, S, N, V, dtype=q.dtype, device=q.device)
    h = torch.empty(B, N, K, V, dtype=torch.float32, device=q.device)

    return y, h


_CACHE = None


@xma_op(mutates_args={}, fake_func=_fake_function)
def _linear_attention_forward_pallas(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h0: torch.Tensor | None,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    global _CACHE

    if _CACHE is None:
        _CACHE = make_kernel_from_pallas(_linear_attention_forward_pallas_jit, _output_shape_dtype_fn)

    if h0 is None:
        # materialize so trace_pallas doesn't drop it from tensor_args, causing an operand-count mismatch
        B, S, K, V, N = _get_output_shapes(q, k, v)
        h0 = torch.zeros(B, N, K, V, dtype=torch.float32, device=q.device)

    return _CACHE(
        q,
        k,
        v,
        h0,
        static_argnames=("attention_multiplier", "BLOCK_SIZE_S"),
        attention_multiplier=attention_multiplier,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
    )
