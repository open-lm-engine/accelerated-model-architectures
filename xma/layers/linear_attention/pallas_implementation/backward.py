# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

from ....custom_op import xma_op
from ....layers_jax.linear_attention.pallas_implementation.backward import (
    _linear_attention_backward_core as _backward_core_jax,
)
from ....layers_jax.linear_attention.pallas_implementation.backward import (
    _linear_attention_checkpoint_core as _checkpoint_core_jax,
)
from ....math import ceil_divide


def _checkpoint_output_shape_dtype_fn(
    k: torch.Tensor, v: torch.Tensor, h0: torch.Tensor, BLOCK_SIZE_S: int
) -> list[tuple[tuple[int, ...], torch.dtype]]:
    B, _, S, K = k.shape
    V = v.shape[-1]
    N = h0.shape[1]
    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)

    # NUM_BLOCKS_S is folded into the N axis (rather than kept as its own axis), matching the jax-side kernel, to
    # keep this rank 4. The underlying pallas_call always produces both the checkpoints and the final running
    # state, even though the jax-side wrapper only returns the checkpoints; both outputs must be declared here to
    # match its window_params
    return [((B, N * NUM_BLOCKS_S, K, V), torch.float32), ((B, N, K, V), torch.float32)]


def _checkpoint_fake_function(k: torch.Tensor, v: torch.Tensor, h0: torch.Tensor, BLOCK_SIZE_S: int) -> torch.Tensor:
    B, _, S, K = k.shape
    V = v.shape[-1]
    N = h0.shape[1]
    NUM_BLOCKS_S = ceil_divide(S, BLOCK_SIZE_S)

    return torch.empty(B, N * NUM_BLOCKS_S, K, V, dtype=torch.float32, device=k.device)


_CHECKPOINT_CACHE = None


@xma_op(mutates_args={}, fake_func=_checkpoint_fake_function)
def _linear_attention_checkpoint_core(
    k: torch.Tensor, v: torch.Tensor, h0: torch.Tensor, BLOCK_SIZE_S: int
) -> torch.Tensor:
    # k, v: already transposed to (B, Nk/Nv, S, K/V); h0: (B, N, K, V), never None (defaulted by the caller)
    global _CHECKPOINT_CACHE

    if _CHECKPOINT_CACHE is None:
        _CHECKPOINT_CACHE = make_kernel_from_pallas(_checkpoint_core_jax, _checkpoint_output_shape_dtype_fn)

    h_checkpoints, _ = _CHECKPOINT_CACHE(k, v, h0, BLOCK_SIZE_S, static_argnums=(3,))
    return h_checkpoints


def _backward_output_shape_dtype_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dy: torch.Tensor,
    h_checkpoints: torch.Tensor,
    dh: torch.Tensor,
) -> list[tuple[tuple[int, ...], torch.dtype]]:
    B, _, S, K = q.shape
    V = v.shape[-1]
    N = dh.shape[1]

    return [
        ((B, N, S, K), q.dtype),
        ((B, N, S, K), q.dtype),
        ((B, N, S, V), q.dtype),
        ((B, N, K, V), torch.float32),
    ]


def _backward_fake_function(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dy: torch.Tensor,
    h_checkpoints: torch.Tensor,
    dh: torch.Tensor,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, _, S, K = q.shape
    V = v.shape[-1]
    N = dh.shape[1]

    dq = torch.empty(B, N, S, K, dtype=q.dtype, device=q.device)
    dk = torch.empty(B, N, S, K, dtype=q.dtype, device=q.device)
    dv = torch.empty(B, N, S, V, dtype=q.dtype, device=q.device)
    dh0 = torch.empty(B, N, K, V, dtype=torch.float32, device=q.device)

    return dq, dk, dv, dh0


_BACKWARD_CACHE = None


@xma_op(mutates_args={}, fake_func=_backward_fake_function)
def _linear_attention_backward_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dy: torch.Tensor,
    h_checkpoints: torch.Tensor,
    dh: torch.Tensor,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # q, k, v, dy: already transposed to (B, N, S, K/V); dh: (B, N, K, V), never None
    global _BACKWARD_CACHE

    if _BACKWARD_CACHE is None:
        _BACKWARD_CACHE = make_kernel_from_pallas(_backward_core_jax, _backward_output_shape_dtype_fn)

    return _BACKWARD_CACHE(
        q,
        k,
        v,
        dy,
        h_checkpoints,
        dh,
        static_argnames=("attention_multiplier", "BLOCK_SIZE_S"),
        attention_multiplier=attention_multiplier,
        BLOCK_SIZE_S=BLOCK_SIZE_S,
    )


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
    B, S, Nq, K = q.shape
    Nk = k.size(-2)
    Nv, V = v.size()[-2:]

    N = max(Nq, Nk, Nv)

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv

    if h0 is None:
        h0 = torch.zeros(B, N, K, V, dtype=torch.float32, device=q.device)
    else:
        h0 = h0.float()

    if dh is None:
        dh = torch.zeros(B, N, K, V, dtype=torch.float32, device=q.device)
    else:
        dh = dh.float()

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    dy = dy.transpose(1, 2)

    h_checkpoints = _linear_attention_checkpoint_core(k, v, h0, BLOCK_SIZE_S)

    dq, dk, dv, dh0 = _linear_attention_backward_core(
        q, k, v, dy, h_checkpoints, dh, attention_multiplier=attention_multiplier, BLOCK_SIZE_S=BLOCK_SIZE_S
    )

    dq = dq.transpose(1, 2)
    dk = dk.transpose(1, 2)
    dv = dv.transpose(1, 2)

    dq = dq.reshape(B, S, Nq, Gq, K).sum(dim=3)
    dk = dk.reshape(B, S, Nk, Gk, K).sum(dim=3)
    dv = dv.reshape(B, S, Nv, Gv, V).sum(dim=3)

    return dq, dk, dv, dh0
