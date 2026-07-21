# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

from ....custom_op import xma_op
from ....layers_jax.linear_attention.pallas_implementation import (
    _linear_attention_backward_checkpoint_pallas as _linear_attention_backward_checkpoint_pallas_jit,
)
from ....layers_jax.linear_attention.pallas_implementation import (
    _linear_attention_backward_main_pallas as _linear_attention_backward_main_pallas_jit,
)


def _get_num_blocks_s(S: int, BLOCK_SIZE_S: int) -> int:
    return (S + BLOCK_SIZE_S - 1) // BLOCK_SIZE_S


def _checkpoint_output_shape_dtype_fn(
    k: torch.Tensor, v: torch.Tensor, h0: torch.Tensor, BLOCK_SIZE_S: int
) -> list[tuple[tuple[int, ...], torch.dtype]]:
    B, _, S, K = k.shape
    V = v.shape[-1]
    N = h0.shape[1]
    NUM_BLOCKS_S = _get_num_blocks_s(S, BLOCK_SIZE_S)

    return [((B, N, NUM_BLOCKS_S, K, V), torch.float32)]


def _checkpoint_fake_function(k: torch.Tensor, v: torch.Tensor, h0: torch.Tensor, BLOCK_SIZE_S: int) -> torch.Tensor:
    B, _, S, K = k.shape
    V = v.shape[-1]
    N = h0.shape[1]
    NUM_BLOCKS_S = _get_num_blocks_s(S, BLOCK_SIZE_S)

    return torch.empty(B, N, NUM_BLOCKS_S, K, V, dtype=torch.float32, device=k.device)


_CHECKPOINT_CACHE = None


@xma_op(mutates_args={}, fake_func=_checkpoint_fake_function)
def _linear_attention_backward_checkpoint_pallas(
    k: torch.Tensor, v: torch.Tensor, h0: torch.Tensor, BLOCK_SIZE_S: int
) -> torch.Tensor:
    # k, v: already transposed to (B, Nk/Nv, S, K/V); h0: (B, N, K, V), never None (defaulted by the caller)
    global _CHECKPOINT_CACHE

    if _CHECKPOINT_CACHE is None:
        _CHECKPOINT_CACHE = make_kernel_from_pallas(
            _linear_attention_backward_checkpoint_pallas_jit, _checkpoint_output_shape_dtype_fn
        )

    return _CHECKPOINT_CACHE(k, v, h0, BLOCK_SIZE_S, static_argnums=(3,))


def _main_output_shape_dtype_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dy: torch.Tensor,
    h_checkpoints: torch.Tensor,
    dh: torch.Tensor,
) -> list[tuple[tuple[int, ...], torch.dtype]]:
    B, Nq, S, K = q.shape
    Nk = k.shape[1]
    Nv, V = v.shape[1], v.shape[-1]
    N = dh.shape[1]

    return [
        ((B, S, Nq, K), q.dtype),
        ((B, S, Nk, K), k.dtype),
        ((B, S, Nv, V), v.dtype),
        ((B, N, K, V), torch.float32),
    ]


def _main_fake_function(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dy: torch.Tensor,
    h_checkpoints: torch.Tensor,
    dh: torch.Tensor,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, Nq, S, K = q.shape
    Nk = k.shape[1]
    Nv, V = v.shape[1], v.shape[-1]
    N = dh.shape[1]

    dq = torch.empty(B, S, Nq, K, dtype=q.dtype, device=q.device)
    dk = torch.empty(B, S, Nk, K, dtype=k.dtype, device=k.device)
    dv = torch.empty(B, S, Nv, V, dtype=v.dtype, device=v.device)
    dh0 = torch.empty(B, N, K, V, dtype=torch.float32, device=q.device)

    return dq, dk, dv, dh0


_MAIN_CACHE = None


@xma_op(mutates_args={}, fake_func=_main_fake_function)
def _linear_attention_backward_main_pallas(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dy: torch.Tensor,
    h_checkpoints: torch.Tensor,
    dh: torch.Tensor,
    attention_multiplier: float,
    BLOCK_SIZE_S: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # q, k, v, dy: already transposed to (B, Nq/Nk/Nv/N, S, K/V); dh: (B, N, K, V), never None
    global _MAIN_CACHE

    if _MAIN_CACHE is None:
        _MAIN_CACHE = make_kernel_from_pallas(_linear_attention_backward_main_pallas_jit, _main_output_shape_dtype_fn)

    return _MAIN_CACHE(
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
    Nk = k.shape[-2]
    Nv, V = v.shape[-2:]

    N = max(Nq, Nk, Nv)

    if h0 is None:
        h0 = torch.zeros(B, N, K, V, dtype=torch.float32, device=q.device)

    if dh is None:
        dh = torch.zeros(B, N, K, V, dtype=torch.float32, device=q.device)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    dy = dy.transpose(1, 2)

    h_checkpoints = _linear_attention_backward_checkpoint_pallas(k, v, h0, BLOCK_SIZE_S)

    return _linear_attention_backward_main_pallas(
        q, k, v, dy, h_checkpoints, dh, attention_multiplier=attention_multiplier, BLOCK_SIZE_S=BLOCK_SIZE_S
    )
