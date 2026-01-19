# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ....custom_op import xma_op
from ....math import ceil_divide
from ....xtuner import XTuneConfig, XTuneParameter, xtune
from ..utils import _get_num_heads
from .output_forward import output_forward_triton_kernel
from .recurrent_state_forward import recurrent_state_forward_triton_kernel


@xtune(
    configs=[XTuneConfig({"use_fused_kernel_in_forward": i}) for i in [True, False]],
    functional_triggers={
        "_": lambda **kwargs: (kwargs["q"].size(1) if kwargs["cu_seqlens"] is None else kwargs["max_seqlen"]) <= 64
    },
)
def _autotuned_linear_attention_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h0: torch.Tensor | None,
    h: torch.Tensor,
    ht: torch.Tensor,
    y: torch.Tensor,
    attention_multiplier: float,
    cu_seqlens: torch.Tensor | None,
    CHUNK_SIZE: int,
    use_fused_kernel_in_forward: XTuneParameter[bool],
) -> None:
    Nq, Nk, Nv, N = _get_num_heads(q=q, k=k, v=v, run_check=False)

    if cu_seqlens is None:
        B, S, _, K = k.size()
    else:
        B = cu_seqlens.size(0) - 1
        S = None
        K = k.size(-1)

    V = v.size(-1)

    kwargs = {
        "k_ptr": k,
        "k_stride": k.stride(),
        "v_ptr": v,
        "v_stride": v.stride(),
        "h0_ptr": h0,
        "h0_stride": None if h0 is None else h0.stride(),
        "h_ptr": h,
        "h_stride": None if h is None else h.stride(),
        "attention_multiplier": attention_multiplier,
        "cu_seqlens_ptr": cu_seqlens,
        "cu_seqlens_stride": None if cu_seqlens is None else cu_seqlens.stride(),
        "S": S,
        "N": N,
        "K": K,
        "V": V,
        "Gq": N // Nq,
        "Gk": N // Nk,
        "Gv": N // Nv,
    }

    GRID = lambda kwargs: (B * N, ceil_divide(K, kwargs["BLOCK_SIZE_K"]), ceil_divide(V, kwargs["BLOCK_SIZE_V"]))

    recurrent_state_forward_triton_kernel[GRID](
        q_ptr=q if use_fused_kernel_in_forward else None,
        q_stride=q.stride() if use_fused_kernel_in_forward else None,
        ht_ptr=ht,
        ht_stride=ht.stride(),
        y_ptr=y if use_fused_kernel_in_forward else None,
        y_stride=y.stride() if use_fused_kernel_in_forward else None,
        CHUNK_SIZE=CHUNK_SIZE,
        **kwargs,
    )

    if not use_fused_kernel_in_forward:
        NUM_CHUNKS = h.size(1)
        GRID = lambda kwargs: (B * N, NUM_CHUNKS + 1, ceil_divide(V, kwargs["BLOCK_SIZE_V"]))

        output_forward_triton_kernel[GRID](
            q_ptr=q, q_stride=q.stride(), y_ptr=y, y_stride=y.stride(), BLOCK_SIZE_S=CHUNK_SIZE, **kwargs
        )


@xma_op(mutates_args={"y", "h", "ht"})
def linear_attention_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h0: torch.Tensor | None,
    h: torch.Tensor,
    ht: torch.Tensor,
    y: torch.Tensor,
    attention_multiplier: float,
    cu_seqlens: torch.Tensor | None,
    CHUNK_SIZE: int,
    use_fused_kernel_in_forward: bool | None,
) -> None:
    if use_fused_kernel_in_forward is None:
        use_fused_kernel_in_forward = XTuneParameter()

    _autotuned_linear_attention_forward_triton(
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
