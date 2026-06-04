# **************************************************
# Copyright (c) 2026, Mayank Mishra, Han Guo
# **************************************************

# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu as fla_chunk_gated_delta_rule_bwd_dhu
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as fla_chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.cp import FLACPContext
from fla.ops.cp.chunk_delta_h import compress_h0, expand_h0
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra as fla_chunk_gated_delta_rule_fwd_intra
from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd

from .chunk_delta_h_cp import chunk_gated_delta_rule_bwd_dhu_pre_process, chunk_gated_delta_rule_fwd_h_pre_process
from .wy_fast import prepare_wy_repr_bwd as fla_prepare_wy_repr_bwd


def chunk_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    chunk_indices: torch.LongTensor | None = None,
    transpose_state_layout: bool = False,
):
    assert q.shape[2] == k.shape[2] == v.shape[2] == beta.shape[2]
    if cp_context is not None:
        # i.e., batch size = 1
        assert cu_seqlens is not None
        assert cu_seqlens.ndim == 1
        assert cu_seqlens.shape[0] == 2

    # obtain WY representation. u is actually the new v.
    # fused kkt + solve_tril + recompute_w_u
    w, u, A = fla_chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=None,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    if cp_context is not None:
        if initial_state is not None:
            assert initial_state.ndim == 4
            assert initial_state.shape[0] == 1
            initial_state = initial_state.squeeze(dim=0)
        initial_state_cp = chunk_gated_delta_rule_fwd_h_pre_process(
            k=k,
            v=v,
            w=w,
            u=u,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            context=cp_context,
            transpose_state_layout=transpose_state_layout,
        )
    else:
        initial_state_cp = initial_state

    h, v_new, final_state = fla_chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=None,
        initial_state=initial_state_cp,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        transpose_state_layout=transpose_state_layout,
    )

    if cp_context is not None:
        initial_state_cp = compress_h0(initial_state_cp, context=cp_context)

    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=None,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        transpose_state_layout=transpose_state_layout,
    )
    return o, A, final_state, initial_state_cp


def chunk_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
    chunk_indices: torch.LongTensor | None = None,
    transpose_state_layout: bool = False,
):
    assert q.shape[2] == k.shape[2] == v.shape[2] == beta.shape[2]
    if cp_context is not None:
        # i.e., batch size = 1
        assert cu_seqlens is not None
        assert cu_seqlens.ndim == 1
        assert cu_seqlens.shape[0] == 2

    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    if cp_context is not None:
        initial_state = expand_h0(initial_state, context=cp_context)

    h, v_new, _ = fla_chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=None,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        transpose_state_layout=transpose_state_layout,
    )
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=None,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    if cp_context is not None:
        if dht is not None:
            assert dht.ndim == 4
            assert dht.shape[0] == 1
            dht = dht.squeeze(dim=0)
        dht, initial_state = chunk_gated_delta_rule_bwd_dhu_pre_process(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            scale=scale,
            cu_seqlens=cu_seqlens,
            dht=dht,
            initial_state=initial_state,
            context=cp_context,
            transpose_state_layout=transpose_state_layout,
        )

    dh, dh0, dv = fla_chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=None,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        transpose_state_layout=transpose_state_layout,
    )
    dq, dk, dw, _ = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=None,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        transpose_state_layout=transpose_state_layout,
    )
    dk2, dv, db, _ = fla_prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=None,
        A=A,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    dk.add_(dk2)

    # Under CP, only rank 0's dh0_local is the true ∂L/∂h0_user (the bwd pre-process
    # merge already propagated later ranks' contributions back to rank 0 via dht
    # chaining). Non-first ranks' dh0_local is ∂L/∂(intermediate state), not w.r.t.
    # h0 — zero it out so DDP all-reduce-sum produces the correct gradient.
    if cp_context is not None:
        if (dh0 is not None) and (not cp_context.is_first_rank):
            assert dht is not None
            assert initial_state is not None
            dh0 = torch.zeros_like(dh0)

    return dq, dk, dv, db, dh0
