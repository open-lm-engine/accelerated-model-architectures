# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import get_next_power_of_2
from .forward import _forward_single_step, _get_autotune_configs


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_K", "BLOCK_SIZE_V"])
@triton.jit
def rsa_chunked_forward_triton_kernel(
    k_ptr,
    k_stride,
    v_ptr,
    v_stride,
    W_ptr,
    W_stride,
    xf_ptr,
    xf_stride,
    h0_ptr,
    h0_stride,
    hc_ptr,
    hc_stride,
    h_ptr,
    h_stride,
    S,
    K: tl.constexpr,
    V: tl.constexpr,
    Gk: tl.constexpr,
    Gv: tl.constexpr,
    Gw: tl.constexpr,
    Gxf: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(0)
    BLOCK_ID_N = tl.program_id(1)
    BLOCK_ID_CHUNK = tl.program_id(2)

    BLOCK_ID_Nk = BLOCK_ID_N // Gk
    BLOCK_ID_Nv = BLOCK_ID_N // Gv

    BLOCK_ID_Nw = BLOCK_ID_N // Gw
    BLOCK_ID_Nxf = BLOCK_ID_N // Gxf

    BLOCK_K = tl.arange(0, BLOCK_SIZE_K)
    BLOCK_V = tl.arange(0, BLOCK_SIZE_V)

    MASK_K = BLOCK_K < K
    MASK_V = BLOCK_V < V

    MASK_KV = MASK_K[:, None] & MASK_V[None, :]
    MASK_VV = MASK_V[:, None] & MASK_V[None, :]

    W = tl.load(
        W_ptr + BLOCK_ID_Nw * W_stride[0] + BLOCK_V[:, None] * W_stride[1] + BLOCK_V[None, :] * W_stride[2],
        mask=MASK_VV,
    )

    start = BLOCK_ID_CHUNK * CHUNK_SIZE
    end = min(S, start + CHUNK_SIZE)

    if BLOCK_ID_CHUNK == 0:
        if h0_ptr is None:
            h = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_V), dtype=k_ptr.dtype.element_ty)
        else:
            h = tl.load(
                h0_ptr
                + BLOCK_ID_B * h0_stride[0]
                + BLOCK_ID_N * h0_stride[1]
                + BLOCK_K[:, None] * h0_stride[2]
                + BLOCK_V[None, :] * h0_stride[3],
                mask=MASK_KV,
            )
    else:
        h = tl.load(
            hc_ptr
            + BLOCK_ID_B * hc_stride[0]
            + (BLOCK_ID_CHUNK - 1) * hc_stride[1]
            + BLOCK_ID_N * hc_stride[2]
            + BLOCK_K[:, None] * hc_stride[3]
            + BLOCK_V[None, :] * hc_stride[4],
            mask=MASK_KV,
        )

    k_ptrs = k_ptr + BLOCK_ID_B * k_stride[0] + start * k_stride[1] + BLOCK_ID_Nk * k_stride[2] + BLOCK_K * k_stride[3]
    v_ptrs = v_ptr + BLOCK_ID_B * v_stride[0] + start * v_stride[1] + BLOCK_ID_Nv * v_stride[2] + BLOCK_V * v_stride[3]
    xf_ptrs = xf_ptr + BLOCK_ID_B * xf_stride[0] + start * xf_stride[1] + BLOCK_ID_Nxf * xf_stride[2]
    h_ptrs = (
        h_ptr
        + tl.cast(BLOCK_ID_B * h_stride[0], tl.uint32)
        + tl.cast(start * h_stride[1], tl.uint32)
        + tl.cast(BLOCK_ID_N * h_stride[2], tl.uint32)
        + tl.cast(BLOCK_K[:, None] * h_stride[3], tl.uint32)
        + tl.cast(BLOCK_V[None, :] * h_stride[4], tl.uint32)
    )

    for s in range(start, end):
        k = tl.load(k_ptrs, mask=MASK_K)
        v = tl.load(v_ptrs, mask=MASK_V)
        f = tl.load(xf_ptrs)

        f, _, h = _forward_single_step(h_prev=h, W=W, k=k, v=v, f=f)
        tl.store(h_ptrs, h, mask=MASK_KV)

        k_ptrs += k_stride[1]
        v_ptrs += v_stride[1]
        xf_ptrs += xf_stride[1]
        h_ptrs += h_stride[1]


@xma_op(mutates_args={"h"})
def rsa_chunked_forward_triton(
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    h0: torch.Tensor | None,
    hc: torch.Tensor,
    h: torch.Tensor,
    CHUNK_SIZE: int,
) -> None:
    N = h.size(-3)
    B, S, Nk, K = k.size()
    Nv, V = v.size()[-2:]
    Nw = W.size(0)
    Nxf = xf.size(-1)

    BLOCK_SIZE_K = get_next_power_of_2(K)
    BLOCK_SIZE_K = max(16, BLOCK_SIZE_K)

    BLOCK_SIZE_V = get_next_power_of_2(V)
    BLOCK_SIZE_V = max(16, BLOCK_SIZE_V)

    NUM_CHUNKS = hc.size(1)

    with torch.device(k.device):
        rsa_chunked_forward_triton_kernel[B, N, NUM_CHUNKS](
            k_ptr=k,
            k_stride=k.stride(),
            v_ptr=v,
            v_stride=v.stride(),
            W_ptr=W,
            W_stride=W.stride(),
            xf_ptr=xf,
            xf_stride=xf.stride(),
            h0_ptr=h0,
            h0_stride=None if h0 is None else h0.stride(),
            hc_ptr=hc,
            hc_stride=hc.stride(),
            h_ptr=h,
            h_stride=h.stride(),
            S=S,
            K=K,
            V=V,
            Gk=N // Nk,
            Gv=N // Nv,
            Gw=N // Nw,
            Gxf=N // Nxf,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
            CHUNK_SIZE=CHUNK_SIZE,
        )
