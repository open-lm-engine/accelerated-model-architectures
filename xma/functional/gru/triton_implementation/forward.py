# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import ceil_divide, get_next_power_of_2
from ....triton_utils import matmul, sigmoid, tanh
from ...rnn.triton_implementation.forward import _get_autotune_configs
from ..utils import _get_num_heads


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_H"])
@triton.jit
def gru_forward_triton_kernel(
    x_ptr,
    x_stride,
    xf_ptr,
    xf_stride,
    xr_ptr,
    xr_stride,
    W_ptr,
    W_stride,
    Wf_ptr,
    Wf_stride,
    Wr_ptr,
    Wr_stride,
    z_ptr,
    z_stride,
    f_ptr,
    f_stride,
    r_ptr,
    r_stride,
    h0_ptr,
    h0_stride,
    y_ptr,
    y_stride,
    cu_seqlens_ptr,
    cu_seqlens_stride,
    IS_MAX_SEQLEN_TENSOR: tl.constexpr,
    max_seqlen_ptr,
    B,
    S,
    H: tl.constexpr,
    Gx: tl.constexpr,
    Gxf: tl.constexpr,
    Gxr: tl.constexpr,
    Gw: tl.constexpr,
    Gwf: tl.constexpr,
    Gwr: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(0)
    BLOCK_ID_N = tl.program_id(1)

    BLOCK_ID_Nx = BLOCK_ID_N // Gx
    BLOCK_ID_Nxf = BLOCK_ID_N // Gxf
    BLOCK_ID_Nxr = BLOCK_ID_N // Gxr

    BLOCK_ID_Nw = BLOCK_ID_N // Gw
    BLOCK_ID_Nwf = BLOCK_ID_N // Gwf
    BLOCK_ID_Nwr = BLOCK_ID_N // Gwr

    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)

    MASK_B = BLOCK_B < B
    MASK_H = BLOCK_H < H

    MASK_BH = MASK_B[:, None] & MASK_H[None, :]
    MASK_HH = MASK_H[:, None] & MASK_H[None, :]

    W = tl.load(
        W_ptr + BLOCK_ID_Nw * W_stride[0] + BLOCK_H[:, None] * W_stride[1] + BLOCK_H[None, :] * W_stride[2],
        mask=MASK_HH,
    )

    Wf = tl.load(
        Wf_ptr + BLOCK_ID_Nwf * Wf_stride[0] + BLOCK_H[:, None] * Wf_stride[1] + BLOCK_H[None, :] * Wf_stride[2],
        mask=MASK_HH,
    )

    Wr = tl.load(
        Wr_ptr + BLOCK_ID_Nwr * Wr_stride[0] + BLOCK_H[:, None] * Wr_stride[1] + BLOCK_H[None, :] * Wr_stride[2],
        mask=MASK_HH,
    )

    if h0_ptr is None:
        h = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=x_ptr.dtype.element_ty)
    else:
        h = tl.load(
            h0_ptr + BLOCK_B[:, None] * h0_stride[0] + BLOCK_ID_N * h0_stride[1] + BLOCK_H[None, :] * h0_stride[2],
            mask=MASK_BH,
        )

    IS_VARLEN: tl.constexpr = cu_seqlens_ptr is not None

    if IS_VARLEN:
        cu_seqlens_ptrs = cu_seqlens_ptr + BLOCK_B[:, None] * cu_seqlens_stride[0]
        start = tl.load(cu_seqlens_ptrs, mask=MASK_B[:, None])
        end = tl.load(cu_seqlens_ptrs + cu_seqlens_stride[0], mask=MASK_B[:, None])

        S = tl.load(max_seqlen_ptr) if IS_MAX_SEQLEN_TENSOR else max_seqlen_ptr

        x_ptrs = x_ptr + start * x_stride[0] + BLOCK_ID_Nx * x_stride[1] + BLOCK_H[None, :] * x_stride[2]
        xf_ptrs = xf_ptr + start * xf_stride[0] + BLOCK_ID_Nxf * xf_stride[1] + BLOCK_H[None, :] * xf_stride[2]
        xr_ptrs = xr_ptr + start * xr_stride[0] + BLOCK_ID_Nxr * xr_stride[1] + BLOCK_H[None, :] * xr_stride[2]

        if z_ptr is not None:
            z_ptrs = z_ptr + start * z_stride[0] + BLOCK_ID_N * z_stride[1] + BLOCK_H[None, :] * z_stride[2]

        if r_ptr is not None:
            r_ptrs = r_ptr + start * r_stride[0] + BLOCK_ID_N * r_stride[1] + BLOCK_H[None, :] * r_stride[2]

        if f_ptr is not None:
            f_ptrs = f_ptr + start * f_stride[0] + BLOCK_ID_N * f_stride[1] + BLOCK_H[None, :] * f_stride[2]

        y_ptrs = y_ptr + start * y_stride[0] + BLOCK_ID_N * y_stride[1] + BLOCK_H[None, :] * y_stride[2]
    else:
        x_ptrs = x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_ID_Nx * x_stride[2] + BLOCK_H[None, :] * x_stride[3]
        xf_ptrs = (
            xf_ptr + BLOCK_B[:, None] * xf_stride[0] + BLOCK_ID_Nxf * xf_stride[2] + BLOCK_H[None, :] * xf_stride[3]
        )
        xr_ptrs = (
            xr_ptr + BLOCK_B[:, None] * xr_stride[0] + BLOCK_ID_Nxr * xr_stride[2] + BLOCK_H[None, :] * xr_stride[3]
        )

        if z_ptr is not None:
            z_ptrs = z_ptr + BLOCK_B[:, None] * z_stride[0] + BLOCK_ID_N * z_stride[2] + BLOCK_H[None, :] * z_stride[3]

        if r_ptr is not None:
            r_ptrs = r_ptr + BLOCK_B[:, None] * r_stride[0] + BLOCK_ID_N * r_stride[2] + BLOCK_H[None, :] * r_stride[3]

        if f_ptr is not None:
            f_ptrs = f_ptr + BLOCK_B[:, None] * f_stride[0] + BLOCK_ID_N * f_stride[2] + BLOCK_H[None, :] * f_stride[3]

        y_ptrs = y_ptr + BLOCK_B[:, None] * y_stride[0] + BLOCK_ID_N * y_stride[2] + BLOCK_H[None, :] * y_stride[3]

    for _ in range(S):
        MASK = ((start < end) & MASK_H[None, :]) if IS_VARLEN else MASK_BH

        x = tl.load(xr_ptrs, mask=MASK)
        r = matmul(A=h, B=Wr, C=x, output_dtype=tl.float32)
        r = sigmoid(r, output_dtype=x.dtype)

        if r_ptr is not None:
            tl.store(r_ptrs, r, mask=MASK)

        x = tl.load(x_ptrs, mask=MASK)
        z = matmul(A=h * r, B=W, C=x, output_dtype=tl.float32)
        z = tanh(z, output_dtype=x.dtype)

        if z_ptr is not None:
            tl.store(z_ptrs, z, mask=MASK)

        x = tl.load(xf_ptrs, mask=MASK)
        f = matmul(A=h, B=Wf, C=x, output_dtype=tl.float32)
        f = sigmoid(f, output_dtype=x.dtype)

        if f_ptr is not None:
            tl.store(f_ptrs, f, mask=MASK)

        h = f * h + (1 - f) * z
        tl.store(y_ptrs, h, mask=MASK)

        x_ptrs += x_stride[1 - IS_VARLEN]
        xr_ptrs += xr_stride[1 - IS_VARLEN]
        xf_ptrs += xf_stride[1 - IS_VARLEN]

        if z_ptr is not None:
            z_ptrs += z_stride[1 - IS_VARLEN]

        if r_ptr is not None:
            r_ptrs += r_stride[1 - IS_VARLEN]

        if f_ptr is not None:
            f_ptrs += f_stride[1 - IS_VARLEN]

        y_ptrs += y_stride[1 - IS_VARLEN]

        if IS_VARLEN:
            start += 1


@xma_op(mutates_args={"f", "r", "z", "y"})
def gru_forward_triton(
    x: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    Wf: torch.Tensor,
    f: torch.Tensor | None,
    xr: torch.Tensor,
    Wr: torch.Tensor,
    r: torch.Tensor | None,
    z: torch.Tensor | None,
    h0: torch.Tensor | None,
    y: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    max_seqlen_tensor: torch.Tensor | None,
    max_seqlen: int | None,
) -> None:
    if cu_seqlens is None:
        assert max_seqlen is None
        assert max_seqlen_tensor is None

        B, S, _, H = x.size()
    else:
        B = cu_seqlens.size(0) - 1
        S = None
        _, _, H = x.size()

    Nx, Nxf, Nxr, Nw, Nwf, Nwr, N = _get_num_heads(x=x, W=W, xf=xf, Wf=Wf, xr=xr, Wr=Wr, run_check=False)
    is_max_seqlen_tensor = max_seqlen_tensor is not None

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)
    GRID = lambda kwargs: (ceil_divide(B, kwargs["BLOCK_SIZE_B"]), N)

    gru_forward_triton_kernel[GRID](
        x_ptr=x,
        x_stride=x.stride(),
        xf_ptr=xf,
        xf_stride=xf.stride(),
        xr_ptr=xr,
        xr_stride=xr.stride(),
        W_ptr=W,
        W_stride=W.stride(),
        Wf_ptr=Wf,
        Wf_stride=Wf.stride(),
        Wr_ptr=Wr,
        Wr_stride=Wr.stride(),
        z_ptr=z,
        z_stride=None if z is None else z.stride(),
        f_ptr=f,
        f_stride=None if f is None else f.stride(),
        r_ptr=r,
        r_stride=None if r is None else r.stride(),
        h0_ptr=h0,
        h0_stride=None if h0 is None else h0.stride(),
        y_ptr=y,
        y_stride=y.stride(),
        cu_seqlens_ptr=cu_seqlens,
        cu_seqlens_stride=None if cu_seqlens is None else cu_seqlens.stride(),
        IS_MAX_SEQLEN_TENSOR=is_max_seqlen_tensor,
        max_seqlen_ptr=max_seqlen_tensor if is_max_seqlen_tensor else max_seqlen,
        B=B,
        S=S,
        H=H,
        Gx=N // Nx,
        Gxf=N // Nxf,
        Gxr=N // Nxr,
        Gw=N // Nw,
        Gwf=N // Nwf,
        Gwr=N // Nwr,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
