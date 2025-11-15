# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....triton_utils import clamp, matmul, sigmoid_backward, tanh_backward
from ...rnn.triton_implementation.backward import _get_autotune_configs, _load_input_state
from .forward import _get_autotune_configs


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_H"], reset_to_zero=["dW_ptr", "dWf_ptr", "dWr_ptr"])
@triton.jit
def gru_backward_triton_kernel(
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
    dx_ptr,
    dx_stride,
    dxf_ptr,
    dxf_stride,
    dxr_ptr,
    dxr_stride,
    dW_ptr,
    dW_stride,
    dWf_ptr,
    dWf_stride,
    dWr_ptr,
    dWr_stride,
    dy_ptr,
    dy_stride,
    cu_seqlens_ptr,
    cu_seqlens_stride,
    IS_MAX_SEQLEN_TENSOR: tl.constexpr,
    max_seqlen_ptr,
    B,
    S,
    H: tl.constexpr,
    gradient_clipping,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    BLOCK_ID_B = tl.program_id(axis=0)
    BLOCK_ID_N = tl.program_id(axis=1)

    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H)

    MASK_B = BLOCK_B < B
    MASK_H = BLOCK_H < H

    MASK_BH = MASK_B[:, None] & MASK_H[None, :]
    MASK_HH = MASK_H[:, None] & MASK_H[None, :]

    dh = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=W_ptr.dtype.element_ty)
    dW = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)
    dWf = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)
    dWr = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_H), dtype=tl.float32)

    W = tl.load(
        W_ptr + BLOCK_ID_N * W_stride[0] + BLOCK_H[:, None] * W_stride[1] + BLOCK_H[None, :] * W_stride[2],
        mask=MASK_HH,
    )

    Wf = tl.load(
        Wf_ptr + BLOCK_ID_N * Wf_stride[0] + BLOCK_H[:, None] * Wf_stride[1] + BLOCK_H[None, :] * Wf_stride[2],
        mask=MASK_HH,
    )

    Wr = tl.load(
        Wr_ptr + BLOCK_ID_N * Wr_stride[0] + BLOCK_H[:, None] * Wr_stride[1] + BLOCK_H[None, :] * Wr_stride[2],
        mask=MASK_HH,
    )

    IS_VARLEN: tl.constexpr = cu_seqlens_ptr is not None

    if IS_VARLEN:
        cu_seqlens_ptrs = cu_seqlens_ptr + BLOCK_B[:, None] * cu_seqlens_stride[0]
        start = tl.load(cu_seqlens_ptrs, mask=MASK_B[:, None])
        end = tl.load(cu_seqlens_ptrs + cu_seqlens_stride[0], mask=MASK_B[:, None])

        if IS_MAX_SEQLEN_TENSOR:
            S = tl.load(max_seqlen_ptr)
        else:
            S = max_seqlen_ptr

        end -= 1

        z_ptrs = z_ptr + end * z_stride[0] + BLOCK_ID_N * z_stride[1] + BLOCK_H[None, :] * z_stride[2]
        f_ptrs = f_ptr + end * f_stride[0] + BLOCK_ID_N * f_stride[1] + BLOCK_H[None, :] * f_stride[2]
        r_ptrs = r_ptr + end * r_stride[0] + BLOCK_ID_N * r_stride[1] + BLOCK_H[None, :] * r_stride[2]
        y_ptrs = y_ptr + end * y_stride[0] + BLOCK_ID_N * y_stride[1] + BLOCK_H[None, :] * y_stride[2]

        dx_ptrs = dx_ptr + end * dx_stride[0] + BLOCK_ID_N * dx_stride[1] + BLOCK_H[None, :] * dx_stride[2]
        dxf_ptrs = dxf_ptr + end * dxf_stride[0] + BLOCK_ID_N * dxf_stride[1] + BLOCK_H[None, :] * dxf_stride[2]
        dxr_ptrs = dxr_ptr + end * dxr_stride[0] + BLOCK_ID_N * dxr_stride[1] + BLOCK_H[None, :] * dxr_stride[2]

        dy_ptrs = dy_ptr + end * dy_stride[0] + BLOCK_ID_N * dy_stride[1] + BLOCK_H[None, :] * dy_stride[2]
    else:
        z_ptrs = (
            z_ptr
            + BLOCK_B[:, None] * z_stride[0]
            + (S - 1) * z_stride[1]
            + BLOCK_ID_N * z_stride[2]
            + BLOCK_H[None, :] * z_stride[3]
        )

        f_ptrs = (
            f_ptr
            + BLOCK_B[:, None] * f_stride[0]
            + (S - 1) * f_stride[1]
            + BLOCK_ID_N * f_stride[2]
            + BLOCK_H[None, :] * f_stride[3]
        )

        r_ptrs = (
            r_ptr
            + BLOCK_B[:, None] * r_stride[0]
            + (S - 1) * r_stride[1]
            + BLOCK_ID_N * r_stride[2]
            + BLOCK_H[None, :] * r_stride[3]
        )

        y_ptrs = (
            y_ptr
            + BLOCK_B[:, None] * y_stride[0]
            + (S - 1) * y_stride[1]
            + BLOCK_ID_N * y_stride[2]
            + BLOCK_H[None, :] * y_stride[3]
        )

        dx_ptrs = (
            dx_ptr
            + BLOCK_B[:, None] * dx_stride[0]
            + (S - 1) * dx_stride[1]
            + BLOCK_ID_N * dx_stride[2]
            + BLOCK_H[None, :] * dx_stride[3]
        )

        dxf_ptrs = (
            dxf_ptr
            + BLOCK_B[:, None] * dxf_stride[0]
            + (S - 1) * dxf_stride[1]
            + BLOCK_ID_N * dxf_stride[2]
            + BLOCK_H[None, :] * dxf_stride[3]
        )

        dxr_ptrs = (
            dxr_ptr
            + BLOCK_B[:, None] * dxr_stride[0]
            + (S - 1) * dxr_stride[1]
            + BLOCK_ID_N * dxr_stride[2]
            + BLOCK_H[None, :] * dxr_stride[3]
        )

        dy_ptrs = (
            dy_ptr
            + BLOCK_B[:, None] * dy_stride[0]
            + (S - 1) * dy_stride[1]
            + BLOCK_ID_N * dy_stride[2]
            + BLOCK_H[None, :] * dy_stride[3]
        )

    # backward counting reduces 1 instruction since we need to compare s == 0, otherwise we have to compare s == S - 1
    for s in range(S - 1, -1, -1):
        if gradient_clipping is not None:
            dh = clamp(dh, min_value=-gradient_clipping, max_value=gradient_clipping)

        if IS_VARLEN:
            MASK = (end >= start) & MASK_H[None, :]
        else:
            MASK = MASK_BH

        dy = tl.load(dy_ptrs, mask=MASK) + dh

        z = tl.load(z_ptrs, mask=MASK)
        f = tl.load(f_ptrs, mask=MASK)
        r = tl.load(r_ptrs, mask=MASK)

        y_ptrs -= y_stride[1 - IS_VARLEN]

        if IS_VARLEN:
            y_prev = tl.where(
                start == end,
                _load_input_state(
                    h0_ptr=h0_ptr,
                    h0_stride=h0_stride,
                    BLOCK_ID_N=BLOCK_ID_N,
                    BLOCK_B=BLOCK_B,
                    BLOCK_H=BLOCK_H,
                    MASK_BH=MASK_BH,
                    BLOCK_SIZE_B=BLOCK_SIZE_B,
                    BLOCK_SIZE_H=BLOCK_SIZE_H,
                    dtype=W.dtype,
                ),
                tl.load(y_ptrs, mask=MASK),
            )
        elif s == 0:
            if h0_ptr is None:
                y_prev = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=W.dtype)
            else:
                y_prev = tl.load(
                    h0_ptr
                    + BLOCK_B[:, None] * h0_stride[0]
                    + BLOCK_ID_N * h0_stride[1]
                    + BLOCK_H[None, :] * h0_stride[2],
                    mask=MASK,
                )
        else:
            y_prev = tl.load(y_ptrs, mask=MASK)

        dh = f * dy
        dz = dy * (1 - f)
        df = dy * (y_prev - z)

        dx = dz * tanh_backward(z)
        drh = matmul(A=dx, B=W.T, C=None, output_dtype=dx.dtype)
        dW = matmul(A=(r * y_prev).T, B=dx, C=dW, output_dtype=dW.dtype)
        tl.store(dx_ptrs, dx, mask=MASK)

        dh += drh * r

        dxf = df * sigmoid_backward(f)
        dh = matmul(A=dxf, B=Wf.T, C=dh, output_dtype=dx.dtype)
        dWf = matmul(A=y_prev.T, B=dxf, C=dWf, output_dtype=dW.dtype)
        tl.store(dxf_ptrs, dxf, mask=MASK)

        dxr = drh * y_prev * sigmoid_backward(r)
        dh = matmul(A=dxr, B=Wr.T, C=dh, output_dtype=dx.dtype)
        dWr = matmul(A=y_prev.T, B=dxr, C=dWr, output_dtype=dW.dtype)
        tl.store(dxr_ptrs, dxr, mask=MASK)

        z_ptrs -= z_stride[1 - IS_VARLEN]
        f_ptrs -= f_stride[1 - IS_VARLEN]
        r_ptrs -= r_stride[1 - IS_VARLEN]

        dx_ptrs -= dx_stride[1 - IS_VARLEN]
        dxf_ptrs -= dxf_stride[1 - IS_VARLEN]
        dxr_ptrs -= dxr_stride[1 - IS_VARLEN]

        dy_ptrs -= dy_stride[1 - IS_VARLEN]

        if IS_VARLEN:
            end -= 1

    tl.atomic_add(
        dW_ptr + BLOCK_ID_N * dW_stride[0] + BLOCK_H[:, None] * dW_stride[1] + BLOCK_H[None, :] * dW_stride[2],
        dW,
        mask=MASK_HH,
        sem="relaxed",
    )

    tl.atomic_add(
        dWf_ptr + BLOCK_ID_N * dWf_stride[0] + BLOCK_H[:, None] * dWf_stride[1] + BLOCK_H[None, :] * dWf_stride[2],
        dWf,
        mask=MASK_HH,
        sem="relaxed",
    )

    tl.atomic_add(
        dWr_ptr + BLOCK_ID_N * dWr_stride[0] + BLOCK_H[:, None] * dWr_stride[1] + BLOCK_H[None, :] * dWr_stride[2],
        dWr,
        mask=MASK_HH,
        sem="relaxed",
    )


@custom_op(
    f"{LIBRARY_NAME}::gru_backward_triton",
    mutates_args={
        "forget_input_grad",
        "forget_weight_grad",
        "reset_input_grad",
        "reset_weight_grad",
        "input_grad",
        "weight_grad",
    },
)
def gru_backward_triton(
    weight: torch.Tensor,
    output: torch.Tensor,
    forget_weight: torch.Tensor,
    forget_gate: torch.Tensor,
    forget_input_grad: torch.Tensor,
    forget_weight_grad: torch.Tensor,
    reset_weight: torch.Tensor,
    reset_gate: torch.Tensor,
    reset_input_grad: torch.Tensor,
    reset_weight_grad: torch.Tensor,
    output_update: torch.Tensor,
    input_state: torch.Tensor | None,
    output_grad: torch.Tensor,
    input_grad: torch.Tensor,
    weight_grad: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    max_seqlen_tensor: torch.Tensor | None,
    max_seqlen: int | None,
    gradient_clipping: float | None,
) -> None:
    if cu_seqlens is None:
        assert max_seqlen is None
        assert max_seqlen_tensor is None

        B, S, N, H = output.size()
    else:
        B = cu_seqlens.size(0) - 1
        S = None
        _, N, H = output.size()

    is_max_seqlen_tensor = max_seqlen_tensor is not None

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), N)

    with torch.device(output.device):
        gru_backward_triton_kernel[GRID](
            W_ptr=weight,
            W_stride=weight.stride(),
            Wf_ptr=forget_weight,
            Wf_stride=forget_weight.stride(),
            Wr_ptr=reset_weight,
            Wr_stride=reset_weight.stride(),
            z_ptr=output_update,
            z_stride=output_update.stride(),
            f_ptr=forget_gate,
            f_stride=forget_gate.stride(),
            r_ptr=reset_gate,
            r_stride=reset_gate.stride(),
            h0_ptr=input_state,
            h0_stride=None if input_state is None else input_state.stride(),
            y_ptr=output,
            y_stride=output.stride(),
            dx_ptr=input_grad,
            dx_stride=input_grad.stride(),
            dxf_ptr=forget_input_grad,
            dxf_stride=forget_input_grad.stride(),
            dxr_ptr=reset_input_grad,
            dxr_stride=reset_input_grad.stride(),
            dW_ptr=weight_grad,
            dW_stride=weight_grad.stride(),
            dWf_ptr=forget_weight_grad,
            dWf_stride=forget_weight_grad.stride(),
            dWr_ptr=reset_weight_grad,
            dWr_stride=reset_weight_grad.stride(),
            dy_ptr=output_grad,
            dy_stride=output_grad.stride(),
            cu_seqlens_ptr=cu_seqlens,
            cu_seqlens_stride=None if cu_seqlens is None else cu_seqlens.stride(),
            IS_MAX_SEQLEN_TENSOR=is_max_seqlen_tensor,
            max_seqlen_ptr=max_seqlen_tensor if is_max_seqlen_tensor else max_seqlen,
            B=B,
            S=S,
            H=H,
            gradient_clipping=gradient_clipping,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
