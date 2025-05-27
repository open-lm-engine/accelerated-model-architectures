# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....constants import LIBRARY_NAME
from ....math import ceil_divide, get_next_power_of_2
from ....utils import cute_op
from ...rnn.triton_implementation.forward import _get_autotune_configs, _rnn_forward_update


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_H"])
@triton.jit
def gru_varlen_forward_triton_kernel(
    x_ptr,
    x_stride_t,
    W_ptr,
    W_stride_n,
    xf_ptr,
    Wf_ptr,
    f_ptr,
    xr_ptr,
    Wr_ptr,
    r_ptr,
    z_ptr,
    HAS_INPUT_STATE: tl.constexpr,
    h_ptr,
    h_stride_b,
    y_ptr,
    cu_seqlens_ptr,
    IS_MAX_SEQLEN_TENSOR: tl.constexpr,
    max_seqlen_ptr,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = tl.arange(0, BLOCK_SIZE_H)

    mask_b = indices_b < B
    mask_h = indices_h < H
    mask_bh = mask_b[:, None] & mask_h[None, :]

    indices = pid_n * W_stride_n + indices_h[:, None] * H + indices_h[None, :]
    mask_hh = mask_h[:, None] & mask_h[None, :]

    W = tl.load(W_ptr + indices, mask=mask_hh)
    Wf = tl.load(Wf_ptr + indices, mask=mask_hh)
    Wr = tl.load(Wr_ptr + indices, mask=mask_hh)

    if HAS_INPUT_STATE:
        h = tl.load(h_ptr + indices_b[:, None] * h_stride_b + pid_n * H + indices_h[None, :], mask=mask_bh)
    else:
        h = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_H), dtype=x_ptr.dtype.element_ty)

    cu_seqlens_ptrs = cu_seqlens_ptr + indices_b[:, None]
    start = tl.load(cu_seqlens_ptrs, mask=mask_b[:, None])
    end = tl.load(cu_seqlens_ptrs + 1, mask=mask_b[:, None])

    if IS_MAX_SEQLEN_TENSOR:
        max_seqlen = tl.load(max_seqlen_ptr)
    else:
        max_seqlen = max_seqlen_ptr

    indices = start * x_stride_t + pid_n * H + indices_h[None, :]

    for _ in range(max_seqlen):
        unfinished = start < end
        mask = unfinished & mask_h[None, :]

        r = _rnn_forward_update(
            h=h,
            W=Wr,
            x=tl.load(xr_ptr + indices, mask=mask),
            ACTIVATION_FUNCTION="sigmoid",
            relu_negative_slope=None,
        )

        tl.store(r_ptr + indices, r, mask=mask)

        z = _rnn_forward_update(
            h=h * r,
            W=W,
            x=tl.load(x_ptr + indices, mask=mask),
            ACTIVATION_FUNCTION="tanh",
            relu_negative_slope=None,
        )

        tl.store(z_ptr + indices, z, mask=mask)

        f = _rnn_forward_update(
            h=h,
            W=Wf,
            x=tl.load(xf_ptr + indices, mask=mask),
            ACTIVATION_FUNCTION="sigmoid",
            relu_negative_slope=None,
        )

        tl.store(f_ptr + indices, f, mask=mask)

        h = f * h + (1 - f) * z
        tl.store(y_ptr + indices, h, mask=mask)

        indices += x_stride_t
        start += 1


@cute_op(
    f"{LIBRARY_NAME}::gru_varlen_forward_triton", mutates_args={"forget_gate", "reset_gate", "output_update", "output"}
)
def gru_varlen_forward_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    forget_input: torch.Tensor,
    forget_weight: torch.Tensor,
    forget_gate: torch.Tensor,
    reset_input: torch.Tensor,
    reset_weight: torch.Tensor,
    reset_gate: torch.Tensor,
    output_update: torch.Tensor,
    input_state: torch.Tensor | None,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen_tensor: torch.Tensor | None,
    max_seqlen: int | None,
) -> None:
    B = cu_seqlens.size(0) - 1
    _, N, H = input.size()

    BLOCK_SIZE_H = get_next_power_of_2(H)
    BLOCK_SIZE_H = max(16, BLOCK_SIZE_H)
    GRID = lambda meta: (ceil_divide(B, meta["BLOCK_SIZE_B"]), N)

    has_input_state = input_state is not None
    is_max_seqlen_tensor = max_seqlen_tensor is not None

    with torch.device(input.device):
        gru_varlen_forward_triton_kernel[GRID](
            x_ptr=input,
            x_stride_t=input.stride(0),
            W_ptr=weight,
            W_stride_n=weight.stride(0),
            xf_ptr=forget_input,
            Wf_ptr=forget_weight,
            f_ptr=forget_gate,
            xr_ptr=reset_input,
            Wr_ptr=reset_weight,
            r_ptr=reset_gate,
            z_ptr=output_update,
            HAS_INPUT_STATE=has_input_state,
            h_ptr=input_state,
            h_stride_b=input_state.stride(0) if has_input_state else None,
            y_ptr=output,
            cu_seqlens_ptr=cu_seqlens,
            IS_MAX_SEQLEN_TENSOR=is_max_seqlen_tensor,
            max_seqlen_ptr=max_seqlen_tensor if is_max_seqlen_tensor else max_seqlen,
            B=B,
            H=H,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
        )
