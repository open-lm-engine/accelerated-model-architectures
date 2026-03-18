# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....custom_op import xma_op
from ....math import ceil_divide, get_next_power_of_2
from ....triton_utils import silu


@triton.jit
def causal_convolution_triton_kernel(
    x_ptr,
    x_stride,
    h0_ptr,
    h0_stride,
    W_ptr,
    W_stride,
    b_ptr,
    b_stride,
    y_ptr,
    y_stride,
    B,
    S,
    H: tl.constexpr,
    K: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    BLOCK_ID_H = tl.program_id(0)
    BLOCK_ID_S = tl.program_id(1)
    BLOCK_ID_B = tl.program_id(2)

    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    BLOCK_H = BLOCK_ID_H * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    BLOCK_K = tl.arange(0, BLOCK_SIZE_K)

    MASK_B = BLOCK_B < B
    MASK_H = BLOCK_H < H
    MASK_K = BLOCK_K < K

    W = tl.load(
        W_ptr + BLOCK_H[:, None] * W_stride[0] + BLOCK_K[None, :] * W_stride[2], mask=MASK_H[:, None] & MASK_K[None, :]
    )

    if h0_ptr is None:
        BLOCK_S = BLOCK_ID_S - K + 1 + BLOCK_K
        MASK_SK = (0 <= BLOCK_S) & (BLOCK_S < S) & MASK_K

        x = tl.load(
            x_ptr
            + BLOCK_B[:, None, None] * x_stride[0]
            + BLOCK_S[None, :, None] * x_stride[1]
            + BLOCK_H[None, None, :] * x_stride[2],
            mask=MASK_B[:, None, None] & MASK_SK[None, :, None] & MASK_H[None, None, :],
        )

        y = tl.sum(x * W.T[None, :, :], axis=1)
    else:
        MASK_K_H0 = BLOCK_K < K - 1  # h0 has only K-1 positions

        # Load x[b, 0, h] — new input token (S=1 in generation)
        x_new = tl.load(
            x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_H[None, :] * x_stride[2],
            mask=MASK_B[:, None] & MASK_H[None, :],
            other=0.0,
        )  # [BLOCK_SIZE_B, BLOCK_SIZE_H]

        # Load h0[b, k, h] for k in [0, K-2] — for dot product
        h0_orig = tl.load(
            h0_ptr
            + BLOCK_B[:, None, None] * h0_stride[0]
            + BLOCK_K[None, :, None] * h0_stride[1]
            + BLOCK_H[None, None, :] * h0_stride[2],
            mask=MASK_B[:, None, None] & MASK_K_H0[None, :, None] & MASK_H[None, None, :],
            other=0.0,
        )  # [BLOCK_SIZE_B, BLOCK_SIZE_K, BLOCK_SIZE_H]

        # Construct full K-length input: h0_orig at 0..K-2, x_new at K-1
        IS_LAST_K = BLOCK_K == K - 1
        full_input = tl.where(IS_LAST_K[None, :, None], x_new[:, None, :], h0_orig)

        # y[b, h] = sum_k full_input[b, k, h] * W[h, k]
        y = tl.sum(full_input * W.T[None, :, :], axis=1)  # [BLOCK_SIZE_B, BLOCK_SIZE_H]

        # Load h0[b, k+1, h] for the state update; clamp to K-2 to stay in-bounds
        BLOCK_K_SHIFTED = tl.where(BLOCK_K < K - 2, BLOCK_K + 1, K - 2)
        h0_shifted = tl.load(
            h0_ptr
            + BLOCK_B[:, None, None] * h0_stride[0]
            + BLOCK_K_SHIFTED[None, :, None] * h0_stride[1]
            + BLOCK_H[None, None, :] * h0_stride[2],
            mask=MASK_B[:, None, None] & MASK_K_H0[None, :, None] & MASK_H[None, None, :],
            other=0.0,
        )  # [BLOCK_SIZE_B, BLOCK_SIZE_K, BLOCK_SIZE_H]

        # Replace last slot (K-2) with x_new
        IS_LAST_H0 = BLOCK_K == K - 2
        h0_updated = tl.where(IS_LAST_H0[None, :, None], x_new[:, None, :], h0_shifted)

        tl.store(
            h0_ptr
            + BLOCK_B[:, None, None] * h0_stride[0]
            + BLOCK_K[None, :, None] * h0_stride[1]
            + BLOCK_H[None, None, :] * h0_stride[2],
            h0_updated,
            mask=MASK_B[:, None, None] & MASK_K_H0[None, :, None] & MASK_H[None, None, :],
        )

    if b_ptr is not None:
        b = tl.load(b_ptr + BLOCK_H * b_stride[0], mask=MASK_H)
        y = y + b[None, :]

    if ACTIVATION == "swiglu" or ACTIVATION == "silu":
        y = silu(y)

    tl.store(
        y_ptr + BLOCK_B[:, None] * y_stride[0] + BLOCK_ID_S * y_stride[1] + BLOCK_H[None, :] * y_stride[2],
        y,
        mask=MASK_B[:, None] & MASK_H[None, :],
    )


@xma_op(mutates_args={"y", "h0"})
def causal_convolution_triton(
    x: torch.Tensor,
    h0: torch.Tensor | None,
    W: torch.Tensor,
    b: torch.Tensor | None,
    y: torch.Tensor,
    activation_function: str,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
) -> None:
    if cu_seqlens is None:
        B, S, H = x.size()
    else:
        T, H = x.size()
        B = cu_seqlens.size(0) - 1
        S = max_seqlen

    K = W.size(-1)

    BLOCK_SIZE_H = max(16, get_next_power_of_2(H))
    BLOCK_SIZE_B = 1

    GRID = lambda kwargs: (
        ceil_divide(kwargs["H"], kwargs["BLOCK_SIZE_H"]),
        S,
        ceil_divide(kwargs["B"], kwargs["BLOCK_SIZE_B"]),
    )

    causal_convolution_triton_kernel[GRID](
        x_ptr=x,
        x_stride=x.stride(),
        h0_ptr=h0,
        h0_stride=None if h0 is None else h0.stride(),
        W_ptr=W,
        W_stride=W.stride(),
        b_ptr=b,
        b_stride=None if b is None else b.stride(),
        y_ptr=y,
        y_stride=y.stride(),
        B=B,
        S=S,
        H=H,
        K=K,
        ACTIVATION=activation_function,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_K=get_next_power_of_2(K),
    )
