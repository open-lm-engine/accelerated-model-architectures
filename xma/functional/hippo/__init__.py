# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...accelerator import Accelerator, KernelBackend
from .triton_implementation import hippo_triton


def hippo(
    input: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    input_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    *,
    kernel_backend: KernelBackend | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    computes multihead RNN recurrent update over the sequence length: `tanh(input_state @ weight + input)`

    :param input: input tensor of shape (B, S, Nx, H) where Nx is the number of input heads and H is the head
        dimension. Should have shape (T, Nx, H) and `cu_seqlens` should be passed.
    :type input: torch.Tensor
    :param A: weight tensor of shape (H, H)
    :type A: torch.Tensor
    :param B: weight tensor of shape (H,)
    :type B: torch.Tensor
    :param input_state: starting state of shape (B, N, H), where N = max{Nx, Nw}. None means starting state is
        0 tensor. Defaults to None.
    :type input_state: torch.Tensor | None
    :param cu_seqlens: cumulative sequence length (must contain 0 as first element). Defaults to None.
    :type cu_seqlens: torch.Tensor | None
    :param max_seqlen: max sequence length in the batch. Defaults to None.
    :type max_seqlen: int | None
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: optimal compression tensor of shape (B, S, N, H) if `cu_seqlens` is None else (T, N, H) and output state of
        shape (B, N, H).
    :rtype: Tensor
    """

    assert input.dim() == 3 + (cu_seqlens is None)

    if cu_seqlens is None:
        assert max_seqlen is None
        B, _, _, H = input.size()
    else:
        assert max_seqlen is not None
        assert cu_seqlens.dim() == 1

        B = cu_seqlens.size(0) - 1
        H = input.size(-1)

    assert A.size() == (H, H)
    assert B.size() == (H,)

    if input_state is not None:
        assert input_state.size() == (B, H)

    if kernel_backend is None:
        kernel_backend = Accelerator.get_kernel_backend()
    else:
        assert kernel_backend.verify_accelerator()

    x = input
    h0 = input_state

    h = torch.empty_like(x)

    if kernel_backend == KernelBackend.triton:
        hippo_triton(x=x, A=A, B=B, h0=h0, h=h, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    elif kernel_backend == KernelBackend.torch:
        if cu_seqlens is None:
            _, S, H = x.size()
        else:
            S = max_seqlen
            H = x.size(-1)

        N = B.size(0)

        if h0 is None:
            h0 = torch.zeros(B, H, N, device=x.device, dtype=x.dtype)

        if cu_seqlens is not None:
            h0 = h0.clone()
            start = cu_seqlens[:-1]
            end = cu_seqlens[1:]

        B = B[None, :]

        for s in range(S):
            if cu_seqlens is None:
                h = h0.flatten(0, 1) @ A.T + x[:, s].flatten()[..., None] * B
            else:
                offset = start + s
                unfinished = offset < end
                offset_unfinished = offset[unfinished]

                h = h0[unfinished].flatten(0, 1) @ A.T + x[offset_unfinished].flatten()[..., None] * B
                h0[unfinished] = h

            h = h0
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return h
