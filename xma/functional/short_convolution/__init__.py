# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...accelerator import KernelBackend
from ...custom_op import CustomOp


class _CausalShortConvolution1D(CustomOp):
    @staticmethod
    def forward_backward_torch(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        stride: int,
        groups: int,
        input_state: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
    ) -> torch.Tensor:
        x = F.conv1d(
            input=x,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=kernel_size - 1,
            groups=groups,
        )

        return x


def causal_short_convolution_1D(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    groups: int,
    input_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    *,
    kernel_backend: KernelBackend | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    1D short causal convolution

    :param input: input tensor
    :type input: torch.Tensor
    :param weight: weight tensor
    :type weight: torch.Tensor
    :param bias: bias tensor
    :type bias: torch.Tensor | None
    :param stride: convolution stride
    :type stride: int
    :param groups: convolution groups
    :type groups: int
    :param cu_seqlens: cumulative sequence length (must contain 0 as first element). Defaults to None.
    :type cu_seqlens: torch.Tensor | None
    :param max_seqlen: max sequence length in the batch. Defaults to None.
    :type max_seqlen: int | None
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output tensor of shape (B, S, H) if `cu_seqlens` is None else (T, H) and output state of shape (B, H).
    :rtype: tuple[Tensor, Tensor]
    """

    if cu_seqlens is None:
        B, _, H = input.size()
    else:
        B = cu_seqlens.size(0) - 1
        H = input.size(-1)

    if input_state is not None:
        assert input_state.size() == (B, H)

    input = _CausalShortConvolution1D.run(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        groups=groups,
        input_state=input_state,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        kernel_backend=kernel_backend,
    )

    input_state = input[:, -1] if cu_seqlens is None else input[cu_seqlens[1:] - 1]

    return input, input_state
