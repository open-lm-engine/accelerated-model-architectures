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
        x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, stride: int, padding: int, groups: int
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
    padding: int,
    groups: int,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: torch.Tensor | int | None = None,
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
    :return:
    :rtype: tuple[Tensor, Tensor]
    """

    input = _CausalShortConvolution1D.run(
        input=input, weight=weight, bias=bias, stride=stride, groups=groups, input_state=input_state
    )

    input_state = input[:, -1] if cu_seqlens is None else input[cu_seqlens[1:] - 1]

    return input, input_state
