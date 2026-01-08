# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...accelerator import KernelBackend
from ...custom_op import CustomOp
from ..sequence_packing import pack_sequence, unpack_sequence


class _CausalShortConvolution1D(CustomOp):
    @staticmethod
    def forward_backward_torch(
        x: torch.Tensor,
        W: torch.Tensor,
        b: torch.Tensor | None,
        stride: int,
        groups: int,
        h0: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
        activation_function: str,
    ) -> torch.Tensor:
        K = W.size(0)
        B = x.size(0) if cu_seqlens is None else cu_seqlens.size(0) - 1

        if h0 is not None:
            if cu_seqlens is None:
                x = torch.cat([h0, x], dim=1)
            else:
                B = cu_seqlens.size(0) - 1
                T = x.size(0)

                y = []
                for b in range(cu_seqlens.size(0) - 1):
                    start = cu_seqlens[b]
                    end = cu_seqlens[b + 1]

                    y.append(torch.cat([h0[b], x[start:end]]))

                x = torch.cat(y)
                max_seqlen += K

        if cu_seqlens is not None:
            x = unpack_sequence(
                inputs=torch.cat(y),
                cu_seqlens=cu_seqlens,
                batch_size=B,
                sequence_length=max_seqlen + K,
                kernel_backend=KernelBackend.torch,
            )

        S = x.size(1)

        h = F.pad(x, (0, 0, K - S, 0)) if S < K else x[:, 1 - K :]
        x = x.transpose(-1, -2)

        x = F.conv1d(input=x, weight=W, bias=b, stride=stride, padding=K - 1, groups=groups)

        # removes padding on the right side of the sequence
        x = x[..., : 1 - K]

        if activation_function == "silu":
            x = F.silu(x)

        x = x.transpose(-1, -2)

        if cu_seqlens is not None:
            x = pack_sequence(inputs=x, cu_seqlens=cu_seqlens, total_tokens=T, kernel_backend=KernelBackend.torch)

        return x, h

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        W: torch.Tensor,
        b: torch.Tensor | None,
        stride: int,
        groups: int,
        h0: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
        activation_function: str,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def causal_short_convolution_1D(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    groups: int,
    input_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    activation_function: str = "identity",
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

    K = weight.size(0)

    if input_state is not None:
        assert input_state.size() == (B, K - 1, H)

    assert activation_function in ["silu", "identity"]

    input, input_state = _CausalShortConvolution1D.run(
        x=input,
        W=weight,
        b=bias,
        stride=stride,
        groups=groups,
        h0=input_state,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        kernel_backend=kernel_backend,
    )

    return input, input_state
