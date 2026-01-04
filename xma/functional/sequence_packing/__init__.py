# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Sequence

import torch

from ...accelerator import KernelBackend
from ...custom_op import CustomOp, ctx_save_for_backward
from ...utils import is_triton_available
from .cuda_implementation import pack_unpack_sequence_cuda


if is_triton_available():
    from .triton_implementation import pack_unpack_sequence_triton


class _PackSequence(CustomOp):
    @staticmethod
    def forward_backward_torch(
        x: torch.Tensor, cu_seqlens: torch.Tensor, output_shape: tuple[int], padding_side: str
    ) -> torch.Tensor:
        B, S = x.size()[:2]
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        batch_indices = torch.arange(B, device=x.device).repeat_interleave(seqlens)

        if padding_side == "left":
            pad_tokens = S - seqlens
            seq_indices = torch.cat([torch.arange(sl, S, device=x.device) for sl in pad_tokens])
        elif padding_side == "right":
            seq_indices = torch.cat([torch.arange(sl, device=x.device) for sl in seqlens])
        else:
            raise ValueError(f"unexpected padding_side ({padding_side})")

        x = x[batch_indices, seq_indices]

        return x

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        output_shape: tuple[int],
        padding_side: str,
        kernel_backend: KernelBackend,
    ) -> torch.Tensor:
        ctx.kernel_backend = kernel_backend

        if kernel_backend == KernelBackend.cuda:
            x = x.contiguous()
            cu_seqlens = cu_seqlens.contiguous()

        ctx_save_for_backward(ctx, cu_seqlens)
        ctx.padding_side = padding_side
        ctx.x_shape = x.size()

        y = torch.empty(output_shape, device=x.device, dtype=x.dtype)

        if kernel_backend == KernelBackend.cuda:
            pack_unpack_sequence_cuda(
                x=x, output=y, cu_seqlens=cu_seqlens, padding_side=padding_side, pack=True, BLOCK_SIZE=1024
            )
        elif kernel_backend == KernelBackend.triton:
            pack_unpack_sequence_triton(x=x, output=y, cu_seqlens=cu_seqlens, padding_side=padding_side, pack=True)
        else:
            raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
        kernel_backend = ctx.kernel_backend
        cu_seqlens = ctx.saved_tensors[0]

        dx = torch.zeros(*ctx.x_shape, device=dy.device, dtype=dy.dtype)

        if kernel_backend == KernelBackend.cuda:
            dy = dy.contiguous()

            pack_unpack_sequence_cuda(
                x=dy,
                output=dx,
                cu_seqlens=cu_seqlens,
                padding_side=ctx.padding_side,
                pack=False,
                BLOCK_SIZE=1024,
            )
        elif kernel_backend == KernelBackend.triton:
            pack_unpack_sequence_triton(
                x=dy, output=dx, cu_seqlens=cu_seqlens, padding_side=ctx.padding_side, pack=False
            )
        else:
            raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

        return dx, *[None] * 4


class _UnpackSequence(CustomOp):
    @staticmethod
    def forward_backward_torch(
        x: torch.Tensor, cu_seqlens: torch.Tensor, output_shape: tuple[int], padding_side: str
    ) -> torch.Tensor:
        B = cu_seqlens.size(0) - 1
        S = output_shape[1]

        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        batch_indices = torch.arange(B, device=x.device).repeat_interleave(seqlens)

        if padding_side == "left":
            pad_tokens = S - seqlens
            seq_indices = torch.cat([torch.arange(sl, S, device=x.device) for sl in pad_tokens])
        elif padding_side == "right":
            seq_indices = torch.cat([torch.arange(sl, device=x.device) for sl in seqlens])
        else:
            raise ValueError(f"unexpected padding_side ({padding_side})")

        padded = torch.zeros(output_shape, dtype=x.dtype, device=x.device)
        padded[batch_indices, seq_indices] = x

        return padded

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        output_shape: tuple[int],
        padding_side: str,
        kernel_backend: KernelBackend,
    ) -> torch.Tensor:
        ctx.kernel_backend = kernel_backend

        if kernel_backend == KernelBackend.cuda:
            x = x.contiguous()
            cu_seqlens = cu_seqlens.contiguous()

        ctx_save_for_backward(ctx, cu_seqlens)
        ctx.padding_side = padding_side
        ctx.x_shape = x.size()

        y = torch.zeros(*output_shape, device=x.device, dtype=x.dtype)

        if kernel_backend == KernelBackend.cuda:
            pack_unpack_sequence_cuda(
                x=x, output=y, cu_seqlens=cu_seqlens, padding_side=padding_side, pack=False, BLOCK_SIZE=1024
            )
        elif kernel_backend == KernelBackend.triton:
            pack_unpack_sequence_triton(x=x, output=y, cu_seqlens=cu_seqlens, padding_side=padding_side, pack=False)
        else:
            raise ValueError(f"unexpected padding_side ({padding_side})")

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
        kernel_backend = ctx.kernel_backend
        padding_side = ctx.padding_side
        cu_seqlens = ctx.saved_tensors[0]

        dx = torch.empty(ctx.x_shape, device=dy.device, dtype=dy.dtype)

        if kernel_backend == KernelBackend.cuda:
            dy = dy.contiguous()

            pack_unpack_sequence_cuda(
                x=dy,
                output=dx,
                cu_seqlens=cu_seqlens,
                padding_side=padding_side,
                pack=True,
                BLOCK_SIZE=1024,
            )
        elif kernel_backend == KernelBackend.triton:
            pack_unpack_sequence_triton(x=dy, output=dx, cu_seqlens=cu_seqlens, padding_side=padding_side, pack=True)
        else:
            raise ValueError(f"unexpected padding_side ({padding_side})")

        return dx, *[None] * 4


def pack_sequence(
    inputs: Sequence[torch.Tensor],
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    padding_side: str = "left",
    *,
    kernel_backend: KernelBackend | None = None,
) -> Sequence[torch.Tensor]:
    """
    pack tensors

    :param inputs: list of tensors
    :type inputs: Sequence[torch.Tensor]
    :param cu_seqlens: cumulative sequence length
    :type cu_seqlens: torch.Tensor
    :param total_tokens: total number of tokens
    :type total_tokens: int
    :param padding_side: padding side
    :type padding_side: str
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: list of packed tensors
    :rtype: Sequence[Tensor]
    """

    assert padding_side in ["left", "right"]
    assert isinstance(inputs, (list, tuple))

    outputs = []

    for x in inputs:
        assert x.dim() >= 2
        assert x.size(0) == cu_seqlens.size(0) - 1

        x = _PackSequence.run(
            x=x,
            cu_seqlens=cu_seqlens,
            output_shape=(total_tokens, *x.size()[2:]),
            padding_side=padding_side,
            kernel_backend=kernel_backend,
        )

        outputs.append(x)

    return outputs


def unpack_sequence(
    inputs: Sequence[torch.Tensor],
    cu_seqlens: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    padding_side: str = "left",
    *,
    kernel_backend: KernelBackend | None = None,
) -> Sequence[torch.Tensor]:
    """
    unpack tensors

    :param inputs: list of tensors
    :type inputs: Sequence[torch.Tensor]
    :param cu_seqlens: cumulative sequence length
    :type cu_seqlens: torch.Tensor
    :param batch_size: batch size
    :type batch_size: int
    :param sequence_length: sequence length
    :type sequence_length: int
    :param padding_side: padding side
    :type padding_side: str
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: list of unpacked tensors
    :rtype: Sequence[Tensor]
    """

    assert padding_side in ["left", "right"]
    assert isinstance(inputs, (list, tuple))

    outputs = []

    for x in inputs:
        assert x.dim() >= 2
        assert cu_seqlens.size(0) - 1 == batch_size

        x = _UnpackSequence.run(
            x=x,
            cu_seqlens=cu_seqlens,
            output_shape=(batch_size, sequence_length, *x.size()[1:]),
            padding_side=padding_side,
            kernel_backend=kernel_backend,
        )

        outputs.append(x)

    return outputs
