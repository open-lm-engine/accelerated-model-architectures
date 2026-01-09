# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...accelerator import KernelBackend
from ...custom_op import CustomOp, ctx_save_for_backward
from ...utils import empty_like_contiguous, is_cute_dsl_available, is_triton_available


if is_cute_dsl_available():
    from .cuda_implementation import softmax_forward_cuda


if is_triton_available():
    from .triton_implementation import softmax_backward_triton, softmax_forward_triton


class _Softmax(CustomOp):
    @staticmethod
    def forward_backward_torch(x: torch.Tensor, logits_multiplier: float | None) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()

        if logits_multiplier is not None:
            x = x * logits_multiplier

        x = F.softmax(x, dim=-1)
        x = x.to(dtype)

        return x

    @staticmethod
    def forward(ctx, x: torch.Tensor, logits_multiplier: float | None, kernel_backend: KernelBackend) -> torch.Tensor:
        y = empty_like_contiguous(x)

        if kernel_backend == KernelBackend.cuda:
            x = x.contiguous()
            softmax_forward_cuda(x=x, y=y, logits_multiplier=logits_multiplier)
        elif kernel_backend == KernelBackend.triton:
            softmax_forward_triton(x=x, y=y, logits_multiplier=logits_multiplier)
        else:
            raise NotImplementedError

        ctx_save_for_backward(ctx, y)
        ctx.logits_multiplier = logits_multiplier

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor | None]:
        y = ctx.saved_tensors[0]
        dx = empty_like_contiguous(y)

        softmax_backward_triton(y=y, dy=dy, dx=dx, logits_multiplier=ctx.logits_multiplier)

        return dx, None, None


def softmax(
    x: torch.Tensor, logits_multiplier: float | None = None, *, kernel_backend: KernelBackend | None = None
) -> torch.Tensor:
    """
    computes softmax activation

    :param x: input activation tensor
    :type x: torch.Tensor
    :param logits_multiplier: pre-multiplies `x` with `logits_multiplier` before computing softmax.
        Defaults to None.
    :type logits_multiplier: float | None
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output tensor
    :rtype: Tensor
    """

    # if 1D -> make 2D
    is_flat = x.dim() == 1
    if is_flat:
        x = x[None, ...]

    x = _Softmax.run(x=x, logits_multiplier=logits_multiplier, kernel_backend=kernel_backend)

    # convert back to 1D
    if is_flat:
        x = x.squeeze(0)

    return x
