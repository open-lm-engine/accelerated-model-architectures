# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...accelerator import KernelBackend
from ...custom_op import CustomOp, ctx_needs_gradients, ctx_save_for_backward
from ...utils import empty_like_contiguous, get_num_elements_and_hidden_size, is_triton_available


if is_triton_available():
    from .triton_implementation import cross_entropy_forward_backward_triton


class _CrossEntropy(CustomOp):
    @staticmethod
    def forward_backward_torch(
        x: torch.Tensor, labels: torch.Tensor, reduction: str = "mean", logits_multiplier: float | None = None
    ) -> torch.Tensor:
        x = x.float()

        if logits_multiplier not in [None, 1]:
            x = x * logits_multiplier

        return F.cross_entropy(x, labels, reduction=reduction)

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        labels: torch.Tensor,
        reduction: str,
        logits_multiplier: float | None,
        kernel_backend: KernelBackend,
    ) -> torch.Tensor:
        assert kernel_backend in [KernelBackend.cuda, KernelBackend.triton]

        loss = torch.zeros((), device=x.device, dtype=torch.float32)
        x_grad = empty_like_contiguous(x) if ctx_needs_gradients(ctx) else None

        cross_entropy_forward_backward_triton(
            x=x, labels=labels, loss=loss, x_grad=x_grad, logits_multiplier=logits_multiplier, reduction=reduction
        )

        ctx_save_for_backward(ctx, x_grad)

        return loss

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        x_grad = ctx.saved_tensors[0]
        x_grad *= output_grad

        return x_grad, *[None] * 4


def cross_entropy(
    x: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
    logits_multiplier: float | None = None,
    *,
    kernel_backend: KernelBackend | None = None,
) -> torch.Tensor:
    """
    cross entropy loss

    :param x: logits
    :type x: torch.Tensor
    :param labels: labels
    :type labels: torch.Tensor
    :param reduction: reduction method: "sum", "mean" or None
    :type reduction: str
    :param logits_multiplier: logits multiplier pre-multiplies logits, None implies 1. Defaults to None.
    :type logits_multiplier: float | None
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: loss
    :rtype: Tensor
    """

    assert reduction in ["sum", "mean"]
    assert x.dim() == 2, "x should be 2 dimensional"
    assert labels.dim() == 1, "labels should be 1 dimensional"
    assert labels.size(0) == x.size(0), "x and labels have different number of elements along batch dimension"

    x = _CrossEntropy.run(
        x=x, labels=labels, reduction=reduction, logits_multiplier=logits_multiplier, kernel_backend=kernel_backend
    )

    return x
