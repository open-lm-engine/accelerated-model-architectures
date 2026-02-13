# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ..accelerator import KernelBackend
from ..custom_op import CustomOp, ctx_needs_gradients, ctx_save_for_backward
from ..math import ceil_divide, get_next_power_of_2
from ..utils import empty_like_contiguous, is_triton_available, zeros_like_contiguous
from .cross_entropy import cross_entropy


if is_triton_available():
    from .cross_entropy import cross_entropy_forward_backward_triton


class _FusedLinearCrossEntropy(CustomOp):
    def forward_backward_torch(
        x: torch.Tensor,
        W: torch.Tensor,
        y: torch.Tensor,
        reduction: str,
        logits_multiplier: float | None,
    ) -> torch.Tensor:
        x = F.linear(x, W)
        l = cross_entropy(
            x=x, labels=y, reduction=reduction, logits_multiplier=logits_multiplier, kernel_backend=KernelBackend.torch
        )

        return l

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        W: torch.Tensor,
        y: torch.Tensor,
        reduction: str,
        logits_multiplier: float | None,
        kernel_backend: KernelBackend,
    ) -> torch.Tensor:
        ctx.kernel_backend = kernel_backend

        if kernel_backend not in [KernelBackend.cuda, KernelBackend.rocm, KernelBackend.triton]:
            raise NotImplementedError

        B, H = x.size()
        V = W.size(0)

        # NOTE chunking is copied from liger kernel
        memory_increase_factor = ceil_divide(V, H)
        # chunk_size needed to reduce memory increase back to 1
        chunk_size = get_next_power_of_2(ceil_divide(B, memory_increase_factor))
        num_chunks = ceil_divide(B, chunk_size)

        l = torch.zeros((), device=x.device, dtype=torch.float32)

        needs_grad = ctx_needs_gradients(ctx)
        dx = empty_like_contiguous(x) if needs_grad else None
        dW = zeros_like_contiguous(W) if needs_grad else None

        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            end = min(end, B)

            _x = x[start:end]
            _h = _x @ W.T

            _dh = empty_like_contiguous(_h)
            _y = y[start:end]

            cross_entropy_forward_backward_triton(
                x=_h, labels=_y, loss=l, x_grad=_dh, logits_multiplier=logits_multiplier, reduction="sum"
            )

            if needs_grad:
                dx[start:end] = _dh @ W
                torch.addmm(dW, _dh.T, _x, alpha=1, beta=1, out=dW)

        if reduction == "mean":
            l /= B
            dx /= B
            dW /= B

        ctx_save_for_backward(ctx, dx, dW)

        return l

    @staticmethod
    def backward(ctx, dl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, None, None, None, None]:
        dx, dW = ctx.saved_tensors

        dx *= dl
        dW *= dl

        return dx, dW, None, None, None, None


def fused_linear_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
    logits_multiplier: float | None = None,
    *,
    kernel_backend: KernelBackend | None = None,
) -> torch.Tensor:
    """
    compute cross entropy loss without materializing the full output logits matrix

    :param x: logits
    :type x: torch.Tensor
    :param weight: vocab weight
    :type weight: torch.Tensor
    :param labels: labels
    :type labels: torch.Tensor
    :param reduction: reduction should be either sum or mean. Defaults to "mean".
    :type reduction: str
    :param logits_multiplier: logits multiplier pre-multiplies logits, None implies 1.
        Defaults to None.
    :type logits_multiplier: float | None
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: loss
    :rtype: Tensor
    """

    assert reduction in ["sum", "mean"]
    assert x.dim() == 2, "x should be 2 dimensional"
    assert labels.dim() == 1, "labels should be 1 dimensional"
    assert x.size(0) == labels.size(0), "x and labels have different number of elements along dim 0"
    assert x.size(-1) == weight.size(-1)

    x = _FusedLinearCrossEntropy.run(
        x=x,
        W=weight,
        y=labels,
        reduction=reduction,
        logits_multiplier=logits_multiplier,
        kernel_backend=kernel_backend,
    )

    return x
