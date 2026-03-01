# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...accelerator import Accelerator, KernelBackend
from ...custom_op import CustomOp, ctx_save_for_backward
from ...utils import empty_like_contiguous, is_triton_available


if is_triton_available():
    from .triton_implementation import pnorm_backward_triton, pnorm_forward_triton


class _PNorm(CustomOp):
    @staticmethod
    def forward_backward_torch(x: torch.Tensor, multiplier: float | None, p: int) -> torch.Tensor:
        if multiplier not in [None, 1]:
            x = x * multiplier

        x = torch.norm(x, p=p, dim=-1)

        return x

    @staticmethod
    def forward(ctx, x: torch.Tensor, multiplier: float | None, p: int, kernel_backend: KernelBackend) -> torch.Tensor:
        assert kernel_backend in [KernelBackend.cuda, KernelBackend.triton]

        B = x.size(0)

        if p == "inf":
            indices = torch.empty(B, device=x.device, dtype=torch.int32)
            y = torch.empty(B, device=x.device, dtype=x.dtype)
        else:
            indices = None
            y = torch.empty(B, device=x.device, dtype=torch.float32)

        pnorm_forward_triton(x=x, y=y, p=p)

        ctx_save_for_backward(ctx, x, y, indices)
        ctx.multiplier = multiplier
        ctx.p = p

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        x, y, indices = ctx.saved_tensors
        dx = empty_like_contiguous(x)

        pnorm_backward_triton(
            x=x,
            y=y,
            dy=dy,
            dx=dx,
            multiplier=ctx.multiplier,
            p=ctx.p,
        )

        return dx, None, None


def p_norm(
    x: torch.Tensor, multiplier: float | None = None, p: int | str = 2, *, kernel_backend: KernelBackend | None = None
) -> torch.Tensor:
    """
    fused residual add RMSNorm computation

    :param x: input activation
    :type x: torch.Tensor
    :param multiplier: if not None, pre-multiplies `x` with `multiplier`. Defaults to None.
    :type multiplier: float | None
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output activation
    :rtype: Tensor
    """

    assert x.dim() == 2

    x = _PNorm.run(x=x, multiplier=multiplier, p=p, kernel_backend=kernel_backend)

    return x
