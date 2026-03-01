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

        y = torch.empty(B, device=x.device, dtype=torch.float32)
        pnorm_forward_triton(x=x, y=y, p=p)

        ctx_save_for_backward(ctx, x)
        ctx.multiplier = multiplier

        return y

    @staticmethod
    def backward(
        ctx, dy: torch.Tensor, dxr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, None, None, None, None]:
        has_residual = ctx.has_residual

        xr, W, s = ctx.saved_tensors
        dx = empty_like_contiguous(xr)
        dr = empty_like_contiguous(xr) if has_residual else None

        dW = (
            None
            if W is None
            else torch.empty(Accelerator.get_sm_count(dx.device), *W.size(), dtype=torch.float32, device=dx.device)
        )

        if not has_residual:
            assert dxr is None

        fused_residual_add_rmsnorm_backward_triton(
            xr=xr,
            W=W,
            dy=dy,
            dxr=dxr,
            s=s,
            dx=dx,
            dr=dr,
            dW=dW,
            eps=ctx.eps,
            multiplier=ctx.multiplier,
        )

        if dW is not None:
            dW = dW.sum(0)
            dW = dW.type_as(W)

        return dx, dr, dW, *[None] * 5


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
