# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...accelerator import Accelerator, KernelBackend
from ...custom_op import CustomOp, ctx_needs_gradients, ctx_save_for_backward
from ...utils import empty_like_contiguous, is_triton_available, zeros_like_contiguous


if is_triton_available():
    from .triton_implementation import (
        fused_residual_add_rmsnorm_backward_triton,
        fused_residual_add_rmsnorm_forward_triton,
    )


class _FusedResidualAddRMSNorm(CustomOp):
    @staticmethod
    def forward_backward_torch(
        x: torch.Tensor,
        r: torch.Tensor | None,
        W: torch.Tensor | None,
        eps: float | None,
        multiplier: float | None,
        memory_efficient: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if multiplier not in [None, 1]:
            x = x * multiplier

        if r is not None:
            x = x + r
            r = x

        x = F.rms_norm(x, normalized_shape=(x.size(-1),), weight=W, eps=eps)

        return x, r

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        r: torch.Tensor | None,
        W: torch.Tensor | None,
        eps: float | None,
        multiplier: float | None,
        memory_efficient: bool,
        kernel_backend: KernelBackend,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kernel_backend in [KernelBackend.cuda, KernelBackend.triton]

        if eps is None:
            eps = torch.finfo(x.dtype).eps

        B = x.size(0)
        has_residual = r is not None

        y = empty_like_contiguous(x)
        xr = empty_like_contiguous(x) if has_residual else None

        s = None
        if ctx_needs_gradients(ctx) and not memory_efficient:
            s = torch.empty(B, device=x.device, dtype=torch.float32)

        fused_residual_add_rmsnorm_forward_triton(x=x, r=r, W=W, y=y, eps=eps, multiplier=multiplier, xr=xr, s=s)

        ctx_save_for_backward(ctx, xr if has_residual else x, W, s)
        ctx.eps = eps
        ctx.has_residual = has_residual
        ctx.multiplier = multiplier

        return y, xr

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


def fused_residual_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor | None,
    eps: float | None,
    multiplier: float | None = None,
    memory_efficient: bool = False,
    *,
    kernel_backend: KernelBackend | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    fused residual add RMSNorm computation

    :param x: input activation
    :type x: torch.Tensor
    :param residual: residual activation
    :type residual: torch.Tensor | None
    :param weight: RMSNorm weight
    :type weight: torch.Tensor | None
    :param eps: epsilon
    :type eps: float | None
    :param multiplier: if not None, pre-multiplies `x` with `multiplier`. Defaults to None.
    :type multiplier: float | None
    :param memory_efficient: memory efficient = False caches RMSNorm's denominator in the forward. Defaults to False.
    :type memory_efficient: bool
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output activations and updated residual stream
    :rtype: tuple[Tensor, Tensor | None]
    """

    if weight is not None:
        assert weight.dim() == 1, "weight should be 1D"
        assert weight.size(-1) == x.size(-1), "hidden size for x and weight tensor is different"
        assert weight.type() == x.type(), "tensors weight and y should have same dtype"

    x, residual = _FusedResidualAddRMSNorm.run(
        x=x,
        r=residual,
        W=weight,
        eps=eps,
        multiplier=multiplier,
        memory_efficient=memory_efficient,
        kernel_backend=kernel_backend,
    )

    return x, residual
