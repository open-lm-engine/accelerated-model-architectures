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
        fused_embedding_residual_add_rmsnorm_backward_triton,
        fused_embedding_residual_add_rmsnorm_forward_triton,
    )


class _FusedEmbeddingResidualAddRMSNorm(CustomOp):
    @staticmethod
    def forward_backward_torch(
        x: torch.Tensor,
        r: torch.Tensor | None,
        W1: torch.Tensor | None,
        W2: torch.Tensor | None,
        eps: float | None,
        multiplier: float | None,
        memory_efficient: bool,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if W1 is not None:
            x = W1[x]

        if multiplier not in [None, 1]:
            x = x * multiplier

        if r is not None:
            x = x + r
            r = x

        x = F.rms_norm(x, normalized_shape=(x.size(-1),), weight=W2, eps=eps)

        return x, r

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        r: torch.Tensor | None,
        W1: torch.Tensor | None,
        W2: torch.Tensor | None,
        eps: float | None,
        multiplier: float | None,
        memory_efficient: bool,
        deterministic: bool,
        kernel_backend: KernelBackend,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kernel_backend in [KernelBackend.cuda, KernelBackend.triton]

        if eps is None:
            eps = torch.finfo(x.dtype).eps

        B = x.numel()
        H = W1.size(-1)
        has_residual = r is not None

        y = torch.empty(B, H, device=x.device, dtype=W1.dtype)
        xr = None

        s = None
        if ctx_needs_gradients(ctx) and not memory_efficient:
            s = torch.empty(B, device=x.device, dtype=torch.float32)

        fused_embedding_residual_add_rmsnorm_forward_triton(x=x, r=r, W1=W1, W2=W2, y=y, eps=eps, multiplier=multiplier, xr=xr, s=s)

        ctx_save_for_backward(ctx, xr if has_residual else x, W1, W2, s)
        ctx.eps = eps
        ctx.has_residual = has_residual
        ctx.multiplier = multiplier
        ctx.deterministic = deterministic

        return y, xr

    @staticmethod
    def backward(
        ctx, dy: torch.Tensor, dxr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, None, None, None, None]:
        has_residual = ctx.has_residual
        deterministic = ctx.deterministic

        xr, W, s = ctx.saved_tensors
        dx = empty_like_contiguous(xr)
        dr = empty_like_contiguous(xr) if has_residual else None

        if W is None:
            dW = None
        elif deterministic:
            dW = torch.empty(Accelerator.get_sm_count(dx.device), *W.size(), dtype=torch.float32, device=dx.device)
        else:
            dW = zeros_like_contiguous(W, dtype=torch.float32)

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
            deterministic=deterministic,
        )

        if dW is not None:
            if deterministic:
                dW = dW.sum(0)

            dW = dW.type_as(W)

        return dx, dr, dW, *[None] * 5


def fused_embedding_residual_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight1: torch.Tensor | None,
    weight2: torch.Tensor | None,
    eps: float | None,
    multiplier: float | None = None,
    memory_efficient: bool = False,
    deterministic: bool = False,
    *,
    kernel_backend: KernelBackend | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    fused residual add RMSNorm computation

    :param x: input activation
    :type x: torch.Tensor
    :param residual: residual activation
    :type residual: torch.Tensor | None
    :param weight1: Embedding weight
    :type weight1: torch.Tensor | None
    :param weight2: RMSNorm weight
    :type weight2: torch.Tensor | None
    :param eps: epsilon
    :type eps: float | None
    :param multiplier: if not None, pre-multiplies `x` with `multiplier`. Defaults to None.
    :type multiplier: float | None
    :param memory_efficient: memory efficient = False caches RMSNorm's denominator in the forward. Defaults to False.
    :type memory_efficient: bool
    :param deterministic: whether to use deterministic backward. Defaults to False.
    :type deterministic: bool
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output activations and updated residual stream
    :rtype: tuple[Tensor, Tensor | None]
    """

    if weight1 is not None and weight2 is not None:
        assert weight2.dim() == 1, "weight2 should be 1D"
        assert weight2.size(-1) == weight1.size(-1), "hidden size for weight1 and weight2 is different"
        assert weight2.type() == weight1.type(), "weight1 and weight2 should have same dtype"

    x, residual = _FusedEmbeddingResidualAddRMSNorm.run(
        x=x,
        r=residual,
        W1=weight1,
        W2=weight2,
        eps=eps,
        multiplier=multiplier,
        memory_efficient=memory_efficient,
        deterministic=deterministic,
        kernel_backend=kernel_backend,
    )

    return x, residual
