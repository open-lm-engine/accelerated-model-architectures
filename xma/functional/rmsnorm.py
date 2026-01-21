# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ..accelerator import KernelBackend
from .fused_residual_add_rmsnorm import fused_residual_add_rmsnorm
from .fused_embedding_residual_add_rmsnorm import fused_embedding_residual_add_rmsnorm

def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float | None = None,
    memory_efficient: bool = False,
    deterministic: bool = False,
    *,
    kernel_backend: KernelBackend | None = None,
) -> torch.Tensor:
    """
    RMSNorm computation

    :param x: input activation
    :type x: torch.Tensor
    :param weight: RMSNorm weight
    :type weight: torch.Tensor | None
    :param eps: epsilon. Defaults to None.
    :type eps: float | None
    :param memory_efficient: memory efficient = False caches RMSNorm's denominator in the forward.
        Defaults to False.
    :type memory_efficient: bool
    :param deterministic: whether to use deterministic backward. Defaults to False.
    :type deterministic: bool
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output tensor
    :rtype: Tensor
    """

    x, _ = fused_residual_add_rmsnorm(
        x=x,
        residual=None,
        weight=weight,
        eps=eps,
        multiplier=None,
        memory_efficient=memory_efficient,
        deterministic=deterministic,
        kernel_backend=kernel_backend,
    )

    return x

def fused_embedding_rmsnorm(
    x: torch.Tensor,
    weight1: torch.Tensor | None,
    weight2: torch.Tensor | None,
    eps: float | None = None,
    memory_efficient: bool = False,
    deterministic: bool = False,
    *,
    kernel_backend: KernelBackend | None = None,
) -> torch.Tensor:
    """
    Fused embedding RMSNorm computation
    """

    x, _ = fused_embedding_residual_add_rmsnorm(
        x=x,
        residual=None,
        weight1=weight1,
        weight2=weight2,
        eps=eps,
        multiplier=None,
        memory_efficient=memory_efficient,
        deterministic=deterministic,
        kernel_backend=kernel_backend,
    )

    return x
