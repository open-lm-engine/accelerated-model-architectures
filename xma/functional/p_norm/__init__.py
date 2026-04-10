# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...accelerator import KernelBackend
from ...custom_op import CustomOp, ctx_save_for_backward
from ...utils import is_triton_available


if is_triton_available():
    from .triton_implementation import _p_norm_forward_triton


class _P_Norm(CustomOp):
    @staticmethod
    def forward_backward_torch(
        x: torch.Tensor, multiplier: float | None, p: int | str, output_dtype: torch.dtype
    ) -> torch.Tensor:
        if multiplier not in [None, 1]:
            x = x * multiplier

        if p == "inf":
            y = x.abs().max(dim=-1)[0]
        else:
            y = torch.norm(x, p=p, dim=-1)

        y = y.to(output_dtype)

        return y

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        multiplier: float | None,
        p: int | str,
        output_dtype: torch.dtype,
        memory_efficient: bool,
        kernel_backend: KernelBackend,
    ) -> torch.Tensor:
        B = x.size(0)
        is_p_inf = p == "inf"

        y = torch.empty(B, device=x.device, dtype=output_dtype)

        if kernel_backend in [KernelBackend.cuda, KernelBackend.triton]:
            _p_norm_forward_triton(x=x, y=y, multiplier=multiplier, p=None if is_p_inf else p, is_p_inf=is_p_inf)
        else:
            raise NotImplementedError(f"unexpected kernel_backend ({kernel_backend})")

        if not memory_efficient:
            ctx_save_for_backward(ctx, x)

        ctx.memory_efficient = memory_efficient

        return y

    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
        memory_efficient = ctx.memory_efficient

        if not memory_efficient:
            x = ctx.saved_tensors[0]

        return dx, [None] * 5


def p_norm(
    x: torch.Tensor,
    multiplier: float | None = None,
    p: int | str = 2,
    output_dtype: torch.dtype = torch.float32,
    *,
    kernel_backend: KernelBackend | None = None,
) -> torch.Tensor:
    """
    computes norm of a vector

    :param x: input activation
    :type x: torch.Tensor
    :param multiplier: if not None, pre-multiplies `x` with `multiplier`. Defaults to None.
    :type multiplier: float | None
    :param p: norm type. can be integer >= 1 or `inf`
    :type p: int | str
    :param output_dtype: output dtype
    :type output_dtype: torch.dtype
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output activation
    :rtype: Tensor
    """

    assert x.dim() == 2

    return _P_Norm.run(x=x, multiplier=multiplier, p=p, output_dtype=output_dtype, kernel_backend=kernel_backend)
