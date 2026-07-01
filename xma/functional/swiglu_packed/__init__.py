# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ...accelerator import KernelBackend
from ...custom_op import CustomOp, ctx_save_for_backward
from ...math import divide_if_divisible
from ...utils import empty_like_contiguous
from ..swiglu import _FUNCTIONS, swiglu


class _SwigluPacked(CustomOp):
    @staticmethod
    def forward_backward_torch(x: torch.Tensor) -> torch.Tensor:
        up, gate = x.chunk(2, dim=-1)
        return swiglu(gate=gate, up=up, kernel_backend=KernelBackend.torch)

    @staticmethod
    def forward(ctx, x: torch.Tensor, kernel_backend: KernelBackend) -> torch.Tensor:
        assert kernel_backend == KernelBackend.cuda
        ctx.kernel_backend = kernel_backend

        ctx_save_for_backward(ctx, x)

        u, g = x.chunk(2, dim=-1)
        y = torch.empty(*x.size()[:-1], divide_if_divisible(x.size(-1), 2), device=x.device, dtype=x.dtype)

        forward_function, backward_function = _FUNCTIONS[kernel_backend]
        forward_function(g=g, u=u, y=y)
        ctx.backward_function = backward_function

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]

        dy = dy.contiguous()
        u, g = x.chunk(2, dim=-1)

        dx = empty_like_contiguous(x)
        du, dg = dx.chunk(2, dim=-1)
        ctx.backward_function(g=g, u=u, dy=dy, dg=dg, du=du)

        return dx, None


def swiglu_packed(x: torch.Tensor, *, kernel_backend: KernelBackend | None = None) -> torch.Tensor:
    """
    computes swiglu activation by splitting the tensor `x` into 2 parts: gate and up activations

    :param x: input activation
    :type x: torch.Tensor
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output tensor
    :rtype: Tensor
    """

    original_shape = x.size()
    x = x.flatten(0, -2)

    H = divide_if_divisible(original_shape[-1], 2)

    y = _SwigluPacked.run(x=x, kernel_backend=kernel_backend)
    y = y.view(*original_shape[:-1], H)

    return y
