# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...accelerator import KernelBackend
from ...custom_op import CustomOp, ctx_save_for_backward
from ...math import divide_if_divisible
from ...utils import (
    empty_like_contiguous,
    is_cute_dsl_available,
    is_torch_neuronx_available,
    is_torch_xla_available,
    is_triton_available,
)
from .mps_implementation import _swiglu_backward_mps, _swiglu_forward_mps


_FUNCTIONS = {KernelBackend.mps: (_swiglu_forward_mps, _swiglu_backward_mps)}


if is_cute_dsl_available():
    from .cuda_implementation import _swiglu_backward_cuda, _swiglu_forward_cuda

    _FUNCTIONS[KernelBackend.cuda] = (_swiglu_forward_cuda, _swiglu_backward_cuda)

if is_torch_neuronx_available():
    from .nki_implementation import _swiglu_backward_nki, _swiglu_forward_nki

    _FUNCTIONS[KernelBackend.nki] = (_swiglu_forward_nki, _swiglu_backward_nki)

if is_torch_xla_available():
    from .pallas_implementation import _swiglu_backward_pallas, _swiglu_forward_pallas

if is_triton_available():
    from .triton_implementation import _swiglu_backward_triton, _swiglu_forward_triton

    _FUNCTIONS[KernelBackend.triton] = (_swiglu_forward_triton, _swiglu_backward_triton)


class _Swiglu(CustomOp):
    @staticmethod
    def forward_backward_torch(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        dtype = g.dtype

        g = g.float()
        u = u.float()

        y = u * F.silu(g)
        y = y.to(dtype)

        return y

    @staticmethod
    def forward(ctx, g: torch.Tensor, u: torch.Tensor, kernel_backend: KernelBackend) -> torch.Tensor:
        ctx.kernel_backend = kernel_backend

        if kernel_backend in [KernelBackend.cuda, KernelBackend.pallas]:
            g = g.contiguous()
            u = u.contiguous()

        ctx_save_for_backward(ctx, g, u)

        if kernel_backend == KernelBackend.pallas:
            return _swiglu_forward_pallas(g=g, u=u)

        y = empty_like_contiguous(g)

        forward_function, backward_function = _FUNCTIONS[kernel_backend]
        forward_function(g=g, u=u, y=y)
        ctx.backward_function = backward_function

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g, u = ctx.saved_tensors
        kernel_backend = ctx.kernel_backend

        if kernel_backend in [KernelBackend.cuda, KernelBackend.pallas]:
            dy = dy.contiguous()

        if kernel_backend == KernelBackend.pallas:
            dg, du = _swiglu_backward_pallas(g=g, u=u, dy=dy)
            return dg, du, None

        dg = empty_like_contiguous(g)
        du = empty_like_contiguous(u)

        ctx.backward_function(g=g, u=u, dy=dy, dg=dg, du=du)

        return dg, du, None


class _SwigluPacked(CustomOp):
    @staticmethod
    def forward_backward_torch(x: torch.Tensor) -> torch.Tensor:
        dtype = x, dtype

        u = x[..., 1::2]
        g = x[..., ::2]

        x = u.float() * F.silu(g.float())

        return x.to(dtype)

    @staticmethod
    def forward(ctx, x: torch.Tensor, kernel_backend: KernelBackend) -> torch.Tensor:
        if kernel_backend != KernelBackend.cuda:
            raise NotImplementedError

        ctx_save_for_backward(ctx, x)

        y = torch.empty(*x.size()[:-1], divide_if_divisible(x.size(-1), 2), device=x.device, dtype=x.dtype)

        forward_function(g=g, u=u, y=y)

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]
        u, g = x.chunk(2, dim=-1)

        dx = empty_like_contiguous(x)
        du, dg = dx.chunk(2, dim=-1)
        backward_function(g=g, u=u, dy=dy, dg=dg, du=du)

        return dx, None


def swiglu(gate: torch.Tensor, up: torch.Tensor, *, kernel_backend: KernelBackend | None = None) -> torch.Tensor:
    """
    computes swiglu activation as `up * gate * sigmoid(gate)`

    :param gate: `gate` activation tensor
    :type gate: torch.Tensor
    :param up: `up` activation tensor
    :type up: torch.Tensor
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output tensor
    :rtype: Tensor
    """

    assert gate.size() == up.size(), "tensors gate and up should have same shape"
    assert gate.type() == up.type(), "tensors gate and up should have same dtype"

    original_shape = gate.size()
    gate = gate.flatten(0, -2)
    up = up.flatten(0, -2)

    y = _Swiglu.run(g=gate, u=up, kernel_backend=kernel_backend)
    y = y.view(original_shape)

    return y


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
