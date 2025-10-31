# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F

from ...custom_op import CustomOp, ctx_save_for_backward
from ...enums import KernelBackend
from ...math import divide_if_divisible
from ...utils import empty_like_contiguous, ensure_contiguous, is_cute_dsl_available, is_triton_available
from .cuda_implementation import swiglu_backward_cuda, swiglu_forward_cuda


if is_cute_dsl_available():
    from .cuda_implementation import swiglu_forward_cuda


if is_triton_available():
    from .triton_implementation import swiglu_backward_triton, swiglu_forward_triton


class _Swiglu(CustomOp):
    @staticmethod
    def forward_backward_torch(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        dtype = gate.dtype

        gate = gate.float()
        up = up.float()

        output = up * F.silu(gate)
        output = output.to(dtype)

        return output

    @staticmethod
    @ensure_contiguous
    def forward_cuda(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        output = empty_like_contiguous(gate)
        swiglu_forward_cuda(gate=gate, up=up, output=output, BLOCK_SIZE=1024)

        ctx_save_for_backward(ctx, gate, up)

        return output

    @staticmethod
    @ensure_contiguous
    def backward_cuda(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate, up = ctx.saved_tensors
        gate_grad = empty_like_contiguous(gate)
        up_grad = empty_like_contiguous(up)

        swiglu_backward_cuda(
            gate=gate, up=up, output_grad=output_grad, gate_grad=gate_grad, up_grad=up_grad, BLOCK_SIZE=1024
        )

        return gate_grad, up_grad

    @staticmethod
    @ensure_contiguous
    def forward_triton(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        output = empty_like_contiguous(gate)
        swiglu_forward_triton(gate=gate, up=up, output=output)

        ctx_save_for_backward(ctx, gate, up)

        return output

    @staticmethod
    @ensure_contiguous
    def backward_triton(ctx, output_grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate, up = ctx.saved_tensors
        gate_grad = empty_like_contiguous(gate)
        up_grad = empty_like_contiguous(up)

        swiglu_backward_triton(gate=gate, up=up, output_grad=output_grad, gate_grad=gate_grad, up_grad=up_grad)

        return gate_grad, up_grad


class _SwigluPacked(CustomOp):
    @staticmethod
    def forward_backward_torch(x: torch.Tensor) -> torch.Tensor:
        up, gate = x.chunk(2, dim=-1)
        return swiglu(gate=gate, up=up, kernel_backend=KernelBackend.torch)

    @staticmethod
    @ensure_contiguous
    def forward_triton(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx_save_for_backward(ctx, x)

        output = torch.empty(*x.size()[:-1], divide_if_divisible(x.size(-1), 2), device=x.device, dtype=x.dtype)
        up, gate = x.chunk(2, dim=-1)

        swiglu_forward_triton(gate=gate, up=up, output=output)

        return output

    @staticmethod
    @ensure_contiguous
    def backward_triton(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        x = ctx.saved_tensors[0]
        x_grad = empty_like_contiguous(x)

        up, gate = x.chunk(2, dim=-1)
        up_grad, gate_grad = x_grad.chunk(2, dim=-1)

        swiglu_backward_triton(gate=gate, up=up, output_grad=output_grad, gate_grad=gate_grad, up_grad=up_grad)

        return x_grad


def swiglu(gate: torch.Tensor, up: torch.Tensor, *, kernel_backend: KernelBackend | None = None) -> torch.Tensor:
    """computes swiglu activation as `up` * `gate` * sigmoid(`gate`)

    Args:
        gate (torch.Tensor): `gate` activation tensor
        up (torch.Tensor): `up` activation tensor

    Returns:
        torch.Tensor: output tensor
    """

    assert gate.size() == up.size(), "tensors gate and up should have same shape"
    assert gate.type() == up.type(), "tensors gate and up should have same dtype"

    return _Swiglu.run(gate=gate, up=up, kernel_backend=kernel_backend)


def swiglu_packed(x: torch.Tensor, *, kernel_backend: KernelBackend | None = None) -> torch.Tensor:
    """computes swiglu activation by splitting the tensor `x` into 2 parts: gate and up activations

    Args:
        x (torch.Tensor): input activation

    Returns:
        torch.Tensor: output tensor
    """

    return _SwigluPacked.run(x=x, kernel_backend=kernel_backend)
