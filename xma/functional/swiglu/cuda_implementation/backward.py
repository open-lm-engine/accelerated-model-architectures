# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import math
from functools import partial

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Float32, const_expr, range_constexpr

from ....custom_op import xma_op
from ....cute_dsl_utils import ElementwiseCUDAKernel, get_compiled_elementwise_cuda_kernel, sigmoid


class _SwiGLUBackwardCUDAKernel(ElementwiseCUDAKernel):
    def compute(self, xs: list[cute.TensorSSA]) -> list[cute.TensorSSA]:
        g, u, dy = xs

        dtype = g.dtype
        g = g.to(Float32)

        g_sigmoid = sigmoid(g)
        g_silu = g * g_sigmoid

        dg = dy * u * (g_sigmoid + g_silu * (1 - g_sigmoid))
        du = dy * g_silu

        return dg.to(dtype), du.to(dtype)


class _SwiGLUBackwardPackedCUDAKernel(ElementwiseCUDAKernel):
    @cute.jit
    def compute(self, xs_1: list[cute.Tensor], xs_2: list[cute.Tensor]) -> tuple[list[cute.Tensor], list[cute.Tensor]]:
        assert const_expr(len(xs_1) == 1)
        assert const_expr(len(xs_2) == 1)

        dy = xs_1[0]
        x = xs_2[0]
        dtype = x.dtype

        N = cute.size(x.shape)
        H = N >> 1

        dx = cute.make_rmem_tensor(H, Float32)

        for j in range_constexpr(H):
            h = j << 1

            g = x[h].to(Float32)
            u = x[h + 1]

            g_sigmoid = sigmoid(g)
            g_silu = g * g_sigmoid

            dx[h] = dy * u * (g_sigmoid + g_silu * (1 - g_sigmoid))
            dx[h + 1] = dy * g_silu

        return [], [dx.load().to(dtype)]


@xma_op(mutates_args={"dg", "du"})
def _swiglu_backward_cuda(
    g: torch.Tensor, u: torch.Tensor, dy: torch.Tensor, dg: torch.Tensor, du: torch.Tensor
) -> None:
    N = g.size(1)
    div = math.gcd(16 // g.dtype.itemsize, N)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    kernel = get_compiled_elementwise_cuda_kernel(
        caller_op=_swiglu_backward_cuda,
        key=(g.dtype, div),
        kernel_class=partial(_SwiGLUBackwardCUDAKernel, BLOCK_SIZE=256),
        example_tensors_list=([g, u, dy], [dg, du]),
        div=div,
        stream=stream,
    )

    kernel([g, u, dy], [dg, du], stream)


@xma_op(mutates_args={"dx"})
def _swiglu_packed_backward_cuda(x: torch.Tensor, dy: torch.Tensor, dx: torch.Tensor) -> None:
    N = x.size(1) >> 1
    div = math.gcd(8 // x.dtype.itemsize, N)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    kernel = get_compiled_elementwise_cuda_kernel(
        caller_op=_swiglu_packed_backward_cuda,
        key=(x.dtype, div),
        kernel_class=partial(_SwiGLUBackwardPackedCUDAKernel, BLOCK_SIZE=256),
        example_tensors_list=([dy], [x], [], [dx]),
        div=div,
        stream=stream,
    )

    kernel([dy], [x], [], [dx], stream)
