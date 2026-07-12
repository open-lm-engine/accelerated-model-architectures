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
from ....cute_dsl_utils import (
    ElementwiseCUDAKernel,
    ElementwisePackedCUDAKernel,
    get_compiled_elementwise_cuda_kernel,
    sigmoid,
)


class _SwiGLUForwardCUDAKernel(ElementwiseCUDAKernel):
    def compute(self, xs: list[cute.TensorSSA]) -> list[cute.TensorSSA]:
        g, u = xs

        dtype = g.dtype
        g = g.to(Float32)
        y = u * g * sigmoid(g)

        return (y.to(dtype),)


class _SwigluPackedForwardCUDAKernel(ElementwisePackedCUDAKernel):
    def compute(self, xs_1: list[cute.Tensor], xs_2: list[cute.Tensor]) -> tuple[list[cute.Tensor], list[cute.Tensor]]:
        assert const_expr(len(xs_1) == 0)
        assert const_expr(len(xs_2) == 1)

        x = xs_2[0]
        dtype = x.dtype

        y = cute.make_rmem_tensor((*x.shape[:-1], x.shape[-1] >> 1), Float32)

        B = cute.size(x, mode=[0])
        H = cute.size(x, mode=[1]) >> 1

        for i in range_constexpr(B):
            for j in range_constexpr(H):
                h = j << 1
                g = x[i, h]
                u = x[i, h + 1]

                g = g.to(Float32)
                y[i, j] = u * g * sigmoid(g)

        return y.to(dtype), None


@xma_op(mutates_args={"y"})
def _swiglu_forward_cuda(g: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> None:
    N = g.size(1)
    div = math.gcd(16 // g.dtype.itemsize, N)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    kernel = get_compiled_elementwise_cuda_kernel(
        caller_op=_swiglu_forward_cuda,
        key=(g.dtype, div),
        kernel_class=partial(_SwiGLUForwardCUDAKernel, BLOCK_SIZE=256),
        example_tensors_list=([g, u], [y]),
        div=div,
        stream=stream,
    )

    kernel([g, u], [y], stream)


@xma_op(mutates_args={"y"})
def _swiglu_packed_forward_cuda(x: torch.Tensor, y: torch.Tensor) -> None:
    N = x.size(1) >> 1
    div = math.gcd(8 // x.dtype.itemsize, N)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    kernel = get_compiled_elementwise_cuda_kernel(
        caller_op=_swiglu_packed_forward_cuda,
        key=(x.dtype, div),
        kernel_class=partial(_SwigluPackedForwardCUDAKernel, BLOCK_SIZE=256),
        example_tensors_list=([], [x], [y], []),
        div=div,
        stream=stream,
    )

    kernel([], [x], [y], [], stream)
