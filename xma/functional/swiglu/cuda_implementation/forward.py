# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import math
from functools import partial

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Float32

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
    def compute(self, xs: list[cute.Tensor]) -> list[cute.Tensor]:
        x = xs[0]
        dtype = x.dtype

        y = cute.make_rmem_tensor((*x.shape[:-1], x.shape[-1] >> 1), Float32)

        for i in cute.size(x, mode=[0]):
            for j in cute.size(x, mode=[1]):
                g = x[i, j]
                u = x[i, j + 1]

                g = g.to(Float32)
                y[i, j] = u * g * sigmoid(g)

        return (y.to(dtype),)


@xma_op(mutates_args={"y"})
def _swiglu_forward_cuda(g: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> None:
    N = g.size(1)
    div = math.gcd(16 // g.dtype.itemsize, N)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    fn = get_compiled_elementwise_cuda_kernel(
        caller_op=_swiglu_forward_cuda,
        key=(g.dtype, div),
        kernel_class=partial(_SwiGLUForwardCUDAKernel, BLOCK_SIZE=256),
        example_tensors_list=((g, u), (y,)),
        div=div,
        stream=stream,
    )

    fn((g, u), (y,), stream)
