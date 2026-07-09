# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Float32

from ....custom_op import xma_op
from ....cute_dsl_utils import sigmoid
from ....cute_dsl_utils.elementwise import ElementwiseCUDAKernel, get_compiled_elementwise_cuda_fn


class SwiGLUForwardCUDAKernel(ElementwiseCUDAKernel):
    def compute(self, xs: list[cute.TensorSSA]) -> list[cute.TensorSSA]:
        g, u = xs

        dtype = g.dtype
        g = g.to(Float32)
        y = u * g * sigmoid(g)

        return (y.to(dtype),)


_CACHE = {}


@xma_op(mutates_args={"y"})
def _swiglu_forward_cuda(g: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> None:
    N = g.size(1)
    div = math.gcd(16 // g.dtype.itemsize, N)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn = get_compiled_elementwise_cuda_fn(_CACHE, (g.dtype, div), SwiGLUForwardCUDAKernel, ((g, u), (y,)), div)
    fn((g, u), (y,), stream)
