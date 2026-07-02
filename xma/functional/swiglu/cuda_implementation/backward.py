# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import torch
from cutlass import Float32

from ....custom_op import xma_op
from ....cute_dsl_utils import ElementwiseUnpackedCUDAKernel, get_compiled_elementwise_cuda_fn, sigmoid


class SwiGLUBackwardCUDAKernel(ElementwiseUnpackedCUDAKernel):
    def compute(self, g, u, dy):
        dtype = g.dtype
        g = g.to(Float32)

        g_sigmoid = sigmoid(g)
        g_silu = g * g_sigmoid

        dg = dy * u * (g_sigmoid + g_silu * (1 - g_sigmoid))
        du = dy * g_silu

        return dg.to(dtype), du.to(dtype)


_CACHE = {}


@xma_op(mutates_args={"dg", "du"})
def _swiglu_backward_cuda(
    g: torch.Tensor, u: torch.Tensor, dy: torch.Tensor, dg: torch.Tensor, du: torch.Tensor
) -> None:
    N = g.size(1)
    div = math.gcd(16 // g.dtype.itemsize, N)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn = get_compiled_elementwise_cuda_fn(_CACHE, (g.dtype, div), SwiGLUBackwardCUDAKernel, (g, u, dy, dg, du), div)
    fn(g, u, dy, dg, du, stream)
