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
from ....cute_dsl_utils import ElementwiseCUDAKernel, get_compiled_elementwise_cuda_kernel, sigmoid


class SwiGLUBackwardCUDAKernel(ElementwiseCUDAKernel):
    def compute(self, xs: list[cute.TensorSSA]) -> list[cute.TensorSSA]:
        g, u, dy = xs

        dtype = g.dtype
        g = g.to(Float32)

        g_sigmoid = sigmoid(g)
        g_silu = g * g_sigmoid

        dg = dy * u * (g_sigmoid + g_silu * (1 - g_sigmoid))
        du = dy * g_silu

        return dg.to(dtype), du.to(dtype)


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
        kernel_class=partial(SwiGLUBackwardCUDAKernel, BLOCK_SIZE=256),
        example_tensors_list=([g, u, dy], [dg, du]),
        div=div,
        stream=stream,
    )

    kernel((g, u, dy), (dg, du), stream)
