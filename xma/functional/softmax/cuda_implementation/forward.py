# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************
from __future__ import annotations

import math

import torch

import cutlass.cute as cute
from cutlass import Boolean, Float32, Numeric, const_expr, range_constexpr
from cutlass.utils import SmemAllocator

from ....constants import LOG_WARP_SIZE, WARP_SIZE
from ....custom_op import xma_op
from ....cute_dsl_utils import get_cute_dtype_from_torch_dtype, get_fake_cute_tensor


class SoftmaxForwardCUDAKernel:
    def __init__(
        self, N: int, dtype: type[Numeric], BLOCK_SIZE: int = 128, NUM_THREADS_N: int = 8
    ) -> SoftmaxForwardCUDAKernel:
        self.N = N
        self.dtype = dtype

        self.NUM_THREADS_N = NUM_THREADS_N
        self.NUM_THREADS_M = BLOCK_SIZE // NUM_THREADS_N

        self.BLOCK_SIZE = BLOCK_SIZE
        assert self.BLOCK_SIZE == self.NUM_THREADS_M * self.NUM_THREADS_N

        self.NUM_WARPS = self.BLOCK_SIZE // WARP_SIZE
        self.NUM_WARPS_N = max(self.NUM_THREADS_N // WARP_SIZE, 1)
        self.NUM_WARPS_M = self.NUM_WARPS // self.NUM_THREADS_N

        self.tiler_mn = (self.NUM_THREADS_M, self.N)

    def _get_reduction_buffer_layout(self, tv_layout: cute.Layout) -> cute.Layout:
        return cute.make_ordered_layout((self.NUM_WARPS_M, self.NUM_WARPS_N), (1, 0))

    @cute.kernel
    def kernel(
        self,
        gX: cute.Tensor,
        gY: cute.Tensor,
        logits_multiplier: float | None,
        copy_atom: cute.CopyAtom,
        tiled_copy: cute.TiledCopy,
        shape: cute.Shape,
    ) -> None:
        BLOCK_ID, _, _ = cute.arch.block_idx()
        THREAD_ID, _, _ = cute.arch.thread_idx()

        tv_layout = tiled_copy.layout_tv_tiled
        block_coord = ((None, None), BLOCK_ID)

        cX = cute.make_identity_tensor(shape)

        bX = gX[block_coord]
        bY = gY[block_coord]

        shared_memory = SmemAllocator()

        sX = shared_memory.allocate_tensor(
            gX.element_type, cute.make_ordered_layout(self.tiler_mn, order=(1, 0)), byte_alignment=16
        )

        sR = shared_memory.allocate_tensor(
            Float32, layout=self._get_reduction_buffer_layout(tv_layout), byte_alignment=16
        )

    @cute.jit
    def __call__(self, mX: cute.Tensor, mY: cute.Tensor, logits_multiplier: float | None) -> None:
        vector_size = const_expr(128 // mX.element_type.width)

        thr_layout = cute.make_ordered_layout((self.NUM_THREADS_M, self.NUM_THREADS_N), order=(1, 0))
        val_layout = cute.make_ordered_layout((1, vector_size), order=(1, 0))

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mY.element_type, num_bits_per_copy=128)
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout=thr_layout, val_layout=val_layout)

        gX = cute.zipped_divide(mX, self.tiler_mn)
        gY = cute.zipped_divide(mY, self.tiler_mn)

        NUM_BLOCKS = cute.ceil_div(mX.shape[0], self.tiler_mn[0])

        self.kernel(
            gX=gX,
            gY=gY,
            logits_multiplier=logits_multiplier,
            copy_atom=copy_atom,
            tiled_copy=tiled_copy,
            shape=mX.shape,
        ).launch(grid=(NUM_BLOCKS, 1, 1), block=(self.BLOCK_SIZE, 1, 1))


_CACHE = {}


@xma_op(mutates_args={"y"})
def softmax_forward_cuda(x: torch.Tensor, y: torch.Tensor, logits_multiplier: float | None) -> None:
    N = x.size(1)

    key = (x.dtype, N, logits_multiplier is None)
    function = _CACHE.get(key, None)

    if function is None:
        divisibility = math.gcd(16 // x.dtype.itemsize, N)

        _x, _y = [
            get_fake_cute_tensor(dtype=i.dtype, shape=(cute.sym_int(), N), divisibility=divisibility) for i in (x, y)
        ]

        function = SoftmaxForwardCUDAKernel(N=N, dtype=get_cute_dtype_from_torch_dtype(x.dtype))
        function = cute.compile(function, _x, _y, logits_multiplier, options="--enable-tvm-ffi")
        _CACHE[key] = function

    function(x, y, logits_multiplier)
