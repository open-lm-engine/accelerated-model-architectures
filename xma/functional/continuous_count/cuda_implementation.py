# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch

import cutlass
import cutlass.cute as cute

from ...custom_op import xma_op
from ...cute_dsl_utils import get_fake_cute_tensor


class _ContinuousCountCUDAKernel:
    def __init__(self, B: int, C: int, BLOCK_SIZE: int = 128) -> _ContinuousCountCUDAKernel:
        self.B = max(B, 4)
        self.C = C
        self.BLOCK_SIZE = BLOCK_SIZE

    @cute.kernel
    def kernel(
        self,
        gX: cute.Tensor,
        gY: cute.Tensor,
        gC: cute.Tensor,
        copy_atom: cute.CopyAtom,
        tiled_copy: cute.TiledCopy,
        shape: cute.Shape,
    ) -> None:
        BLOCK_ID, _, _ = cute.arch.block_idx()
        THREAD_ID, _, _ = cute.arch.thread_idx()
        block_coord = ((None, None), BLOCK_ID)

        bX = gX[block_coord]
        bC = gC[block_coord]

        thr_copy = tiled_copy.get_slice(THREAD_ID)

        tX = thr_copy.partition_S(bX)
        tC = thr_copy.partition_S(bC)

        rX = cute.make_rmem_tensor_like(tX)
        rC = cute.make_rmem_tensor_like(tC, dtype=cutlass.Boolean)
        for i in cutlass.range_constexpr(cute.size(rC)):
            rC[i] = cute.elem_less(tC[i], shape)

        is_within_boundary = cute.elem_less(tC[cute.size(tC) - 1], shape)

        if is_within_boundary:
            cute.copy(copy_atom, tX, rX)
        else:
            cute.copy(copy_atom, tX, rX, pred=rC)

        smem_allocator = cutlass.utils.SmemAllocator()

        sY = smem_allocator.allocate_tensor(
            gY.element_type, layout=cute.make_ordered_layout((1, self.C), order=(1, 0)), byte_alignment=16
        )
        sY.fill(0)

        cute.arch.sync_threads()

        x = rX.load()

        for i in cutlass.range_constexpr(cute.size(rX)):
            q = sY.iterator + x[i]
            if is_within_boundary or rC[i]:
                cute.arch.atomic_add(q, 1, sem="relaxed", scope="cta")

        cute.arch.sync_threads()

    @cute.jit
    def __call__(self, mX: cute.Tensor, mY: cute.Tensor) -> None:
        vector_size = 128 // mX.element_type.width

        thr_layout = cute.make_ordered_layout((1, self.BLOCK_SIZE), order=(1, 0))
        val_layout = cute.make_ordered_layout((self.B, vector_size), order=(1, 0))
        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        mC = cute.make_identity_tensor(mX.shape)

        gX = cute.zipped_divide(mX, tiler_mn)
        gC = cute.zipped_divide(mC, tiler_mn)

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type)
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        NUM_BLOCKS = cute.size(gX, mode=[1])
        kernel = self.kernel(gX=gX, gY=mY, gC=gC, copy_atom=copy_atom, tiled_copy=tiled_copy, shape=mX.shape)
        kernel.launch(grid=(NUM_BLOCKS, 1, 1), block=(self.BLOCK_SIZE, 1, 1))


_CACHE = {}


@xma_op(mutates_args={"y"})
def continuous_count_cuda(x: torch.Tensor, y: torch.Tensor) -> None:
    N = x.numel()
    C = y.numel()

    key = (x.dtype, C)
    function = _CACHE.get(key, None)

    if x.dim() == 1:
        x = x[None, ...]

    B = x.size(0)

    if function is None:
        _x = get_fake_cute_tensor(
            dtype=x.dtype, shape=(B, cute.sym_int()), divisibility=math.gcd(16 // x.dtype.itemsize, N)
        )

        _y = get_fake_cute_tensor(dtype=y.dtype, shape=(C,), divisibility=math.gcd(16 // y.dtype.itemsize, C))

        function = _ContinuousCountCUDAKernel(B=B, C=C)
        function = cute.compile(function, _x, _y, options="--enable-tvm-ffi")
        _CACHE[key] = function

    function(x, y)
