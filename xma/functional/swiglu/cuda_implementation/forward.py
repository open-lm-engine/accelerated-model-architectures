# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch

import cutlass.cute as cute
from cutlass import Boolean, Float32, range_constexpr

from ....constants import LOG_WARP_SIZE, WARP_SIZE
from ....custom_op import xma_op
from ....cute_dsl_utils import get_fake_cute_tensor, sigmoid


class SwiGLUForwardCUDAKernel:
    def __init__(self, BLOCK_SIZE: int = 128) -> SwiGLUForwardCUDAKernel:
        self.BLOCK_SIZE = BLOCK_SIZE

    @cute.kernel
    def kernel(
        self,
        gG: cute.Tensor,
        gU: cute.Tensor,
        gY: cute.Tensor,
        gC: cute.Tensor,
        copy_atom: cute.CopyAtom,
        tiled_copy: cute.TiledCopy,
        shape: cute.Shape,
    ) -> None:
        BLOCK_ID, _, _ = cute.arch.block_idx()
        THREAD_ID, _, _ = cute.arch.thread_idx()

        block_coord = ((None, None), BLOCK_ID)

        bG = gG[block_coord]
        bU = gU[block_coord]
        bY = gY[block_coord]
        bC = gC[block_coord]

        thr_copy = tiled_copy.get_slice(THREAD_ID)

        tG = thr_copy.partition_S(bG)
        tU = thr_copy.partition_S(bU)
        tY = thr_copy.partition_D(bY)
        tC = thr_copy.partition_S(bC)

        rG = cute.make_rmem_tensor_like(tG)
        rU = cute.make_rmem_tensor_like(tU)
        rY = cute.make_rmem_tensor_like(tY)

        rC = cute.make_rmem_tensor(tC.shape, Boolean)
        for i in range_constexpr(cute.size(rC)):
            rC[i] = cute.elem_less(tC[i], shape)

        is_within_boundary = cute.elem_less(tC[cute.size(tC) - 1], shape)

        if is_within_boundary:
            cute.copy(copy_atom, tG, rG)
            cute.copy(copy_atom, tU, rU)
        else:
            cute.copy(copy_atom, tG, rG, pred=rC)
            cute.copy(copy_atom, tU, rU, pred=rC)

        g = rG.load()
        u = rU.load()

        dtype = g.dtype
        g = g.to(Float32)
        y = u * g * sigmoid(g)
        y = y.to(dtype)

        rY.store(y)

        if is_within_boundary:
            cute.copy(copy_atom, rY, tY)
        else:
            cute.copy(copy_atom, rY, tY, pred=rC)

    @cute.jit
    def __call__(self, mG: cute.Tensor, mU: cute.Tensor, mY: cute.Tensor) -> None:
        vector_size = 128 // mG.element_type.width

        thr_layout = cute.make_ordered_layout((self.BLOCK_SIZE >> LOG_WARP_SIZE, WARP_SIZE), order=(1, 0))
        val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        gG = cute.zipped_divide(mG, tiler_mn)
        gU = cute.zipped_divide(mU, tiler_mn)
        gY = cute.zipped_divide(mY, tiler_mn)

        mC = cute.make_identity_tensor(mG.shape)
        gC = cute.zipped_divide(mC, tiler_mn)

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gG.element_type)
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        NUM_BLOCKS = cute.size(gG, mode=[1])

        self.kernel(gG=gG, gU=gU, gY=gY, gC=gC, copy_atom=copy_atom, tiled_copy=tiled_copy, shape=mG.shape).launch(
            grid=(NUM_BLOCKS, 1, 1), block=(self.BLOCK_SIZE, 1, 1)
        )


_CACHE = {}


@xma_op(mutates_args={"y"})
def swiglu_forward_cuda(g: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> None:
    key = g.dtype
    function = _CACHE.get(key, None)

    if function is None:
        N = g.size(1)
        divisibility = math.gcd(16 // key.itemsize, N)

        _g, _u, _y = [
            get_fake_cute_tensor(
                dtype=i.dtype,
                shape=(cute.sym_int(), cute.sym_int(divisibility=divisibility)),
                divisibility=divisibility,
            )
            for i in (g, u, y)
        ]

        function = SwiGLUForwardCUDAKernel()
        function = cute.compile(function, _g, _u, _y, options="--enable-tvm-ffi")
        _CACHE[key] = function

    function(g, u, y)
