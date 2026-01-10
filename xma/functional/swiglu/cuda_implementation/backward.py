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


class SwiGLUBackwardCUDAKernel:
    def __init__(self, BLOCK_SIZE: int = 128) -> SwiGLUBackwardCUDAKernel:
        self.BLOCK_SIZE = BLOCK_SIZE

    @cute.kernel
    def kernel(
        self,
        gG: cute.Tensor,
        gU: cute.Tensor,
        gdY: cute.Tensor,
        gdG: cute.Tensor,
        gdU: cute.Tensor,
        gID: cute.Tensor,
        copy_atom: cute.CopyAtom,
        tiled_copy: cute.TiledCopy,
        shape: cute.Shape,
    ) -> None:
        BLOCK_ID, _, _ = cute.arch.block_idx()
        THREAD_ID, _, _ = cute.arch.thread_idx()

        block_coord = ((None, None), BLOCK_ID)

        bG = gG[block_coord]
        bU = gU[block_coord]
        bdY = gdY[block_coord]
        bdG = gdG[block_coord]
        bdU = gdU[block_coord]
        bID = gID[block_coord]

        thr_copy = tiled_copy.get_slice(THREAD_ID)

        tG = thr_copy.partition_S(bG)
        tU = thr_copy.partition_S(bU)
        tdY = thr_copy.partition_S(bdY)
        tdG = thr_copy.partition_D(bdG)
        tdU = thr_copy.partition_D(bdU)
        tID = thr_copy.partition_S(bID)

        fragG = cute.make_rmem_tensor_like(tG)
        fragU = cute.make_rmem_tensor_like(tU)
        fragdY = cute.make_rmem_tensor_like(tdY)
        fragdG = cute.make_rmem_tensor_like(tdG)
        fragdU = cute.make_rmem_tensor_like(tdU)

        fragID = cute.make_rmem_tensor(tID.shape, Boolean)
        for i in range_constexpr(cute.size(fragID)):
            fragID[i] = cute.elem_less(tID[i], shape)

        is_within_boundary = cute.elem_less(tID[cute.size(tID) - 1], shape)

        if is_within_boundary:
            cute.copy(copy_atom, tG, fragG)
            cute.copy(copy_atom, tU, fragU)
            cute.copy(copy_atom, tdY, fragdY)
        else:
            cute.copy(copy_atom, tG, fragG, pred=fragID)
            cute.copy(copy_atom, tU, fragU, pred=fragID)
            cute.copy(copy_atom, tdY, fragdY, pred=fragID)

        g = fragG.load()
        u = fragU.load()
        dy = fragdY.load()

        dtype = g.dtype
        g = g.to(Float32)

        g_sigmoid = sigmoid(g)
        g_silu = g * g_sigmoid

        dg = dy * u * (g_sigmoid + g_silu * (1 - g_sigmoid))
        du = dy * g_silu

        dg = dg.to(dtype)
        du = du.to(dtype)

        fragdG.store(dg)
        fragdU.store(du)

        if is_within_boundary:
            cute.copy(copy_atom, fragdG, tdG)
            cute.copy(copy_atom, fragdU, tdU)
        else:
            cute.copy(copy_atom, fragdG, tdG, pred=fragID)
            cute.copy(copy_atom, fragdU, tdU, pred=fragID)

    @cute.jit
    def __call__(self, mG: cute.Tensor, mU: cute.Tensor, mdY: cute.Tensor, mdG: cute.Tensor, mdU: cute.Tensor) -> None:
        vector_size = 128 // mG.element_type.width

        thr_layout = cute.make_ordered_layout((self.BLOCK_SIZE >> LOG_WARP_SIZE, WARP_SIZE), order=(1, 0))
        val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        gG = cute.zipped_divide(mG, tiler_mn)
        gU = cute.zipped_divide(mU, tiler_mn)
        gdY = cute.zipped_divide(mdY, tiler_mn)
        gdG = cute.zipped_divide(mdG, tiler_mn)
        gdU = cute.zipped_divide(mdU, tiler_mn)

        mID = cute.make_identity_tensor(mG.shape)
        gID = cute.zipped_divide(mID, tiler_mn)

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gG.element_type)
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        NUM_BLOCKS = cute.size(gG, mode=[1])

        self.kernel(
            gG=gG,
            gU=gU,
            gdY=gdY,
            gdG=gdG,
            gdU=gdU,
            gID=gID,
            copy_atom=copy_atom,
            tiled_copy=tiled_copy,
            shape=mG.shape,
        ).launch(grid=(NUM_BLOCKS, 1, 1), block=(self.BLOCK_SIZE, 1, 1))


_CACHE = {}


@xma_op(mutates_args={"dg", "du"})
def swiglu_backward_cuda(
    g: torch.Tensor, u: torch.Tensor, dy: torch.Tensor, dg: torch.Tensor, du: torch.Tensor
) -> None:
    key = g.element_type
    function = _CACHE.get(key, None)

    if function is None:
        N = g.size(1)
        divisibility = math.gcd(16 // key.itemsize, N)

        _g, _u, _dy, _dg, _du = [
            get_fake_cute_tensor(
                dtype=i.dtype,
                shape=(cute.sym_int(), cute.sym_int(divisibility=divisibility)),
                divisibility=divisibility,
            )
            for i in (g, u, dy, dg, du)
        ]

        function = SwiGLUBackwardCUDAKernel()
        function = cute.compile(function, _g, _u, _dy, _dg, _du)
        _CACHE[key] = function

    function(g, u, dy, dg, du)
