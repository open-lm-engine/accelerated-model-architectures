# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import const_expr

from ..constants import LOG_WARP_SIZE, WARP_SIZE
from .boundary import lane_boundary
from .elementwise import _load, _store


class ElementwisePackedCUDAKernel:
    def __init__(self, BLOCK_SIZE: int) -> ElementwisePackedCUDAKernel:
        self.BLOCK_SIZE = BLOCK_SIZE

    def compute(self, *inputs):
        raise NotImplementedError

    @cute.kernel
    def kernel(
        self,
        gXs_1: cute.Tensor,
        gXs_2: cute.Tensor,
        gYs_1: cute.Tensor,
        gYs_2: cute.Tensor,
        gC: cute.Tensor,
        copy_atom_Xs_1: cute.CopyAtom,
        copy_atom_Ys_1: cute.CopyAtom,
        tiled_copy_Xs_1: cute.TiledCopy,
        shape: cute.Shape,
    ) -> None:
        BLOCK_ID, _, _ = cute.arch.block_idx()
        THREAD_ID, _, _ = cute.arch.thread_idx()

        block_coord = ((None, None), BLOCK_ID)

        thr_copy, rC, is_within_boundary = lane_boundary(
            gC=gC, tiled_copy=tiled_copy_Xs_1[0], block_coord=block_coord, THREAD_ID=THREAD_ID, shape=shape
        )

        xs_1 = [
            _load(
                gX=gX,
                rC=rC,
                thr_copy=thr_copy,
                copy_atom=copy_atom,
                block_coord=block_coord,
                is_within_boundary=is_within_boundary,
            )
            for gX, copy_atom in zip(gXs_1, copy_atom_Xs_1)
        ]

        xs_2 = [
            _load(
                gX=gX,
                rC=rC,
                thr_copy=thr_copy,
                copy_atom=copy_atom,
                block_coord=block_coord,
                is_within_boundary=is_within_boundary,
            )
            for gX, copy_atom in zip(gXs_2, copy_atom_Xs_1)
        ]

        ys_1, ys_2 = self.compute(xs_1, xs_2)

        for y, gY in [(ys_1, gYs_1), (ys_2, gYs_2)]:
            _store(
                gY=gY,
                y=y,
                rC=rC,
                thr_copy=thr_copy,
                copy_atom=copy_atom_Ys_1[0],
                block_coord=block_coord,
                is_within_boundary=is_within_boundary,
            )

    @cute.jit
    def __call__(
        self,
        mXs_1: list[cute.Tensor],
        mXs_2: list[cute.Tensor],
        mYs_1: list[cute.Tensor],
        mYs_2: list[cute.Tensor],
        stream: cuda.CUstream,
    ) -> None:
        vector_size = min([128 // i.element_type.width for i in mXs_1 + mXs_2 + mYs_1 + mYs_2])

        thr_layout = cute.make_ordered_layout((self.BLOCK_SIZE >> LOG_WARP_SIZE, WARP_SIZE), order=(1, 0))

        val_layout_1 = cute.make_ordered_layout((4, vector_size >> 1), order=(1, 0))
        val_layout_2 = cute.make_ordered_layout((4, vector_size), order=(1, 0))

        tiler_mn_1, _ = cute.make_layout_tv(thr_layout, val_layout_1)
        tiler_mn_2, _ = cute.make_layout_tv(thr_layout, val_layout_2)

        mC = cute.make_identity_tensor((mYs_1 if const_expr(len(mXs_1) == 0) else mXs_1)[0].shape)
        gC = cute.zipped_divide(mC, tiler_mn_1)

        gXs_1 = [cute.zipped_divide(i, tiler_mn_1) for i in mXs_1]
        gXs_2 = [cute.zipped_divide(i, tiler_mn_2) for i in mXs_2]
        gYs_1 = [cute.zipped_divide(i, tiler_mn_1) for i in mYs_1]
        gYs_2 = [cute.zipped_divide(i, tiler_mn_2) for i in mYs_2]

        copy_atom_Xs_1 = [cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), i.element_type) for i in gXs_1]
        copy_atom_Ys_1 = [cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), i.element_type) for i in gYs_1]

        tiled_copy_Xs_1 = [
            cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout_1) for copy_atom in copy_atom_Xs_1
        ]

        NUM_BLOCKS = cute.size(gXs_1[0], mode=[1])

        self.kernel(
            gXs_1=gXs_1,
            gXs_2=gXs_2,
            gYs_1=gYs_1,
            gYs_2=gYs_2,
            gC=gC,
            copy_atom_Xs_1=copy_atom_Xs_1,
            copy_atom_Ys_1=copy_atom_Ys_1,
            tiled_copy_Xs_1=tiled_copy_Xs_1,
            shape=mXs_1[0].shape,
        ).launch(grid=(NUM_BLOCKS, 1, 1), block=(self.BLOCK_SIZE, 1, 1), stream=stream)
