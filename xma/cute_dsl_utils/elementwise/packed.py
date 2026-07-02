# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Boolean, const_expr, range_constexpr

from ..constants import LOG_WARP_SIZE, WARP_SIZE
from .unpacked import _load, _store


class ElementwisePackedCUDAKernel:
    BLOCK_SIZE: int = 128
    INPUT_PACKED: bool = True

    def compute(self, *inputs):
        raise NotImplementedError

    @cute.kernel
    def kernel(
        self,
        gX0: cute.Tensor,
        gX1: cute.Tensor | None,
        gX2: cute.Tensor | None,
        gY0: cute.Tensor,
        gY1: cute.Tensor | None,
        gC: cute.Tensor,
        copy_atom: cute.CopyAtom,
        tiled_copy: cute.TiledCopy,
        shape: cute.Shape,
    ) -> None:
        BLOCK_ID, _, _ = cute.arch.block_idx()
        THREAD_ID, _, _ = cute.arch.thread_idx()

        block_coord = ((None, None), BLOCK_ID)
        bC = gC[block_coord]

        thr_copy = tiled_copy.get_slice(THREAD_ID)
        tC = thr_copy.partition_S(bC)

        rC = cute.make_rmem_tensor(tC.shape, Boolean)
        for i in range_constexpr(cute.size(rC)):
            rC[i] = cute.elem_less(tC[i], shape)

        is_within_boundary = cute.elem_less(tC[cute.size(tC) - 1], shape)

        x0 = _load(
            gX=gX0,
            rC=rC,
            thr_copy=thr_copy,
            copy_atom=copy_atom,
            block_coord=block_coord,
            is_within_boundary=is_within_boundary,
        )

        if const_expr(not is_x1_none):
            x1 = _load(
                gX=gX1,
                rC=rC,
                thr_copy=thr_copy,
                copy_atom=copy_atom,
                block_coord=block_coord,
                is_within_boundary=is_within_boundary,
            )

        if const_expr(not is_x2_none):
            assert not is_x1_none

            x2 = _load(
                gX=gX2,
                rC=rC,
                thr_copy=thr_copy,
                copy_atom=copy_atom,
                block_coord=block_coord,
                is_within_boundary=is_within_boundary,
            )

        if const_expr(is_x1_none):
            y = self.compute(x0)
        elif const_expr(is_x2_none):
            y = self.compute(x0, x1)
        else:
            y = self.compute(x0, x1, x2)

        if const_expr(is_y1_none):
            y0 = y
        else:
            y0, y1 = y

        _store(
            gY=gY0,
            y=y0,
            rC=rC,
            thr_copy=thr_copy,
            copy_atom=copy_atom,
            block_coord=block_coord,
            is_within_boundary=is_within_boundary,
        )

        if const_expr(not is_y1_none):
            _store(
                gY=gY1,
                y=y1,
                rC=rC,
                thr_copy=thr_copy,
                copy_atom=copy_atom,
                block_coord=block_coord,
                is_within_boundary=is_within_boundary,
            )

    @cute.jit
    def __call__(self, mX: cute.Tensor, mY: cute.Tensor, stream: cuda.CUstream) -> None:
        vector_size = 128 // mX.element_type.width

        thr_layout = cute.make_ordered_layout((self.BLOCK_SIZE >> LOG_WARP_SIZE, WARP_SIZE), order=(1, 0))
        val_layout_X = cute.make_ordered_layout((4, vector_size >> (1 - self.INPUT_PACKED)), order=(1, 0))
        tiler_mn_X, tv_layout_X = cute.make_layout_tv(thr_layout, val_layout_X)

        val_layout_Y = cute.make_ordered_layout((4, vector_size >> self.INPUT_PACKED), order=(1, 0))
        tiler_mn_Y, tv_layout_Y = cute.make_layout_tv(thr_layout, val_layout_Y)

        mC = cute.make_identity_tensor(mY.shape)

        gX = cute.zipped_divide(mX, tiler_mn_X)
        gC = cute.zipped_divide(mC, tiler_mn_Y if self.INPUT_PACKED else tiler_mn_X)
        gY = cute.zipped_divide(mY, tiler_mn_Y)

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type)

        tiled_copy_X = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout_X)
        tiled_copy_Y = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout_Y)

        NUM_BLOCKS = cute.size(gX, mode=[1])

        self.kernel(
            gX=gX,
            gY=gY,
            gC=gC,
            copy_atom=copy_atom,
            tiled_copy_X=tiled_copy_X,
            tiled_copy=tiled_copy_Y,
            shape=mY.shape,
        ).launch(grid=(NUM_BLOCKS, 1, 1), block=(self.BLOCK_SIZE, 1, 1), stream=stream)
