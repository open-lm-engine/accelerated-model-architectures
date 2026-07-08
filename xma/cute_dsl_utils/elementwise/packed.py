# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Boolean, const_expr, range_constexpr

from ..constants import LOG_WARP_SIZE, WARP_SIZE
from .base import _load, _store


class ElementwisePackedCUDAKernel:
    BLOCK_SIZE: int = 128
    X0_PACKED: bool = True
    X1_PACKED: bool = True
    X2_PACKED: bool = True
    Y0_PACKED: bool = True
    Y1_PACKED: bool = True

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

        if const_expr(gX1 is not None):
            x1 = _load(
                gX=gX1,
                rC=rC,
                thr_copy=thr_copy,
                copy_atom=copy_atom,
                block_coord=block_coord,
                is_within_boundary=is_within_boundary,
            )

        if const_expr(gX2 is not None):
            assert const_expr(gX1 is not None)

            x2 = _load(
                gX=gX2,
                rC=rC,
                thr_copy=thr_copy,
                copy_atom=copy_atom,
                block_coord=block_coord,
                is_within_boundary=is_within_boundary,
            )

        if const_expr(gX1 is None):
            if self.INPUT_PACKED:
                vector_size = cute.size(x0, mode=[1])
                for i in range_constexpr(0, vector_size, 2):
                    y = self.compute(x0[i], x0[i + 1])
        elif const_expr(gX2 is None):
            y = self.compute(x0, x1)
        else:
            y = self.compute(x0, x1, x2)

        if const_expr(gY1 is None):
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

        if const_expr(gY1 is not None):
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
    def __call__(
        self,
        mX0: cute.Tensor,
        mX1: cute.Tensor | None,
        mX2: cute.Tensor | None,
        mY0: cute.Tensor,
        mY1: cute.Tensor | None,
        stream: cuda.CUstream,
    ) -> None:
        dtype = mX0.element_type

        assert mY0.element_type == dtype

        if const_expr(mX1 is not None):
            assert mX1.element_type == dtype

        if const_expr(mX2 is not None):
            assert mX2.element_type == dtype

        if const_expr(mY1 is not None):
            assert mY1.element_type == dtype

        vector_size = 128 // dtype.width

        thr_layout = cute.make_ordered_layout((self.BLOCK_SIZE >> LOG_WARP_SIZE, WARP_SIZE), order=(1, 0))
        val_layout_2 = cute.make_ordered_layout((4, vector_size), order=(1, 0))
        tiler_mn_2, tv_layout_2 = cute.make_layout_tv(thr_layout, val_layout_2)

        val_layout_1 = cute.make_ordered_layout((4, vector_size >> 1), order=(1, 0))
        tiler_mn_1, tv_layout_1 = cute.make_layout_tv(thr_layout, val_layout_1)

        mC = cute.make_identity_tensor(mX0.shape)

        gC = cute.zipped_divide(mC, tiler_mn)
        gX0 = cute.zipped_divide(mX0, tiler_mn)

        if const_expr(mX1 is None):
            gX1 = None
        else:
            gX1 = cute.zipped_divide(mX1, tiler_mn)

        if const_expr(mX2 is None):
            gX2 = None
        else:
            assert const_expr(mX1 is not None)
            gX2 = cute.zipped_divide(mX2, tiler_mn)

        gY0 = cute.zipped_divide(mY0, tiler_mn)

        if const_expr(mY1 is None):
            gY1 = None
        else:
            gY1 = cute.zipped_divide(mY1, tiler_mn)

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX0.element_type)
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        NUM_BLOCKS = cute.size(gX0, mode=[1])

        self.kernel(
            gX0=gX0,
            gX1=gX1,
            gX2=gX2,
            gY0=gY0,
            gY1=gY1,
            gC=gC,
            copy_atom=copy_atom,
            tiled_copy=tiled_copy,
            shape=mX0.shape,
        ).launch(grid=(NUM_BLOCKS, 1, 1), block=(self.BLOCK_SIZE, 1, 1), stream=stream)
