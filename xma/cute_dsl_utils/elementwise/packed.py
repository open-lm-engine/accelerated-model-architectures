# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Boolean, const_expr, range_constexpr

from ..constants import LOG_WARP_SIZE, WARP_SIZE
from .base import _load, _store


@cute.jit
def _lane_boundary(gC: cute.Tensor, tiled_copy: cute.TiledCopy, block_coord, THREAD_ID: int, shape: cute.Shape):
    thr_copy = tiled_copy.get_slice(THREAD_ID)

    bC = gC[block_coord]
    tC = thr_copy.partition_S(bC)

    rC = cute.make_rmem_tensor(tC.shape, Boolean)
    for i in range_constexpr(cute.size(rC)):
        rC[i] = cute.elem_less(tC[i], shape)

    is_within_boundary = cute.elem_less(tC[cute.size(tC) - 1], shape)

    return thr_copy, rC, is_within_boundary


class ElementwisePackedCUDAKernel:
    BLOCK_SIZE: int = 128
    X0_PACKED: bool = True
    X1_PACKED: bool = True
    Y0_PACKED: bool = True

    def compute(self, *inputs):
        raise NotImplementedError

    @cute.kernel
    def kernel(
        self,
        gX0: cute.Tensor,
        gX1: cute.Tensor | None,
        gY0: cute.Tensor,
        gC_1: cute.Tensor,
        gC_2: cute.Tensor,
        copy_atom: cute.CopyAtom,
        tiled_copy_1: cute.TiledCopy,
        tiled_copy_2: cute.TiledCopy,
        shape: cute.Shape,
    ) -> None:
        BLOCK_ID, _, _ = cute.arch.block_idx()
        THREAD_ID, _, _ = cute.arch.thread_idx()

        block_coord = ((None, None), BLOCK_ID)

        thr_copy_1, rC_1, is_within_boundary_1 = _lane_boundary(gC_1, tiled_copy_1, block_coord, THREAD_ID, shape)
        thr_copy_2, rC_2, is_within_boundary_2 = _lane_boundary(gC_2, tiled_copy_2, block_coord, THREAD_ID, shape)

        x0 = _load(
            gX=gX0,
            rC=rC_1 if const_expr(self.X0_PACKED) else rC_2,
            thr_copy=thr_copy_1 if const_expr(self.X0_PACKED) else thr_copy_2,
            copy_atom=copy_atom,
            block_coord=block_coord,
            is_within_boundary=is_within_boundary_1 if const_expr(self.X0_PACKED) else is_within_boundary_2,
        )

        if const_expr(gX1 is not None):
            x1 = _load(
                gX=gX1,
                rC=rC_1 if const_expr(self.X1_PACKED) else rC_2,
                thr_copy=thr_copy_1 if const_expr(self.X1_PACKED) else thr_copy_2,
                copy_atom=copy_atom,
                block_coord=block_coord,
                is_within_boundary=is_within_boundary_1 if const_expr(self.X1_PACKED) else is_within_boundary_2,
            )

        if const_expr(gX1 is None):
            if self.X0_PACKED:
                # x0 holds interleaved pairs; each pair collapses to one output element
                vector_size = cute.size(x0, mode=[1])
                y0_vals = cute.make_rmem_tensor((vector_size >> 1,), gY0.element_type)
                for i in range_constexpr(0, vector_size, 2):
                    y0_vals[i // 2] = self.compute(x0[i], x0[i + 1])
                y = y0_vals.load()
            else:
                y = self.compute(x0)
        else:
            y = self.compute(x0, x1)

        _store(
            gY=gY0,
            y=y,
            rC=rC_1 if const_expr(self.Y0_PACKED) else rC_2,
            thr_copy=thr_copy_1 if const_expr(self.Y0_PACKED) else thr_copy_2,
            copy_atom=copy_atom,
            block_coord=block_coord,
            is_within_boundary=is_within_boundary_1 if const_expr(self.Y0_PACKED) else is_within_boundary_2,
        )

    @cute.jit
    def __call__(self, mX0: cute.Tensor, mX1: cute.Tensor | None, mY0: cute.Tensor, stream: cuda.CUstream) -> None:
        dtype = mX0.element_type

        assert mY0.element_type == dtype

        if const_expr(mX1 is not None):
            assert mX1.element_type == dtype

        if const_expr(mY0 is not None):
            assert mY0.element_type == dtype

        vector_size = 128 // dtype.width

        thr_layout = cute.make_ordered_layout((self.BLOCK_SIZE >> LOG_WARP_SIZE, WARP_SIZE), order=(1, 0))

        val_layout_1 = cute.make_ordered_layout((4, vector_size >> 1), order=(1, 0))
        tiler_mn_1, tv_layout_1 = cute.make_layout_tv(thr_layout, val_layout_1)

        val_layout_2 = cute.make_ordered_layout((4, vector_size), order=(1, 0))
        tiler_mn_2, tv_layout_2 = cute.make_layout_tv(thr_layout, val_layout_2)

        mC = cute.make_identity_tensor(mX0.shape)
        gC_1 = cute.zipped_divide(mC, tiler_mn_1)
        gC_2 = cute.zipped_divide(mC, tiler_mn_2)

        tiler_mn_X0 = tiler_mn_1 if self.X0_PACKED else tiler_mn_2
        gX0 = cute.zipped_divide(mX0, tiler_mn_X0)

        if const_expr(mX1 is None):
            gX1 = None
        else:
            tiler_mn_X1 = tiler_mn_1 if self.X1_PACKED else tiler_mn_2
            gX1 = cute.zipped_divide(mX1, tiler_mn_X1)

        tiler_mn_Y0 = tiler_mn_1 if self.Y0_PACKED else tiler_mn_2
        gY0 = cute.zipped_divide(mY0, tiler_mn_Y0)

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX0.element_type)
        tiled_copy_1 = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout_1)
        tiled_copy_2 = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout_2)

        NUM_BLOCKS = cute.size(gX0, mode=[1])

        self.kernel(
            gX0=gX0,
            gX1=gX1,
            gY0=gY0,
            gC_1=gC_1,
            gC_2=gC_2,
            copy_atom=copy_atom,
            tiled_copy_1=tiled_copy_1,
            tiled_copy_2=tiled_copy_2,
            shape=mX0.shape,
        ).launch(grid=(NUM_BLOCKS, 1, 1), block=(self.BLOCK_SIZE, 1, 1), stream=stream)
