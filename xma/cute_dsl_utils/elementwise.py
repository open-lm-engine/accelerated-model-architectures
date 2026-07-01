# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Boolean, const_expr, range_constexpr

from ..constants import LOG_WARP_SIZE, WARP_SIZE
from .utils import get_fake_cute_tensor


def get_compiled_elementwise_cuda_fn(cache: dict, key, kernel_class: type, example_tensors: tuple, div: int):
    fn = cache.get(key)
    if fn is None:
        fake_tensors = [
            get_fake_cute_tensor(
                dtype=t.dtype, shape=(cute.sym_int(), cute.sym_int(divisibility=div)), divisibility=div
            )
            for t in example_tensors
        ]
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        fn = cute.compile(kernel_class(), *fake_tensors, stream, options="--enable-tvm-ffi")
        cache[key] = fn
    return fn


def _load_store(
    gX: cute.Tensor, rC: cute.Tensor, thr_copy, copy_atom: cute.CopyAtom, block_coord, is_within_boundary
) -> cute.TensorSSA:
    bX = gX[block_coord]
    tX = thr_copy.partition_S(bX)
    rX = cute.make_rmem_tensor_like(tX)

    if is_within_boundary:
        cute.copy(copy_atom, tX, rX)
    else:
        cute.copy(copy_atom, tX, rX, pred=rC)

    return rX.load()


class ElementwiseCUDAKernel:
    BLOCK_SIZE: int = 128

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
        is_x1_none = const_expr(gX1 is None)
        is_x2_none = const_expr(gX2 is None)
        is_y1_none = const_expr(gY1 is None)

        BLOCK_ID, _, _ = cute.arch.block_idx()
        THREAD_ID, _, _ = cute.arch.thread_idx()

        block_coord = ((None, None), BLOCK_ID)

        bY0 = gY0[block_coord]
        bC = gC[block_coord]

        thr_copy = tiled_copy.get_slice(THREAD_ID)

        tY0 = thr_copy.partition_D(bY0)
        tC = thr_copy.partition_S(bC)

        rY0 = cute.make_rmem_tensor_like(tY0)

        rC = cute.make_rmem_tensor(tC.shape, Boolean)
        for i in range_constexpr(cute.size(rC)):
            rC[i] = cute.elem_less(tC[i], shape)

        is_within_boundary = cute.elem_less(tC[cute.size(tC) - 1], shape)

        x0 = _load_store(
            gX=gX0,
            rC=rC,
            thr_copy=thr_copy,
            copy_atom=copy_atom,
            block_coord=block_coord,
            is_within_boundary=is_within_boundary,
        )

        if not is_x1_none:
            x1 = _load_store(
                gX=gX1,
                rC=rC,
                thr_copy=thr_copy,
                copy_atom=copy_atom,
                block_coord=block_coord,
                is_within_boundary=is_within_boundary,
            )

        if not is_x2_none:
            assert not is_x1_none

            x2 = _load_store(
                gX=gX2,
                rC=rC,
                thr_copy=thr_copy,
                copy_atom=copy_atom,
                block_coord=block_coord,
                is_within_boundary=is_within_boundary,
            )

        if is_x1_none:
            y = self.compute(x0)
        elif is_x2_none:
            y = self.compute(x0, x1)
        else:
            y = self.compute(x0, x1, x2)

        if is_y1_none:
            y0 = y
        else:
            y0, y1 = y

        rY0.store(y0)

        if is_within_boundary:
            cute.copy(copy_atom, rY0, tY0)
        else:
            cute.copy(copy_atom, rY0, tY0, pred=rC)

        if self.HAS_Y1:
            bY1 = gY1[block_coord]
            tY1 = thr_copy.partition_D(bY1)
            rY1 = cute.make_rmem_tensor_like(tY1)
            rY1.store(y1)
            if is_within_boundary:
                cute.copy(copy_atom, rY1, tY1)
            else:
                cute.copy(copy_atom, rY1, tY1, pred=rC)

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
        vector_size = 128 // mX0.element_type.width

        thr_layout = cute.make_ordered_layout((self.BLOCK_SIZE >> LOG_WARP_SIZE, WARP_SIZE), order=(1, 0))
        val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        mC = cute.make_identity_tensor(mX0.shape)

        is_x1_none = const_expr(mX1 is None)
        is_x2_none = const_expr(mX2 is None)
        is_y1_none = const_expr(mY1 is None)

        gC = cute.zipped_divide(mC, tiler_mn)
        gX0 = cute.zipped_divide(mX0, tiler_mn)

        if not is_x1_none:
            gX1 = cute.zipped_divide(mX1, tiler_mn)

        if not is_x2_none:
            assert not is_x2_none
            gX2 = cute.zipped_divide(mX2, tiler_mn)

        gY0 = cute.zipped_divide(mY0, tiler_mn)

        if not is_y1_none:
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
