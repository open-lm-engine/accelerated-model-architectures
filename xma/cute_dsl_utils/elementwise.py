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


@cute.jit
def _load(
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


@cute.jit
def _store(
    gY: cute.Tensor,
    y: cute.TensorSSA,
    rC: cute.Tensor,
    thr_copy,
    copy_atom: cute.CopyAtom,
    block_coord,
    is_within_boundary,
) -> None:
    bY = gY[block_coord]
    tY = thr_copy.partition_D(bY)
    rY = cute.make_rmem_tensor_like(tY)

    rY.store(y)

    if is_within_boundary:
        cute.copy(copy_atom, rY, tY)
    else:
        cute.copy(copy_atom, rY, tY, pred=rC)


class ElementwiseCUDAKernel:
    def __init__(self, BLOCK_SIZE: int) -> ElementwiseCUDAKernel:
        self.BLOCK_SIZE = BLOCK_SIZE

    def compute(self, xs: list[cute.TensorSSA]) -> list[cute.TensorSSA]:
        raise NotImplementedError

    @cute.kernel
    def kernel(
        self,
        gXs: list[cute.Tensor],
        gYs: list[cute.Tensor],
        gC: cute.Tensor,
        copy_atom_Xs: list[cute.CopyAtom],
        copy_atom_Ys: list[cute.CopyAtom],
        tiled_copy_Xs: list[cute.TiledCopy],
        tiled_copy_Ys: list[cute.TiledCopy],
        shape: cute.Shape,
    ) -> None:
        BLOCK_ID, _, _ = cute.arch.block_idx()
        THREAD_ID, _, _ = cute.arch.thread_idx()

        block_coord = ((None, None), BLOCK_ID)
        bC = gC[block_coord]

        thr_copy = tiled_copy_Xs[0].get_slice(THREAD_ID)
        tC = thr_copy.partition_S(bC)

        rC = cute.make_rmem_tensor(tC.shape, Boolean)
        for i in range_constexpr(cute.size(rC)):
            rC[i] = cute.elem_less(tC[i], shape)

        is_within_boundary = cute.elem_less(tC[cute.size(tC) - 1], shape)

        xs = [
            _load(
                gX=gX,
                rC=rC,
                thr_copy=thr_copy,
                copy_atom=copy_atom,
                block_coord=block_coord,
                is_within_boundary=is_within_boundary,
            )
            for gX, copy_atom in zip(gXs, copy_atom_Xs)
        ]

        ys = self.compute(xs)

        for y, gY, copy_atom in zip(ys, gYs, copy_atom_Ys):
            _store(
                gY=gY,
                y=y,
                rC=rC,
                thr_copy=thr_copy,
                copy_atom=copy_atom,
                block_coord=block_coord,
                is_within_boundary=is_within_boundary,
            )

    @cute.jit
    def __call__(self, mXs: list[cute.Tensor], mYs: list[cute.Tensor], stream: cuda.CUstream) -> None:
        vector_size = min([128 // i.element_type.width for i in mXs])

        thr_layout = cute.make_ordered_layout((self.BLOCK_SIZE >> LOG_WARP_SIZE, WARP_SIZE), order=(1, 0))
        val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
        tiler_mn, _ = cute.make_layout_tv(thr_layout, val_layout)

        mC = cute.make_identity_tensor(mXs[0].shape)
        gC = cute.zipped_divide(mC, tiler_mn)

        gXs = [cute.zipped_divide(i, tiler_mn) for i in mXs]
        gYs = [cute.zipped_divide(i, tiler_mn) for i in mYs]

        copy_atom_Xs = [cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), i.element_type) for i in gXs]
        copy_atom_Ys = [cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), i.element_type) for i in gYs]

        tiled_copy_Xs = [cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout) for copy_atom in copy_atom_Xs]
        tiled_copy_Ys = [cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout) for copy_atom in copy_atom_Ys]

        NUM_BLOCKS = cute.size(gXs[0], mode=[1])

        self.kernel(
            gXs=gXs,
            gYs=gYs,
            gC=gC,
            copy_atom_Xs=copy_atom_Xs,
            copy_atom_Ys=copy_atom_Ys,
            tiled_copy_Xs=tiled_copy_Xs,
            tiled_copy_Ys=tiled_copy_Ys,
            shape=mXs[0].shape,
        ).launch(grid=(NUM_BLOCKS, 1, 1), block=(self.BLOCK_SIZE, 1, 1), stream=stream)


def get_compiled_elementwise_cuda_kernel(
    cache: dict,
    key,
    kernel_class: type,
    example_tensors_list: tuple[tuple[torch.Tensor]],
    div: int,
    stream: cuda.CUstream,
) -> ElementwiseCUDAKernel:
    kernel = cache.get(key)

    if kernel is None:
        fake_tensors = [
            [
                (
                    None
                    if t is None
                    else get_fake_cute_tensor(
                        dtype=t.dtype, shape=(cute.sym_int(), cute.sym_int(divisibility=div)), divisibility=div
                    )
                )
                for t in example_tensors
            ]
            for example_tensors in example_tensors_list
        ]

        kernel = cute.compile(kernel_class(), *fake_tensors, stream, options="--enable-tvm-ffi")
        cache[key] = kernel

    return kernel
