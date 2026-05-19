# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import cuda.bindings.driver as cuda
import torch

import cutlass.cute as cute

from ...custom_op import xma_op


class _PackUnpackSequenceCUDAKernel:
    def __init__(self, BLOCK_SIZE: int) -> _PackUnpackSequenceCUDAKernel:
        self.BLOCK_SIZE = BLOCK_SIZE

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mC: cute.Tensor,
        copy_atom: cute.CopyAtom,
        tiled_copy: cute.TiledCopy,
        shape: cute.Shape,
    ) -> None:
        BLOCK_ID_S, BLOCK_ID_B, _ = cute.arch.block_idx()
        THREAD_ID, _, _ = cute.arch.thread_idx()

    @cute.jit
    def __call__(self, mX: cute.Tensor, mY: cute.Tensor, stream: cuda.CUstream) -> None:
        vector_size = 128 // mX.element_type.width

        thr_layout = cute.make_ordered_layout((1, self.BLOCK_SIZE), order=(1, 0))
        val_layout = cute.make_ordered_layout((self.B, vector_size), order=(1, 0))
        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        mC = cute.make_identity_tensor(mX.shape)

        gX = cute.zipped_divide(mX, tiler_mn)
        gY = cute.zipped_divide(mY, tiler_mn)
        gC = cute.zipped_divide(mC, tiler_mn)

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type)
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        B = cute.size(gX, mode=[0])
        S = cute.size(gX, mode=[1])

        kernel = self.kernel(gX=gX, gY=gY, gC=gC, copy_atom=copy_atom, tiled_copy=tiled_copy, shape=mX.shape)
        kernel.launch(grid=(S, B, 1), block=(self.BLOCK_SIZE, 1, 1), stream=stream)


@xma_op(mutates_args={"y"})
def _pack_unpack_sequence_cuda(
    x: torch.Tensor, y: torch.Tensor, cu_seqlens: torch.Tensor, padding_side: str, pack: bool
) -> None: ...
