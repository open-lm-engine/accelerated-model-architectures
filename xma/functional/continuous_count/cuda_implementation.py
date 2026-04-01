from __future__ import annotations

import math

import torch

import cutlass.cute as cute

from ...custom_op import xma_op
from ...cute_dsl_utils import get_fake_cute_tensor


class _ContinuousCountCUDAKernel:
    def __init__(self, BLOCK_SIZE: int = 128) -> _ContinuousCountCUDAKernel:
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
        block_coord = (None, BLOCK_ID)

        bX = gX[block_coord]

    @cute.jit
    def __call__(self, mX: cute.Tensor, mY: cute.Tensor) -> None:
        vector_size = 128 // mX.element_type.width

        Q = 4

        thr_layout = cute.make_ordered_layout((1, self.BLOCK_SIZE), order=(1, 0))
        val_layout = cute.make_ordered_layout((Q, vector_size), order=(1, 0))
        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

        mC = cute.make_identity_tensor(mX.shape)

        cute.recast_tensor()
        mX = cute.make_tensor(
            mX.iterator, layout=cute.make_layout(shape=(Q, cute.size(mX) // Q), stride=(cute.size(mX) // Q, 1))
        )
        mC = cute.make_tensor(
            mC.iterator, layout=cute.make_layout(shape=(Q, cute.size(mC) // Q), stride=(cute.size(mC) // Q, 1))
        )
        print(mX)
        print(mC)

        gX = cute.zipped_divide(mX, tiler_mn)
        gC = cute.zipped_divide(mC, tiler_mn)

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type)
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        NUM_BLOCKS = cute.Size(gX, mode=[1])
        kernel = self.kernel(gX=gX, gY=mY, gC=gC, copy_atom=copy_atom, tiled_copy=tiled_copy, shape=gX.shape)
        kernel.launch(grid=(NUM_BLOCKS, 1, 1), block=(self.BLOCK_SIZE, 1, 1))


_CACHE = {}


@xma_op(mutates_args={"y"})
def continuous_count_cuda(x: torch.Tensor, y: torch.Tensor) -> None:
    N = x.numel()
    C = y.numel()

    key = (x.dtype, C)
    function = _CACHE.get(key, None)

    if function is None:
        _x = get_fake_cute_tensor(
            dtype=x.dtype, shape=(cute.sym_int(),), divisibility=math.gcd(16 // x.dtype.itemsize, N)
        )
        _y = get_fake_cute_tensor(dtype=x.dtype, shape=(C,), divisibility=math.gcd(16 // x.dtype.itemsize, C))

        function = _ContinuousCountCUDAKernel()
        function = cute.compile(function, _x, _y, options="--enable-tvm-ffi")
        _CACHE[key] = function

    function(x, y)
