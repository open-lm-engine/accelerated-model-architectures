# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import torch

import cutlass.cute as cute
from cutlass import const_expr

from ...custom_op import xma_op
from ...cute_dsl_utils import get_fake_cute_tensor


class _PackUnpackSequenceCUDAKernel:
    def __init__(self, N: int, padding_side: str, pack: bool, BLOCK_SIZE: int) -> _PackUnpackSequenceCUDAKernel:
        self.N = N
        self.padding_side = padding_side
        self.pack = pack
        self.BLOCK_SIZE = BLOCK_SIZE

    def _copy_array(self, copy_atom: cute.CopyAtom, tiled_copy: cute.TiledCopy) -> None: ...

    @cute.kernel
    def kernel(
        self,
        gX: cute.Tensor,
        gY: cute.Tensor,
        gC: cute.Tensor,
        gCu_seqlens: cute.Tensor,
        copy_atom: cute.CopyAtom,
        tiled_copy: cute.TiledCopy,
        shape: cute.Shape,
    ) -> None:
        BLOCK_ID_S, BLOCK_ID_B, _ = cute.arch.block_idx()
        THREAD_ID, _, _ = cute.arch.thread_idx()

        start = gCu_seqlens[BLOCK_ID_B]
        end = gCu_seqlens[BLOCK_ID_B + 1]
        seqlens = end - start

        S = cute.size(gX if const_expr(self.pack) else gY, mode=[1])

        if const_expr(self.padding_side == "left"):
            pad_tokens = S - seqlens

            if BLOCK_ID_S >= pad_tokens:
                ...
        elif BLOCK_ID_S < seqlens:
            ...

    @cute.jit
    def __call__(self, mX: cute.Tensor, mY: cute.Tensor, mCu_seqlens: cute.Tensor, stream: cuda.CUstream) -> None:
        vector_size = 128 // mX.element_type.width

        thr_layout = cute.make_ordered_layout((1, self.BLOCK_SIZE), order=(1, 0))
        val_layout = cute.make_ordered_layout((1, vector_size), order=(1, 0))

        mC = cute.make_identity_tensor(mX.shape)

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mX.element_type)
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        B = cute.size(mX, mode=[0])
        S = cute.size(mX, mode=[1])

        kernel = self.kernel(
            gX=mX, gY=mY, gC=mC, gCu_seqlens=mCu_seqlens, copy_atom=copy_atom, tiled_copy=tiled_copy, shape=mX.shape
        )

        kernel.launch(grid=(S, B, 1), block=(self.BLOCK_SIZE, 1, 1), stream=stream)


_CACHE = {}


@xma_op(mutates_args={"y"})
def _pack_unpack_sequence_cuda(
    x: torch.Tensor, y: torch.Tensor, cu_seqlens: torch.Tensor, padding_side: str, pack: bool, BLOCK_SIZE: int
) -> None:
    if pack:
        x = x.flatten(2, -1)
        y = y.flatten(1, -1)
        B, _, N = x.size()
    else:
        x = x.flatten(1, -1)
        y = y.flatten(2, -1)
        B, _, N = y.size()

    key = (x.dtype, N, pack, padding_side, BLOCK_SIZE)
    function = _CACHE.get(key, None)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if function is None:
        x_div = math.gcd(16 // x.dtype.itemsize, N)
        cu_seqlens_div = math.gcd(16 // cu_seqlens.dtype.itemsize, B + 1)

        x_shape = [cute.sym_int(), cute.sym_int(), N]
        y_shape = [cute.sym_int(), N]

        if not pack:
            x_shape, y_shape = y_shape, x_shape

        _x = get_fake_cute_tensor(dtype=x.dtype, shape=x_shape, divisibility=x_div)
        _y = get_fake_cute_tensor(dtype=x.dtype, shape=y_shape, divisibility=x_div)
        _cu_seqlens = get_fake_cute_tensor(
            dtype=cu_seqlens.dtype, shape=(cute.sym_int(),), divisibility=cu_seqlens_div
        )

        if not pack:
            _x, _y = _y, _x

        function = _PackUnpackSequenceCUDAKernel(N=N, padding_side=padding_side, pack=pack, BLOCK_SIZE=BLOCK_SIZE)

        function = cute.compile(function, _x, _y, _cu_seqlens, stream, options="--enable-tvm-ffi")
        _CACHE[key] = function

    function(x, y, cu_seqlens, stream)
