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
        N: int,
        C: int,
        copy_atom: cute.CopyAtom,
        tiled_copy: cute.TiledCopy,
        shape: cute.Shape,
    ) -> None:
        BLOCK_ID, _, _ = cute.arch.block_idx()
        THREAD_ID, _, _ = cute.arch.thread_idx()
        block_coord = (None, BLOCK_ID)

        bX = gX[block_coord]


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
