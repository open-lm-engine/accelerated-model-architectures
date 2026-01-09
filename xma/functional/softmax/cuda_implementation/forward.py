# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

import cutlass.cute as cute
from cutlass import Boolean, Float32, range_constexpr

from ....constants import LOG_WARP_SIZE, WARP_SIZE
from ....custom_op import xma_op
from ....cute_dsl_utils import torch_tensor_to_cute_tensor


@cute.kernel
def softmax_forward_cuda_kernel(
    gX: cute.Tensor,
    gY: cute.Tensor,
    gID: cute.Tensor,
    copy_atom: cute.CopyAtom,
    tiled_copy: cute.TiledCopy,
    shape: cute.Shape,
) -> None:
    BLOCK_ID, _, _ = cute.arch.block_idx()
    THREAD_ID, _, _ = cute.arch.thread_idx()

    block_coord = ((None, None), BLOCK_ID)


@cute.jit
def softmax_forward_cuda_jit(mX: cute.Tensor, mY: cute.Tensor) -> None:
    BLOCK_SIZE = 128
    vector_size = 128 // mX.element_type.width

    thr_layout = cute.make_ordered_layout((BLOCK_SIZE >> LOG_WARP_SIZE, WARP_SIZE), order=(1, 0))
    val_layout = cute.make_ordered_layout((1, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    mID = cute.make_identity_tensor(gX.shape)

    gX = cute.zipped_divide(mX, tiler_mn)
    gY = cute.zipped_divide(mY, tiler_mn)
    gID = cute.zipped_divide(mID, tiler_mn)

    softmax_forward_cuda_kernel(gX=gX, gY=gY, gID=gID, copy_atom=copy_atom, tiled_copy=tiled_copy, shape=mX.shape)


@xma_op(mutates_args={"y"})
def softmax_forward_cuda(x: torch.Tensor, y: torch.Tensor, logits_multiplier: float | None) -> None:
    x = torch_tensor_to_cute_tensor(x, leading_dim=-1)
    y = torch_tensor_to_cute_tensor(y, leading_dim=-1)

    key = x.element_type
    function = softmax_forward_cuda.cache.get(key, None)

    if function is None:
        function = cute.compile(softmax_forward_cuda_jit, x, y)
        softmax_forward_cuda.cache[key] = function

    function(x, y)


softmax_forward_cuda.cache = {}
