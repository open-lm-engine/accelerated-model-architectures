# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

import cutlass.cute as cute
from cutlass import Boolean, Float32, range_constexpr

from ....constants import LOG_WARP_SIZE, WARP_SIZE
from ....custom_op import xma_op
from ....cute_dsl_utils import sigmoid, torch_tensor_to_cute_tensor


@cute.kernel
def swiglu_forward_cuda_kernel(
    gG: cute.Tensor,
    gU: cute.Tensor,
    gY: cute.Tensor,
    gID: cute.Tensor,
    copy_atom: cute.CopyAtom,
    tiled_copy: cute.TiledCopy,
    shape: cute.Shape,
) -> None:
    BLOCK_ID, _, _ = cute.arch.block_idx()
    THREAD_ID, _, _ = cute.arch.thread_idx()

    block_coord = ((None, None), BLOCK_ID)

    bG = gG[block_coord]
    bU = gU[block_coord]
    bY = gY[block_coord]
    bID = gID[block_coord]

    thr_copy = tiled_copy.get_slice(THREAD_ID)

    tG = thr_copy.partition_S(bG)
    tU = thr_copy.partition_S(bU)
    tY = thr_copy.partition_D(bY)
    tID = thr_copy.partition_S(bID)

    fragG = cute.make_rmem_tensor_like(tG)
    fragU = cute.make_rmem_tensor_like(tU)
    fragY = cute.make_rmem_tensor_like(tY)

    fragID = cute.make_rmem_tensor(tID.shape, Boolean)
    for i in range_constexpr(cute.size(fragID)):
        fragID[i] = cute.elem_less(tID[i], shape)

    is_within_boundary = cute.elem_less(tID[cute.size(tID) - 1], shape)

    if is_within_boundary:
        cute.copy(copy_atom, tG, fragG)
        cute.copy(copy_atom, tU, fragU)
    else:
        cute.copy(copy_atom, tG, fragG, pred=fragID)
        cute.copy(copy_atom, tU, fragU, pred=fragID)

    g = fragG.load()
    u = fragU.load()

    dtype = g.dtype
    g = g.to(Float32)
    y = u * g * sigmoid(g)
    y = y.to(dtype)

    fragY.store(y)

    if is_within_boundary:
        cute.copy(copy_atom, fragY, tY)
    else:
        cute.copy(copy_atom, fragY, tY, pred=fragID)


@cute.jit
def swiglu_forward_cuda_jit(mG: cute.Tensor, mU: cute.Tensor, mY: cute.Tensor) -> None:
    BLOCK_SIZE = 128
    vector_size = 128 // mG.element_type.width

    thr_layout = cute.make_ordered_layout((BLOCK_SIZE >> LOG_WARP_SIZE, WARP_SIZE), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gG = cute.zipped_divide(mG, tiler_mn)
    gU = cute.zipped_divide(mU, tiler_mn)
    gY = cute.zipped_divide(mY, tiler_mn)

    mID = cute.make_identity_tensor(mG.shape)
    gID = cute.zipped_divide(mID, tiler_mn)

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gG.element_type)
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    NUM_BLOCKS = cute.size(gG, mode=[1])

    kernel = swiglu_forward_cuda_kernel(
        gG=gG, gU=gU, gY=gY, gID=gID, copy_atom=copy_atom, tiled_copy=tiled_copy, shape=mG.shape
    )

    kernel.launch(grid=(NUM_BLOCKS, 1, 1), block=(BLOCK_SIZE, 1, 1))


@xma_op(mutates_args={"y"})
def swiglu_forward_cuda(g: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> None:
    g, u, y = [torch_tensor_to_cute_tensor(i, leading_dim=1) for i in (g, u, y)]

    key = g.element_type
    function = swiglu_forward_cuda.cache.get(key, None)

    if function is None:
        function = cute.compile(swiglu_forward_cuda_jit, g, u, y)
        swiglu_forward_cuda.cache[key] = function

    function(g, u, y)


swiglu_forward_cuda.cache = {}
