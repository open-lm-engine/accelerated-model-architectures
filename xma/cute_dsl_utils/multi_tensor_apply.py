# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Any, Callable

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, range_constexpr

from ..constants import LOG_WARP_SIZE, WARP_SIZE
from .boundary import lane_boundary
from .elementwise import _load, _store
from .utils import get_fake_cute_tensor


class MultiTensorApplyCUDAKernel:
    def __init__(
        self, BLOCK_SIZE: int, M: int, depth_x: int, depth_y: int, num_tensors: int
    ) -> MultiTensorApplyCUDAKernel:
        self.BLOCK_SIZE = BLOCK_SIZE
        self.M = M
        self.depth_x = depth_x
        self.depth_y = depth_y
        self.num_tensors = num_tensors

    def compute(self, xs: list[cute.TensorSSA], scalars: list[Float32]) -> list[cute.TensorSSA | None]:
        raise NotImplementedError

    @cute.kernel
    def kernel(
        self,
        gXss: list[list[cute.Tensor]],
        gYss: list[list[cute.Tensor]],
        gC: cute.Tensor,
        copy_atom_Xs: list[cute.CopyAtom],
        copy_atom_Ys: list[cute.CopyAtom],
        tiled_copy_Xs: list[cute.TiledCopy],
        shape: cute.Shape,
        total_tiles_per_tensor: Int32,
        grand_total_tiles: Int32,
        scalars: list[Float32],
    ) -> None:
        BLOCK_ID, _, _ = cute.arch.block_idx()
        NUM_BLOCKS, _, _ = cute.arch.grid_dim()
        THREAD_ID, _, _ = cute.arch.thread_idx()

        while BLOCK_ID < grand_total_tiles:
            for t in range_constexpr(self.num_tensors):
                local_tile = BLOCK_ID - t * total_tiles_per_tensor

                if local_tile >= 0:
                    if local_tile < total_tiles_per_tensor:
                        block_coord = ((None, None), local_tile)

                        thr_copy, rC, is_within_boundary = lane_boundary(
                            gC=gC,
                            tiled_copy=tiled_copy_Xs[0],
                            block_coord=block_coord,
                            THREAD_ID=THREAD_ID,
                            shape=shape,
                        )

                        xs = [
                            _load(
                                gX=gXss[d][t],
                                rC=rC,
                                thr_copy=thr_copy,
                                copy_atom=copy_atom_Xs[d],
                                block_coord=block_coord,
                                is_within_boundary=is_within_boundary,
                            )
                            for d in range_constexpr(self.depth_x)
                        ]

                        ys = self.compute(xs, scalars)

                        for d in range_constexpr(self.depth_y):
                            _store(
                                gY=gYss[d][t],
                                y=ys[d],
                                rC=rC,
                                thr_copy=thr_copy,
                                copy_atom=copy_atom_Ys[d],
                                block_coord=block_coord,
                                is_within_boundary=is_within_boundary,
                            )

            BLOCK_ID += NUM_BLOCKS

    @cute.jit
    def __call__(
        self,
        mXss: list[list[cute.Tensor]],
        mYss: list[list[cute.Tensor]],
        scalars: list[Float32],
        stream: cuda.CUstream,
    ) -> None:
        vector_size = min(
            [128 // mXs[0].element_type.width for mXs in mXss] + [128 // mYs[0].element_type.width for mYs in mYss]
        )

        thr_layout = cute.make_ordered_layout((self.BLOCK_SIZE >> LOG_WARP_SIZE, WARP_SIZE), order=(1, 0))
        val_layout = cute.make_ordered_layout((self.M, vector_size), order=(1, 0))
        tiler_mn, _ = cute.make_layout_tv(thr_layout, val_layout)

        shape = mXss[0][0].shape
        gC = cute.zipped_divide(cute.make_identity_tensor(shape), tiler_mn)

        gXss = [[cute.zipped_divide(mX, tiler_mn) for mX in mXs] for mXs in mXss]
        gYss = [[cute.zipped_divide(mY, tiler_mn) for mY in mYs] for mYs in mYss]

        copy_atom_Xs = [cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mXs[0].element_type) for mXs in mXss]
        copy_atom_Ys = [cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mYs[0].element_type) for mYs in mYss]

        tiled_copy_Xs = [cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout) for copy_atom in copy_atom_Xs]

        total_tiles_per_tensor = cute.size(gC, mode=[1])
        grand_total_tiles = total_tiles_per_tensor * self.num_tensors

        NUM_BLOCKS = torch.cuda.get_device_properties(0).multi_processor_count

        self.kernel(
            gXss=gXss,
            gYss=gYss,
            gC=gC,
            copy_atom_Xs=copy_atom_Xs,
            copy_atom_Ys=copy_atom_Ys,
            tiled_copy_Xs=tiled_copy_Xs,
            shape=shape,
            total_tiles_per_tensor=total_tiles_per_tensor,
            grand_total_tiles=grand_total_tiles,
            scalars=scalars,
        ).launch(grid=(NUM_BLOCKS, 1, 1), block=(self.BLOCK_SIZE, 1, 1), stream=stream)


def _build_fake_tensor_groups(
    example_tensors_list: tuple[tuple[torch.Tensor]], divisibility_list_list: tuple[tuple[int]]
) -> list[list[cute.Tensor]]:
    fake_tensors = []

    for example_tensors, divisibility_list in zip(example_tensors_list, divisibility_list_list):
        fake_tensors.append(
            [
                get_fake_cute_tensor(
                    dtype=tensor.dtype, shape=(cute.sym_int(), cute.sym_int(divisibility=div)), divisibility=div
                )
                for tensor, div in zip(example_tensors, divisibility_list)
            ]
        )

    return fake_tensors


def get_compiled_multi_tensor_apply_cuda_kernel(
    caller_op: Callable,
    key: Any,
    kernel_class: type,
    example_x_tensors_list: tuple[tuple[torch.Tensor]],
    example_y_tensors_list: tuple[tuple[torch.Tensor]],
    divisibility_x_list_list: tuple[tuple[int]],
    divisibility_y_list_list: tuple[tuple[int]],
    scalars: list[float],
    stream: cuda.CUstream,
) -> MultiTensorApplyCUDAKernel:
    if not hasattr(caller_op, "cache"):
        caller_op.cache = {}

    kernel = caller_op.cache.get(key)

    if kernel is None:
        fake_mXss = _build_fake_tensor_groups(example_x_tensors_list, divisibility_x_list_list)
        fake_mYss = _build_fake_tensor_groups(example_y_tensors_list, divisibility_y_list_list)

        kernel = cute.compile(kernel_class(), fake_mXss, fake_mYss, scalars, stream, options="--enable-tvm-ffi")
        caller_op.cache[key] = kernel

    return kernel


def multi_tensor_apply(
    caller_op: Callable,
    key: Any,
    kernel_class: type,
    x_tensor_lists: list[list[torch.Tensor]],
    y_tensor_lists: list[list[torch.Tensor]],
    divisibility_list: list[int],
    scalars: list[float],
    stream: cuda.CUstream,
) -> None:
    depth_x = len(x_tensor_lists)
    depth_y = len(y_tensor_lists)
    num_tensors = len(x_tensor_lists[0])
    shape = x_tensor_lists[0][0].shape

    for tensors in x_tensor_lists + y_tensor_lists:
        assert len(tensors) == num_tensors, "all tensor lists must have the same length"

        for tensor in tensors:
            assert tensor.shape == shape, f"all tensors must have the same shape, got {shape} and {tensor.shape}"

    compiled_kernel = get_compiled_multi_tensor_apply_cuda_kernel(
        caller_op=caller_op,
        key=key,
        kernel_class=lambda: kernel_class(depth_x=depth_x, depth_y=depth_y, num_tensors=num_tensors),
        example_x_tensors_list=tuple(tuple(mXs) for mXs in x_tensor_lists),
        example_y_tensors_list=tuple(tuple(mYs) for mYs in y_tensor_lists),
        divisibility_x_list_list=tuple(tuple(divisibility_list) for _ in range(depth_x)),
        divisibility_y_list_list=tuple(tuple(divisibility_list) for _ in range(depth_y)),
        scalars=scalars,
        stream=stream,
    )

    compiled_kernel(x_tensor_lists, y_tensor_lists, scalars, stream)
