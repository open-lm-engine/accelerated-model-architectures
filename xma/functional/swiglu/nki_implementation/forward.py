# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import neuronxcc.nki.language as nl
import torch
from torch_neuronx import TorchNeuronNKIKernel

from ....custom_op import xma_op
from ....math import ceil_divide


BLOCK_SIZE_B = 128
BLOCK_SIZE_H = 512


def swiglu_forward_nki_kernel(g_ptr, u_ptr, y_ptr):

    B, H = g_ptr.shape

    BLOCK_ID_B = nl.program_id(0)
    BLOCK_ID_H = nl.program_id(1)

    BLOCK_B = BLOCK_ID_B * BLOCK_SIZE_B + nl.arange(BLOCK_SIZE_B)[:, None]
    BLOCK_H = BLOCK_ID_H * BLOCK_SIZE_H + nl.arange(BLOCK_SIZE_H)[None, :]

    MASK_B = BLOCK_B < B
    MASK_H = BLOCK_H < H

    MASK = MASK_B & MASK_H

    g = nl.load(g_ptr[BLOCK_B, BLOCK_H], mask=MASK)
    u = nl.load(u_ptr[BLOCK_B, BLOCK_H], mask=MASK)

    g = nl.copy(g, dtype=nl.tfloat32)
    y = u * g * nl.sigmoid(g)

    nl.store(y_ptr[BLOCK_B, BLOCK_H], y, mask=MASK)


_CACHE = {}


@xma_op(mutates_args={"y"})
def swiglu_forward_nki(g: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> None:
    B, H = g.size()

    compile_key = (B, H, g.dtype)
    kernel = _CACHE.get(compile_key, None)

    if kernel is None:
        kernel = TorchNeuronNKIKernel(
            func=swiglu_forward_nki_kernel,
            grid=(ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H)),
            kernel_return=False,
            return_tensor_overrides=(y,),
        )

        _CACHE[compile_key] = kernel

    kernel(g, u, y)
