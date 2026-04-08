# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ....custom_op import xma_op
from ....jit import cpp_jit


@xma_op(mutates_args={"output"})
@cpp_jit(is_cuda=True)
def _pack_unpack_sequence_cuda(
    x: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padding_side: str,
    pack: bool,
    BLOCK_SIZE: int,
) -> None: ...
