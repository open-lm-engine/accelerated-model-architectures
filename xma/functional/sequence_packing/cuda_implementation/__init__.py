# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ....custom_op import xma_op
from ....cute_dsl_utils import cuda_jit


@xma_op(mutates_args={"output"})
@cuda_jit()
def pack_unpack_sequence_cuda(
    x: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padding_side: str,
    pack: bool,
    BLOCK_SIZE: int,
) -> None: ...
