# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch

from ....custom_op import xma_op
from ....jit import cpp_jit


@xma_op(mutates_args={"output"})
@cpp_jit(is_cuda=True)
def _pack_unpack_sequence_cuda(
    x: torch.Tensor,
    y: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padding_side: str,
    pack: bool,
) -> None: ...
