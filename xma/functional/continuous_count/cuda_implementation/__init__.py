# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ....custom_op import xma_op
from ....jit import cpp_jit
from ....utils import ensure_contiguous


@ensure_contiguous
@xma_op(mutates_args={"output"})
@cpp_jit(is_cuda=True)
def continuous_count_cuda(
    x: torch.Tensor, output: torch.Tensor, E: int, THREAD_BLOCK_CLUSTER_SIZE: int, BLOCK_SIZE: int
) -> None: ...
