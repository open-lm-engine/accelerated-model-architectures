# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from torch.library import custom_op

from ....constants import LIBRARY_NAME
from ....jit import cpp_jit
from .forward import swiglu_forward_cuda


@custom_op(f"{LIBRARY_NAME}::swiglu_backward_cuda", mutates_args={"gate_grad", "up_grad"})
@cpp_jit()
def swiglu_backward_cuda(
    gate: torch.Tensor,
    up: torch.Tensor,
    output_grad: torch.Tensor,
    gate_grad: torch.Tensor,
    up_grad: torch.Tensor,
    BLOCK_SIZE: int,
) -> None: ...
