# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ....constants import LIBRARY_NAME
from ....cutotune import CutoTuneConfig, cutotune
from ....jit import cpp_jit
from ....math import get_powers_of_2
from ....utils import cute_op


_FORWARD_KERNEL_NAME = "swiglu_forward_cuda"
_BACKWARD_KERNEL_NAME = "swiglu_backward_cuda"


@cutotune(
    configs=[CutoTuneConfig({"BLOCK_SIZE": BLOCK_SIZE}) for BLOCK_SIZE in get_powers_of_2(128, 1024)],
    triggers={"gate.dtype"},
)
@cute_op(f"{LIBRARY_NAME}::{_FORWARD_KERNEL_NAME}", mutates_args={"output"})
@cpp_jit()
def swiglu_forward_cuda(gate: torch.Tensor, up: torch.Tensor, output: torch.Tensor, BLOCK_SIZE: int) -> None: ...


@cutotune(
    configs=[CutoTuneConfig({"BLOCK_SIZE": BLOCK_SIZE}) for BLOCK_SIZE in get_powers_of_2(128, 1024)],
    triggers={"gate.dtype"},
)
@cute_op(f"{LIBRARY_NAME}::{_BACKWARD_KERNEL_NAME}", mutates_args={"gate_grad", "up_grad"})
@cpp_jit()
def swiglu_backward_cuda(
    gate: torch.Tensor,
    up: torch.Tensor,
    output_grad: torch.Tensor,
    gate_grad: torch.Tensor,
    up_grad: torch.Tensor,
    BLOCK_SIZE: int,
) -> None: ...
