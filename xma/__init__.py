# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from .accelerator import Accelerator, KernelBackend
from .counters import enable_counters, get_counter_value, reset_counters
from .functional import (
    bmm,
    continuous_count,
    cross_entropy,
    fused_linear_cross_entropy,
    fused_residual_add_rmsnorm,
    pack_sequence,
    rmsnorm,
    softmax,
    swiglu,
    swiglu_packed,
    unpack_sequence,
)
from .inductor import enable_kernels, init_inductor
from .layers import GRU, RNN, HiPPO, LinearAttention, MoE
from .math import ceil_divide, divide_if_divisible, get_powers_of_2
from .utils import get_ptx_from_triton_kernel, get_triton_num_warps, set_seed
from .xtuner import XTuneConfig, XTuneParameter, get_cartesian_product_xtune_configs, get_xtune_cache, xtune
