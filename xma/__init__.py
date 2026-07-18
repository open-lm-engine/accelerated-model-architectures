# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from .accelerator import Accelerator, KernelBackend
from .counters import enable_counters, get_counter_value, reset_counters
from .math import ceil_divide, divide_if_divisible, get_powers_of_2
from .utils import get_ptx_from_triton_kernel, is_jax_available, is_torch_available


if is_jax_available():
    from .functional_jax import swiglu_jax


if is_torch_available():
    from .autotuner import AutotuneConfig, autotune, get_cartesian_product_autotune_configs
    from .functional import (
        bmm,
        continuous_count,
        cross_entropy,
        fused_linear_cross_entropy,
        fused_residual_add_rmsnorm,
        p_norm,
        pack_sequence,
        rmsnorm,
        softmax,
        swiglu,
        swiglu_packed,
        unpack_sequence,
    )
    from .inductor import enable_kernels
    from .layers import GRU, M2RNN, RNN, LinearAttention, MoE, gru, linear_attention, m2rnn, rnn
    from .optimizers import SGD, sgd
    from .utils import is_jax_available, is_torch_available, set_seed
