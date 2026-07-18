# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from .env import get_boolean_env_variable
from .packages import (
    is_cute_dsl_available,
    is_jax_available,
    is_torch_available,
    is_torch_neuronx_available,
    is_torch_xla_available,
    is_triton_available,
)
from .ptx import get_ptx_from_triton_kernel


if is_torch_available():
    from .contiguous import ensure_contiguous
    from .debugging import print_gradient
    from .random import set_seed
    from .settings import get_triton_num_warps
    from .tensor import empty_like_contiguous, get_alignment, zeros_like_contiguous
