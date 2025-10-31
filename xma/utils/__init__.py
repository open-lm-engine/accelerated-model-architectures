# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from .contiguous import ensure_contiguous, ensure_same_strides
from .debugging import print_gradient
from .device import device_synchronize, get_sm_count, is_hip, is_nvidia_gpu
from .env import get_boolean_env_variable
from .packages import is_cute_dsl_available, is_triton_available
from .ptx import get_ptx_from_triton_kernel
from .random import set_seed
from .settings import get_triton_num_warps
from .tensor import empty_like_contiguous, get_num_elements_and_hidden_size, zeros_like_contiguous
