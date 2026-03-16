# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

try:
    import causal_conv1d

    _IS_CAUSAL_CONV1D_AVAILABLE = True
except ImportError:
    _IS_CAUSAL_CONV1D_AVAILABLE = False


def is_causal_conv1d_available() -> bool:
    return _IS_CAUSAL_CONV1D_AVAILABLE


try:
    import cutlass.cute

    _IS_CUTE_DSL_AVAILABLE = True
except:
    _IS_CUTE_DSL_AVAILABLE = False


def is_cute_dsl_available() -> bool:
    return _IS_CUTE_DSL_AVAILABLE


try:
    import torch_neuronx

    _IS_TORCH_NEURONX_AVAILABLE = True
except:
    _IS_TORCH_NEURONX_AVAILABLE = False


def is_torch_neuronx_available() -> bool:
    return _IS_TORCH_NEURONX_AVAILABLE


try:
    import torch_xla

    _IS_TORCH_XLA_AVAILABLE = True
except ImportError:
    _IS_TORCH_XLA_AVAILABLE = False


def is_torch_xla_available() -> bool:
    return _IS_TORCH_XLA_AVAILABLE


try:
    import triton

    _IS_TRITON_AVAILABLE = True
except:
    _IS_TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    return _IS_TRITON_AVAILABLE
