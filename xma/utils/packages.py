# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

try:
    import cutlass.cute

    _IS_CUTE_DSL_AVAILABLE = True
except:
    _IS_CUTE_DSL_AVAILABLE = False


def is_cute_dsl_available() -> bool:
    return _IS_CUTE_DSL_AVAILABLE


try:
    import torch

    _IS_TORCH_AVAILABLE = True
except ImportError:
    _IS_TORCH_AVAILABLE = False


def is_torch_available() -> bool:
    return _IS_TORCH_AVAILABLE


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


if is_torch_xla_available():
    # jax_import_guard() must run before JAX is imported anywhere in the process, so JAX attaches to the same
    # libtpu runtime torch_xla is using instead of racing it for the TPU chip. This has to happen here since
    # `import jax` below is the first JAX import reached by importing the `xma` package.
    from torch_xla.experimental.custom_kernel import jax_import_guard

    jax_import_guard()


try:
    import jax

    _IS_JAX_AVAILABLE = True
except ImportError:
    _IS_JAX_AVAILABLE = False


def is_jax_available() -> bool:
    return _IS_JAX_AVAILABLE


try:
    import triton

    _IS_TRITON_AVAILABLE = True
except:
    _IS_TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    return _IS_TRITON_AVAILABLE
