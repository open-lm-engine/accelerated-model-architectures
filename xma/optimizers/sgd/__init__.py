# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

# Register the DTensor sharding strategy for _single_tensor_sgd_triton.
# This must be imported after op.py (which registers the custom op via
# @xma_op) and is guarded so that importing the SGD module without
# Triton installed does not attempt to look up an unregistered op.
from ...utils.packages import is_triton_available
from .module import SGD
from .op import sgd


if is_triton_available():
    from . import sharding  # noqa: F401
