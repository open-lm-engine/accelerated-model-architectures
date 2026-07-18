# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from ....utils import is_torch_xla_available


if is_torch_xla_available():
    from .backward import _swiglu_backward_pallas
    from .forward import _swiglu_forward_pallas
