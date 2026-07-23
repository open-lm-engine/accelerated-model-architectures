# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from ...utils import is_haliax_available
from .op import linear_attention_jax


if is_haliax_available():
    from .module import LinearAttentionJAX
