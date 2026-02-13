# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Generic, TypeVar


T = TypeVar("T")


class XTuneParameter(Generic[T]):
    """Marker annotation for XTune parameters."""
