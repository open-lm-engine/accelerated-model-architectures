# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os

import torch
from torch.utils.cpp_extension import load as load_cpp_extension


_module = None


@torch.compiler.disable
def _get_module():
    global _module

    if _module is None:
        _dir = os.path.dirname(__file__)
        _module = load_cpp_extension(
            "xma_swiglu_mps",
            sources=[os.path.join(_dir, "ops.mm")],
            extra_cflags=["-O3", "-Wall", "-std=c++17"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            verbose=True,
        )

    return _module


def swiglu_forward_mps(g: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> None:
    _get_module().swiglu_forward_mps(g, u, y)
