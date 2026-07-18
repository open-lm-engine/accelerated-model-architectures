# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

from ....custom_op import xma_op
from ....functional_jax.swiglu.pallas_implementation import _swiglu_forward_pallas_jit


def _fake_function(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    assert g.is_contiguous()
    assert u.is_contiguous()

    return torch.empty_like(g)


_CACHE = None


@xma_op(mutates_args={}, fake_func=_fake_function)
def _swiglu_forward_pallas(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    assert g.is_contiguous()
    assert u.is_contiguous()

    global _CACHE

    if _CACHE is None:
        _CACHE = make_kernel_from_pallas(_swiglu_forward_pallas_jit, lambda g, u: [(g.shape, g.dtype)])

    return _CACHE(g, u)
