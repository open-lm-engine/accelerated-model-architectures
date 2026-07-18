# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

from ....custom_op import xma_op
from ....functional_jax.swiglu.pallas_implementation import _swiglu_backward_pallas_jit


def _fake_func(g: torch.Tensor, u: torch.Tensor, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert g.is_contiguous()
    assert u.is_contiguous()
    assert dy.is_contiguous()

    return torch.empty_like(g), torch.empty_like(u)


_CACHE = None


@xma_op(mutates_args={}, fake_func=_fake_func)
def _swiglu_backward_pallas(g: torch.Tensor, u: torch.Tensor, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert g.is_contiguous()
    assert u.is_contiguous()
    assert dy.is_contiguous()

    global _CACHE

    if _CACHE is None:
        _CACHE = make_kernel_from_pallas(
            _swiglu_backward_pallas_jit, lambda g, u, dy: [(g.shape, g.dtype), (g.shape, g.dtype)]
        )

    return _CACHE(g, u, dy)
