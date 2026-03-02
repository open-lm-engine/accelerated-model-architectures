# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...accelerator import Accelerator, KernelBackend
from ...utils import is_triton_available


if is_triton_available():
    from .triton_implementation import norm_triton


def norm(
    x: torch.Tensor,
    multiplier: float | None = None,
    p: int | str = 2,
    output_dtype: torch.dtype = torch.float32,
    *,
    kernel_backend: KernelBackend | None = None,
) -> torch.Tensor:
    """
    computes norm of a vector

    :param x: input activation
    :type x: torch.Tensor
    :param multiplier: if not None, pre-multiplies `x` with `multiplier`. Defaults to None.
    :type multiplier: float | None
    :param p: norm type. can be integer >= 1 or `inf`
    :type p: int | str
    :param output_dtype: output dtype
    :type torch.dtype
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output activation
    :rtype: Tensor
    """

    assert x.dim() == 2

    if kernel_backend is None:
        kernel_backend = Accelerator.get_kernel_backend()
    else:
        assert kernel_backend.verify_accelerator()

    if kernel_backend in [KernelBackend.cuda, KernelBackend.triton]:
        B = x.size(0)
        is_p_inf = p == "inf"

        y = torch.empty(B, device=x.device, dtype=output_dtype)
        norm_triton(x=x, y=y, multiplier=multiplier, p=None if is_p_inf else p, is_p_inf=is_p_inf)
    elif kernel_backend == KernelBackend.torch:
        if multiplier not in [None, 1]:
            x = x * multiplier

        if p == "inf":
            y = x.abs().max(dim=-1)[0]
        else:
            y = torch.norm(x, p=p, dim=-1)

        y = y.to(output_dtype)
    else:
        raise NotImplementedError(f"unexpected kernel_backend ({kernel_backend})")

    return y
