# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...accelerator import Accelerator, KernelBackend
from .cuda_implementation import continuous_count_cuda


@torch.no_grad()
def continuous_count(x: torch.Tensor, bins: int, *, kernel_backend: KernelBackend | None = None) -> torch.Tensor:
    """
    counts the number of occurances of the values [0, 1, ..., `bins`) in the input tensor (`bins` is excluded).
    NOTE: the user is responsible for ensuring that the values lie in the valid range, any values outside this
    range are ignored and not counted.

    :param x: input tensor
    :type x: torch.Tensor
    :param bins: values [0, 1, ..., `bins`) are counted (`bins` is excluded)
    :type bins: int
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output tensor
    :rtype: Tensor
    """

    if bins == 1:
        return torch.tensor([x.numel()], dtype=torch.uint32, device=x.device)

    assert x.dim() == 1, "x should be 1-dimensional"
    assert x.dtype in [torch.int32, torch.long]

    if kernel_backend is None:
        kernel_backend = Accelerator.get_kernel_backend()
    else:
        assert kernel_backend.verify_accelerator()

    if kernel_backend == KernelBackend.torch:
        output = x.bincount(minlength=bins).to(torch.uint32)
    elif kernel_backend in [KernelBackend.cuda, KernelBackend.triton]:
        output = torch.empty(bins, dtype=torch.uint32, device=x.device)
        continuous_count_cuda(x=x, output=output, E=bins, THREAD_BLOCK_CLUSTER_SIZE=1, BLOCK_SIZE=1024)
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return output
