# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...accelerator import Accelerator, KernelBackend
from ...utils import is_triton_available


if is_triton_available():
    from .triton_implementation import bmm_triton


def bmm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor | None,
    is_A_transposed: bool = False,
    is_B_transposed: bool = False,
    alpha: float = 1,
    beta: float = 1,
    *,
    kernel_backend: KernelBackend | None = None,
) -> torch.Tensor:
    """
    computes alpha * (A @ B) + beta * C`

    :param A: `A` matrix
    :type A: torch.Tensor
    :param B: `B` matrix
    :type B: torch.Tensor
    :param C: `C` matrix, function returns `A @ B` if `C` is None
    :type C: torch.Tensor | None
    :param is_A_transposed: whether `A` has shape K x M. Defaults to False.
    :type is_A_transposed: bool
    :param is_B_transposed: whether `B` has shape N x K. Defaults to False.
    :type is_B_transposed: bool
    :param alpha: alpha. Defaults to 1.
    :type alpha: float
    :param beta: beta. Defaults to 1.
    :type beta: float
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: output tensor
    :rtype: Tensor
    """

    assert A.dim() == 3
    assert B.dim() == 3

    L, M, K = A.size()
    if is_A_transposed:
        M, K = K, M

    assert B.size(2 if is_B_transposed else 1) == K
    N = B.size(1 if is_B_transposed else 2)

    if beta == 0:
        assert C is None
    else:
        assert C is not None
        assert C.size() == (L, M, N)

    if kernel_backend is None:
        kernel_backend = Accelerator.get_kernel_backend()
    else:
        assert kernel_backend.verify_accelerator()

    if kernel_backend == KernelBackend.torch:
        if is_A_transposed:
            A = A.transpose(1, 2)

        if is_B_transposed:
            B = B.transpose(1, 2)

        if beta == 0:
            D = torch.bmm(A, B)
            if alpha != 1:
                D = alpha * D
        else:
            D = torch.baddbmm(C, A, B, alpha=alpha, beta=beta)
    elif kernel_backend in [KernelBackend.cuda, KernelBackend.triton]:
        D = torch.empty(L, M, N, dtype=A.dtype, device=A.device)

        bmm_triton(
            A=A,
            B=B,
            C=C,
            D=D,
            is_A_transposed=is_A_transposed,
            is_B_transposed=is_B_transposed,
            alpha=alpha,
            beta=beta,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")

    return D
