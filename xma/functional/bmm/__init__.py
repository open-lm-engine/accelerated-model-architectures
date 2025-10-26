# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from ...enums import KernelBackend
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
    """computes `alpha` * (`A` @ `B`) + `beta` * `C`

    Args:
        A (torch.Tensor): `A` matrix
        B (torch.Tensor): `B` matrix
        C (torch.Tensor | None): `C` matrix, function returns `A` @ `B` if C is None
        is_A_transposed (bool, optional): whether A has shape K x M. Defaults to False.
        is_B_transposed (bool, optional): whether B has shape N x K. Defaults to False.
        alpha (float, optional): alpha. Defaults to 1.
        beta (float, optional): beta. Defaults to 1.

    Raises:
        ValueError: if unexpected `kernel_backend` is passed

    Returns:
        torch.Tensor: output tensor
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
        kernel_backend = KernelBackend.get_kernel_backend_from_device(A)

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
