# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import pytest
import torch

from xma import KernelBackend, pack_sequence, unpack_sequence

from ..utils import assert_equal_tensors, get_random_duplicated_tensors, skip_if_incompatible_kernel_backend


@pytest.mark.parametrize("size", [(7, 1000, 12, 14)])
@pytest.mark.parametrize("cu_seqlens", [[0, 70, 170, 295, 393, 412, 515, 691]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("kernel_backend", [KernelBackend.cuda, KernelBackend.triton])
@torch._dynamo.config.patch(recompile_limit=1024)
def test_pack_sequence(
    size: tuple[int],
    cu_seqlens: list[int],
    dtype: torch.dtype,
    padding_side: str,
    kernel_backend: KernelBackend,
) -> None:
    skip_if_incompatible_kernel_backend(kernel_backend)
    device = kernel_backend.get_compatible_accelerator().get_current_device()

    x_kernel, x_expected = get_random_duplicated_tensors(size, device=device, dtype=dtype)
    cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.uint32)

    z_kernel = pack_sequence(
        [x_kernel],
        cu_seqlens=cu_seqlens,
        total_tokens=cu_seqlens[-1].item(),
        padding_side=padding_side,
        kernel_backend=kernel_backend,
    )[0]

    z_expected = pack_sequence(
        [x_expected],
        cu_seqlens=cu_seqlens.to(torch.int),
        total_tokens=cu_seqlens[-1].item(),
        padding_side=padding_side,
        kernel_backend=KernelBackend.torch,
    )[0]

    z_expected.sum().backward()
    z_kernel.sum().backward()

    assert_equal_tensors(z_kernel, z_expected, True)
    assert_equal_tensors(x_kernel.grad, x_expected.grad, True)


@pytest.mark.parametrize("size", [(691, 12, 14)])
@pytest.mark.parametrize("cu_seqlens", [[0, 70, 170, 295, 393, 412, 515, 691]])
@pytest.mark.parametrize("sequence_length", [1000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("kernel_backend", [KernelBackend.cuda, KernelBackend.triton])
def test_unpack_sequence(
    size: tuple[int],
    cu_seqlens: list[int],
    sequence_length: tuple[int],
    dtype: torch.dtype,
    padding_side: str,
    kernel_backend: KernelBackend,
) -> None:
    skip_if_incompatible_kernel_backend(kernel_backend)
    device = kernel_backend.get_compatible_accelerator().get_current_device()

    x_kernel, x_expected = get_random_duplicated_tensors(size, device=device, dtype=dtype)
    cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.uint32)

    z_kernel = unpack_sequence(
        [x_kernel],
        cu_seqlens=cu_seqlens,
        batch_size=cu_seqlens.size(0) - 1,
        sequence_length=sequence_length,
        padding_side=padding_side,
        kernel_backend=kernel_backend,
    )[0]

    z_expected = unpack_sequence(
        [x_expected],
        cu_seqlens=cu_seqlens.to(torch.int),
        batch_size=cu_seqlens.size(0) - 1,
        sequence_length=sequence_length,
        padding_side=padding_side,
        kernel_backend=KernelBackend.torch,
    )[0]

    z_expected.sum().backward()
    z_kernel.sum().backward()

    assert_equal_tensors(z_kernel, z_expected, True)
    assert_equal_tensors(x_kernel.grad, x_expected.grad, True)
