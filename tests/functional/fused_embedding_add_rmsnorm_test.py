# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from parameterized import parameterized

from xma import KernelBackend, fused_embedding_residual_add_rmsnorm, set_seed

from ..test_commons import TestCommons


_EPSILON = 1e-5
_SEED = 42


def _get_sizes() -> list[tuple]:
    """Returns list of (batch_size, vocab_size, hidden_size) tuples."""
    sizes = []
    # Test various hidden sizes
    for hidden_size in [64, 128, 256, 512, 1024]:
        sizes.append((32, 1000, hidden_size))
    # Test various batch sizes
    for batch_size in [1, 8, 16, 64, 128]:
        sizes.append((batch_size, 1000, 256))
    return sizes


class FusedEmbeddingRMSNormTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            _get_sizes(),  # (batch_size, vocab_size, hidden_size)
            [KernelBackend.triton],  # KernelBackend
            [torch.float32],  # dtype
            [False],  # has_residual
            [True, False],  # has_rmsnorm_weight
            [None, 0.9],  # multiplier
            [
                fused_embedding_residual_add_rmsnorm,
                # torch.compile(fused_embedding_residual_add_rmsnorm, fullgraph=True),
            ],  # function
        )
    )
    def test_fused_embedding_rmsnorm(
        self,
        size: tuple[int, int, int],
        kernel_backend: KernelBackend,
        dtype: torch.dtype,
        has_residual: bool,
        has_rmsnorm_weight: bool,
        multiplier: float | None,
        function: Callable,
    ) -> None:
        self.skip_if_incompatible_kernel_backend(kernel_backend)
        device = kernel_backend.get_compatible_accelerator().get_current_device()

        set_seed(_SEED)

        batch_size, vocab_size, hidden_size = size

        # x is integer indices (token IDs)
        x = torch.randint(0, vocab_size, (batch_size,), device=device, dtype=torch.long)

        # weight1 is the embedding table (V, H)
        weight1_kernel = torch.randn(vocab_size, hidden_size, device=device, dtype=dtype, requires_grad=True)
        weight1_expected = weight1_kernel.clone().detach().requires_grad_()

        # weight2 is the rmsnorm weight (H,)
        if has_rmsnorm_weight:
            weight2_kernel = torch.randn(hidden_size, device=device, dtype=dtype, requires_grad=True)
            weight2_expected = weight2_kernel.clone().detach().requires_grad_()
        else:
            weight2_kernel = None
            weight2_expected = None

        # residual has shape (B, H) matching embedded output
        if has_residual:
            residual_kernel, residual_expected = self.get_random_duplicated_tensors(
                (batch_size, hidden_size), device=device, dtype=dtype
            )
        else:
            residual_kernel = None
            residual_expected = None

        # Kernel implementation
        z_kernel, r_kernel = function(
            x=x,
            residual=residual_kernel,
            weight1=weight1_kernel,
            weight2=weight2_kernel,
            eps=_EPSILON,
            multiplier=multiplier,
            kernel_backend=kernel_backend,
        )

        # Reference implementation
        z_expected, r_expected = fused_embedding_residual_add_rmsnorm(
            x=x,
            residual=residual_expected,
            weight1=weight1_expected,
            weight2=weight2_expected,
            eps=_EPSILON,
            multiplier=multiplier,
            kernel_backend=KernelBackend.torch,
        )

        # Forward check
        self.assert_equal_tensors(z_kernel, z_expected, False, atol_float32=1.4e-4, rtol_float32=0)

        if has_residual:
            self.assert_equal_tensors(r_kernel, r_expected, False, atol_float32=1.4e-4, rtol_float32=0)

        # Backward
        assert not has_residual, "backward is not supported for has_residual"
        if has_residual:
            loss_kernel = z_kernel.sum() * 2 + r_kernel.sum() * 3
            loss_expected = z_expected.sum() * 2 + r_expected.sum() * 3
        else:
            loss_kernel = z_kernel.sum()
            loss_expected = z_expected.sum()

        loss_kernel.backward()
        loss_expected.backward()

        # Check embedding weight gradients
        self.assert_equal_tensors(
            weight1_kernel.grad, weight1_expected.grad, False, atol_float32=1.5e-4, rtol_float32=0
        )

        # Check rmsnorm weight gradients
        if has_rmsnorm_weight:
            self.assert_equal_tensors(
                weight2_kernel.grad, weight2_expected.grad, False, atol_float32=1.1e-4, rtol_float32=0
            )

        # Check residual gradients
        if has_residual:
            self.assert_equal_tensors(
                residual_kernel.grad, residual_expected.grad, False, atol_float32=1.6e-4, rtol_float32=0
            )

    # @parameterized.expand(
    #     TestCommons.make_args_matrix(
    #         [(16, 500, 128)],  # (batch_size, vocab_size, hidden_size)
    #         [KernelBackend.triton],  # KernelBackend
    #         [torch.float32],  # dtype
    #     )
    # )
    # def test_fused_embedding_rmsnorm_2d_input(
    #     self,
    #     size: tuple[int, int, int],
    #     kernel_backend: KernelBackend,
    #     dtype: torch.dtype,
    # ) -> None:
    #     """Test with 2D input (batch_size, seq_len) for sequence models."""
    #     self.skip_if_incompatible_kernel_backend(kernel_backend)
    #     device = kernel_backend.get_compatible_accelerator().get_current_device()

    #     set_seed(_SEED)

    #     batch_size, vocab_size, hidden_size = size
    #     seq_len = 32

    #     # x is 2D: (B, S) token indices
    #     x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)

    #     # weight1 is the embedding table (V, H)
    #     weight1_kernel = torch.randn(vocab_size, hidden_size, device=device, dtype=dtype, requires_grad=True)
    #     weight1_expected = weight1_kernel.clone().detach().requires_grad_()

    #     # weight2 is the rmsnorm weight (H,)
    #     weight2_kernel = torch.randn(hidden_size, device=device, dtype=dtype, requires_grad=True)
    #     weight2_expected = weight2_kernel.clone().detach().requires_grad_()

    #     # residual has shape (B, S, H)
    #     residual_kernel, residual_expected = self.get_random_duplicated_tensors(
    #         (batch_size, seq_len, hidden_size), device=device, dtype=dtype
    #     )
    #     function = fused_embedding_residual_add_rmsnorm
    #     # Kernel implementation
    #     z_kernel, r_kernel = function(
    #         x=x,
    #         residual=residual_kernel,
    #         weight1=weight1_kernel,
    #         weight2=weight2_kernel,
    #         eps=_EPSILON,
    #         kernel_backend=kernel_backend,
    #     )

    #     # Reference implementation
    #     z_expected, r_expected = fused_embedding_residual_add_rmsnorm(
    #         x=x,
    #         residual=residual_expected,
    #         weight1=weight1_expected,
    #         weight2=weight2_expected,
    #         eps=_EPSILON,
    #         multiplier=None,
    #         kernel_backend=KernelBackend.torch,
    #     )

    #     # Forward check
    #     self.assert_equal_tensors(z_kernel, z_expected, False, atol_float32=1.4e-4, rtol_float32=0)
    #     self.assert_equal_tensors(r_kernel, r_expected, False, atol_float32=1.4e-4, rtol_float32=0)

    #     # Backward
    #     loss_kernel = z_kernel.sum() + r_kernel.sum()
    #     loss_expected = z_expected.sum() + r_expected.sum()

    #     loss_kernel.backward()
    #     loss_expected.backward()

    #     # Check gradients
    #     self.assert_equal_tensors(
    #         weight1_kernel.grad, weight1_expected.grad, False, atol_float32=1.5e-4, rtol_float32=0
    #     )
    #     self.assert_equal_tensors(
    #         weight2_kernel.grad, weight2_expected.grad, False, atol_float32=1.1e-4, rtol_float32=0
    #     )
    #     self.assert_equal_tensors(
    #         residual_kernel.grad, residual_expected.grad, False, atol_float32=1.6e-4, rtol_float32=0
    #     )
