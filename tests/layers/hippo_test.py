# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn as nn
from parameterized import parameterized

from xma import Accelerator, HiPPO, KernelBackend, set_seed

from ..test_commons import TestCommons


_SEED = 42


class HiPPOTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [KernelBackend.triton],  # KernelBackend
            [torch.float32, torch.float16],  # dtype
            [(4, 1024, None), (None, None, [0, 7, 19, 27, 93])],  # B, S, cu_seqlens
            [(8, 37, 8)],  # state_head_dim, hidden_size
            [False, True],  # has_input_state
            [False, True],  # is_compiling
        )
    )
    def test_hippo(
        self,
        kernel_backend: KernelBackend,
        dtype: torch.dtype,
        input_shape: tuple[int, int, list[int]],
        problem_shape: tuple[int, int],
        has_input_state: bool,
        is_compiling: bool,
    ) -> None:
        self.skip_if_incompatible_kernel_backend(kernel_backend)
        device = kernel_backend.get_compatible_accelerator().get_current_device()

        set_seed(_SEED)

        state_head_dim, hidden_size, num_weight_heads = problem_shape

        B, S, cu_seqlens = input_shape
        max_seqlen = None

        if B is None:
            cu_seqlens = torch.tensor(cu_seqlens, device=device)
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            B = cu_seqlens.size(0) - 1

        x_kernel, x_torch, input_state_kernel, input_state_torch = self._get_packed_tensor_inputs(
            batch_size=B,
            sequence_length=S if cu_seqlens is None else None,
            total_tokens=None if cu_seqlens is None else cu_seqlens[-1],
            hidden_size=hidden_size,
            state_head_dim=state_head_dim,
            has_input_state=has_input_state,
            dtype=dtype,
            device=device,
        )

        with torch.device(device):
            hippo = HiPPO(state_head_dim=state_head_dim, measure="legS").to(dtype)

        hippo_torch = hippo
        hippo_kernel = hippo

        if is_compiling:
            hippo_kernel = torch.compile(hippo_kernel, fullgraph=True)

        y_kernel, output_state_kernel = hippo_kernel(
            input=x_kernel,
            input_state=input_state_kernel,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=KernelBackend.triton,
        )

        y_torch, output_state_torch = hippo_torch(
            input=x_torch,
            input_state=input_state_torch,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=KernelBackend.torch,
        )

        self.assert_equal_tensors(y_kernel, y_torch, False)
        self.assert_equal_tensors(output_state_kernel, output_state_torch, False)

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [KernelBackend.torch],  # KernelBackend
            TestCommons.get_dtypes(),  # dtype
            [[0, 7, 19, 27, 93]],  # cu_seqlens
            [(8, 4, 8), (8, 8, 4), (9, 7, 7)],  # state_head_dim, num_input_heads, num_weight_heads
            [False, True],  # has_input_state
        )
    )
    def test_hippo_varlen_torch(
        self,
        kernel_backend: KernelBackend,
        dtype: torch.dtype,
        cu_seqlens: list[int],
        snn: tuple[int, int, int],
        has_input_state: bool,
    ) -> None:
        if Accelerator.get_accelerator() != Accelerator.cuda:
            self.skipTest("Sufficient to run on CUDA device")

        self.skip_if_incompatible_kernel_backend(kernel_backend)
        device = kernel_backend.get_compatible_accelerator().get_current_device()

        set_seed(_SEED)

        batch_size = len(cu_seqlens) - 1
        cu_seqlens = torch.tensor(cu_seqlens, device=device)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        state_head_dim, num_input_heads, num_weight_heads = snn
        num_heads = max(num_input_heads, num_weight_heads)
        state_size = state_head_dim * num_heads

        x_packed_kernel, x_packed_torch, input_state_kernel, input_state_torch = self._get_packed_tensor_inputs(
            batch_size=batch_size,
            sequence_length=None,
            total_tokens=cu_seqlens[-1],
            state_size=state_size,
            has_input_state=has_input_state,
            dtype=dtype,
            device=device,
        )

        with torch.device(device):
            hippo = HiPPO(state_head_dim=state_head_dim, measure="legS").to(dtype)

        y_kernel, _ = hippo(
            input=x_packed_kernel,
            input_state=input_state_kernel,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=KernelBackend.torch,
        )

        y_torch = []
        for i in range(batch_size):
            y, _ = hippo(
                input=x_packed_torch[cu_seqlens[i] : cu_seqlens[i + 1]].unsqueeze(0),
                input_state=input_state_torch[i].unsqueeze(0) if has_input_state else None,
                kernel_backend=KernelBackend.torch,
            )
            y_torch.append(y.squeeze(0))
        y_torch = torch.cat(y_torch)

        self.assert_equal_tensors(y_kernel, y_torch, False)

    def _get_packed_tensor_inputs(
        self,
        batch_size: int,
        sequence_length: int | None,
        total_tokens: int | None,
        hidden_size: int,
        state_head_dim: int,
        has_input_state: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor | None]:
        x_kernel, x_torch = self.get_random_duplicated_tensors(
            ((batch_size, sequence_length, hidden_size) if total_tokens is None else (total_tokens, hidden_size)),
            device=device,
            dtype=dtype,
            std=0.01,
        )

        input_state_kernel = None
        input_state_torch = None
        if has_input_state:
            input_state_kernel, input_state_torch = self.get_random_duplicated_tensors(
                (batch_size, hidden_size, state_head_dim), device=device, dtype=dtype, std=0.01
            )

        return x_kernel, x_torch, input_state_kernel, input_state_torch
