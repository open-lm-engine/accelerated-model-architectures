# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from contextlib import nullcontext

import torch
import torch.nn as nn
from parameterized import parameterized

from xma import RNN, Accelerator, KernelBackend, set_seed

from ..test_commons import TestCommons


_SEED = 42


class RNNTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [KernelBackend.triton],  # KernelBackend
            [torch.float32, torch.float16],  # dtype
            [(4, 1024, None), (None, None, [0, 7, 19, 27, 93])],  # B, S, cu_seqlens
            [(8, 4, 8), (8, 8, 4), (9, 7, 7)],  # state_head_dim, num_input_heads, num_weight_heads
            [False, True],  # has_input_state
            [False, True],  # is_compiling
        )
    )
    def test_rnn(
        self,
        kernel_backend: KernelBackend,
        dtype: torch.dtype,
        input_shape: tuple[int, int, list[int]],
        snn: tuple[int, int, int],
        has_input_state: bool,
        is_compiling: bool,
    ) -> None:
        self.skip_if_incompatible_kernel_backend(kernel_backend)
        device = kernel_backend.get_compatible_accelerator().get_current_device()

        set_seed(_SEED)

        state_head_dim, num_input_heads, num_weight_heads = snn
        num_heads = max(num_input_heads, num_weight_heads)
        state_size = state_head_dim * num_heads

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
            state_size=state_size,
            has_input_state=has_input_state,
            dtype=dtype,
            device=device,
        )

        with torch.device(device):
            rnn = RNN(
                input_size=state_size,
                state_head_dim=state_head_dim,
                output_size=state_size,
                num_input_heads=num_input_heads,
                num_weight_heads=num_weight_heads,
                add_bias=False,
                gradient_clipping=None,
            ).to(dtype)

            nn.init.normal_(rnn.state_weight, std=0.1)

        rnn_torch = rnn
        rnn_kernel = rnn

        if is_compiling:
            rnn_kernel = torch.compile(rnn_kernel, fullgraph=True)

        y_kernel, output_state_kernel = rnn_kernel(
            input=x_kernel,
            input_state=input_state_kernel,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=KernelBackend.triton,
        )

        y_torch, output_state_torch = rnn_torch(
            input=x_torch,
            input_state=input_state_torch,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=KernelBackend.torch,
        )

        self.assert_equal_tensors(y_kernel, y_torch, False)
        self.assert_equal_tensors(output_state_kernel, output_state_torch, False)

        y_kernel.sum().backward()
        weight_kernel_grads = self.collect_gradients_from_module_and_zero_grads(rnn)

        y_torch.sum().backward()
        weight_torch_grads = self.collect_gradients_from_module_and_zero_grads(rnn)

        self.assert_equal_tensors(
            x_kernel.grad,
            x_torch.grad,
            False,
            atol_float32=None if cu_seqlens is None else 3.8e-4,
            rtol_float32=None if cu_seqlens is None else 0,
            atol_float16=1e-3,
            rtol_float16=0,
        )

        if has_input_state:
            self.assert_equal_tensors(
                input_state_kernel.grad,
                input_state_torch.grad,
                False,
                atol_float32=None if cu_seqlens is None else 8.2e-4,
                rtol_float32=None if cu_seqlens is None else 0,
                atol_float16=3.7e-4 if cu_seqlens is None else 4.9e-4,
                rtol_float16=0,
            )

        for weight_name in weight_kernel_grads:
            self.assert_equal_tensors(
                weight_kernel_grads[weight_name],
                weight_torch_grads[weight_name],
                False,
                atol_float32=None if cu_seqlens is None else 4.9e-4,
                rtol_float32=None if cu_seqlens is None else 0,
                atol_float16=1.3e-2,
                rtol_float16=0,
            )

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [KernelBackend.torch],  # KernelBackend
            TestCommons.get_dtypes(),  # dtype
            [[0, 7, 19, 27, 93]],  # cu_seqlens
            [(8, 4, 8), (8, 8, 4), (9, 7, 7)],  # state_head_dim, num_input_heads, num_weight_heads
            [False, True],  # has_input_state
        )
    )
    def test_rnn_varlen_torch(
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
            rnn = RNN(
                input_size=state_size,
                state_head_dim=state_head_dim,
                output_size=state_size,
                num_input_heads=num_input_heads,
                num_weight_heads=num_weight_heads,
                add_bias=False,
                gradient_clipping=None,
            ).to(dtype)

            nn.init.normal_(rnn.state_weight, std=0.1)

        y_kernel, _ = rnn(
            input=x_packed_kernel,
            input_state=input_state_kernel,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=KernelBackend.torch,
        )

        y_torch = []
        for i in range(batch_size):
            y, _ = rnn(
                input=x_packed_torch[cu_seqlens[i] : cu_seqlens[i + 1]].unsqueeze(0),
                input_state=input_state_torch[i].unsqueeze(0) if has_input_state else None,
                kernel_backend=KernelBackend.torch,
            )
            y_torch.append(y.squeeze(0))
        y_torch = torch.cat(y_torch)

        self.assert_equal_tensors(y_kernel, y_torch, False)

        y_kernel.sum().backward()
        weight_kernel_grads = self.collect_gradients_from_module_and_zero_grads(rnn)

        y_torch.sum().backward()
        weight_torch_grads = self.collect_gradients_from_module_and_zero_grads(rnn)

        self.assert_equal_tensors(x_packed_kernel.grad, x_packed_torch.grad, False, atol_float32=2e-5, rtol_float32=0)

        for weight_name in weight_kernel_grads:
            self.assert_equal_tensors(
                weight_kernel_grads[weight_name],
                weight_torch_grads[weight_name],
                False,
                atol_float32=3e-7,
                rtol_float32=0,
                atol_float16=5e-4,
                rtol_float16=0,
                atol_bfloat16=5e-3,
                rtol_bfloat16=0,
            )

    def _get_packed_tensor_inputs(
        self,
        batch_size: int,
        sequence_length: int | None,
        total_tokens: int | None,
        state_size: int,
        has_input_state: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor | None]:
        x_kernel, x_torch = self.get_random_duplicated_tensors(
            ((batch_size, sequence_length, state_size) if total_tokens is None else (total_tokens, state_size)),
            device=device,
            dtype=dtype,
            std=0.01,
        )

        input_state_kernel = None
        input_state_torch = None
        if has_input_state:
            input_state_kernel, input_state_torch = self.get_random_duplicated_tensors(
                (batch_size, state_size), device=device, dtype=dtype, std=0.01
            )

        return x_kernel, x_torch, input_state_kernel, input_state_torch
