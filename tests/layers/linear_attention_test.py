# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from contextlib import nullcontext

import torch
from parameterized import parameterized

from xma import Accelerator, KernelBackend, LinearAttention, set_seed

from ..test_commons import TestCommons


_SEED = 42


def _get_problem_shapes() -> list[tuple[int, int, int, int, int, int, int]]:
    result = [(7, 9, 3, 3, 3)]

    base = [8, 4, 3, 3, 3]
    for i in range(2, len(base)):
        t = base.copy()
        t[i] = 6
        result.append(tuple(t))

    return result


class LinearAttentionTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [KernelBackend.triton],  # KernelBackend
            [torch.float32, torch.bfloat16],
            [4],  # batch_size
            [1024],  # sequence_length
            _get_problem_shapes(),  # problem_shape
            [False, True],  # has_input_state
            [False],  # is_compiling
            [True],  # no_grad
        )
    )
    def test_linear_attention(
        self,
        kernel_backend: KernelBackend,
        dtype: torch.dtype,
        batch_size: int,
        sequence_length: int,
        problem_shape: tuple[int, int, int, int, int, int, int],
        has_input_state: bool,
        is_compiling: bool,
        no_grad: bool,
    ) -> None:
        self.skip_if_incompatible_kernel_backend(kernel_backend)
        device = kernel_backend.get_compatible_accelerator().get_current_device()

        set_seed(_SEED)

        context = torch.no_grad if no_grad else nullcontext

        key_head_dim, value_head_dim, num_query_heads, num_key_heads, num_value_heads = problem_shape
        num_heads = max(num_query_heads, num_key_heads, num_value_heads)
        state_size = num_heads * key_head_dim * value_head_dim

        with context():
            x_kernel, x_torch, input_state_kernel, input_state_torch = self._get_packed_tensor_inputs(
                batch_size=batch_size,
                sequence_length=sequence_length,
                total_tokens=None,
                state_size=state_size,
                has_input_state=has_input_state,
                dtype=dtype,
                device=device,
            )

            with torch.device(device):
                linear_attention = LinearAttention(
                    input_size=state_size,
                    key_head_dim=key_head_dim,
                    value_head_dim=value_head_dim,
                    output_size=state_size,
                    num_query_heads=num_query_heads,
                    num_key_heads=num_key_heads,
                    num_value_heads=num_value_heads,
                    add_bias=False,
                ).to(dtype)

            linear_attention_torch = linear_attention
            linear_attention_kernel = linear_attention

            if is_compiling:
                linear_attention_kernel = torch.compile(linear_attention_kernel, fullgraph=True)

            y_kernel, output_state_kernel = linear_attention_kernel(
                input=x_kernel, input_state=input_state_kernel, kernel_backend=KernelBackend.triton
            )

            y_torch, output_state_torch = linear_attention_torch(
                input=x_torch, input_state=input_state_torch, kernel_backend=KernelBackend.torch
            )

            # self.assert_equal_tensors(
            #     y_kernel,
            #     y_torch,
            #     False,
            #     atol_float32=4e-6,
            #     rtol_float32=0,
            #     atol_float16=6.5e-5,
            #     rtol_float16=0,
            # )

            self.assert_equal_tensors(
                output_state_kernel,
                output_state_torch,
                False,
                atol_float32=3e-3,
                rtol_float32=0,
                atol_bfloat16=2.6e-3,
                rtol_bfloat16=0,
            )

            # if not no_grad:
            #     y_kernel.sum().backward()
            #     weight_kernel_grads = self.collect_gradients_from_module_and_zero_grads(gru)

            #     y_torch.sum().backward()
            #     weight_torch_grads = self.collect_gradients_from_module_and_zero_grads(gru)

            #     self.assert_equal_tensors(
            #         x_kernel.grad,
            #         x_torch.grad,
            #         False,
            #         atol_float16=1e-3,
            #         rtol_float16=0,
            #     )

            #     if has_input_state:
            #         self.assert_equal_tensors(
            #             input_state_kernel.grad,
            #             input_state_torch.grad,
            #             False,
            #             atol_float32=4e-6,
            #             rtol_float32=0,
            #             atol_float16=2e-3,
            #             rtol_float16=0,
            #         )

            #     for weight_name in weight_kernel_grads:
            #         self.assert_equal_tensors(
            #             weight_kernel_grads[weight_name],
            #             weight_torch_grads[weight_name],
            #             False,
            #             atol_float32=6e-3,
            #             rtol_float32=0,
            #             atol_float16=2.3e-2,
            #             rtol_float16=0,
            #         )

    @parameterized.expand(
        TestCommons.make_args_matrix(
            [KernelBackend.torch],  # KernelBackend
            TestCommons.get_dtypes(),  # dtype
            [[0, 7, 19, 27, 93]],  # cu_seqlens
            [(8, 4, 3, 3, 3)],  # problem_shape
            [False, True],  # has_input_state
        )
    )
    def test_linear_attention_varlen_torch(
        self,
        kernel_backend: KernelBackend,
        dtype: torch.dtype,
        cu_seqlens: list[int],
        problem_shape: tuple[int, int, int, int, int],
        has_input_state: bool,
    ) -> None:
        if Accelerator.get_accelerator() != Accelerator.cuda:
            self.skipTest("Sufficient to run on CUDA device")

        self.skip_if_incompatible_kernel_backend(kernel_backend)
        device = kernel_backend.get_compatible_accelerator().get_current_device()

        set_seed(_SEED)

        batch_size = len(cu_seqlens) - 1
        cu_seqlens = torch.tensor(cu_seqlens, device=device)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()

        key_head_dim, value_head_dim, num_query_heads, num_key_heads, num_value_heads = problem_shape
        num_heads = max(num_query_heads, num_key_heads, num_value_heads)
        state_size = num_heads * key_head_dim * value_head_dim

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
            linear_attention = LinearAttention(
                input_size=state_size,
                key_head_dim=key_head_dim,
                value_head_dim=value_head_dim,
                output_size=state_size,
                num_query_heads=num_query_heads,
                num_key_heads=num_key_heads,
                num_value_heads=num_value_heads,
                add_bias=False,
            ).to(dtype)

        y_kernel, _ = linear_attention(
            input=x_packed_kernel,
            input_state=input_state_kernel,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=KernelBackend.torch,
        )

        y_torch = []
        for i in range(batch_size):
            y, _ = linear_attention(
                input=x_packed_torch[cu_seqlens[i] : cu_seqlens[i + 1]].unsqueeze(0),
                input_state=input_state_torch[i].unsqueeze(0) if has_input_state else None,
                kernel_backend=KernelBackend.torch,
            )
            y_torch.append(y.squeeze(0))
        y_torch = torch.cat(y_torch)

        self.assert_equal_tensors(y_kernel, y_torch, False)

        y_kernel.sum().backward()
        weight_kernel_grads = self.collect_gradients_from_module_and_zero_grads(linear_attention)

        y_torch.sum().backward()
        weight_torch_grads = self.collect_gradients_from_module_and_zero_grads(linear_attention)

        self.assert_equal_tensors(
            x_packed_kernel.grad,
            x_packed_torch.grad,
            False,
            atol_float32=2e-5,
            rtol_float32=0,
            atol_bfloat16=2.5e-4,
            rtol_bfloat16=0,
        )

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
