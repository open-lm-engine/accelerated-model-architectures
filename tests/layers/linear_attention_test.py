# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from itertools import product

import pytest

from xma import KernelBackend, is_torch_available

from ..utils import assert_equal_tensors, skip_if_incompatible_kernel_backend
from .rnn_test import _get_packed_tensor_inputs


_SEED = 42


def _get_problem_shapes() -> list[tuple[int, int, int, int, int]]:
    result = [(7, 9, 3, 3, 3)]

    base = [8, 4, 3, 3, 3]
    for i in range(2, len(base)):
        t = base.copy()
        t[i] = 6
        result.append(tuple(t))

    return result


if is_torch_available():
    import torch

    from xma import LinearAttention, set_seed

    def _generate_args() -> list:
        args = list(
            product(
                [KernelBackend.triton],  # kernel_backend
                [torch.float32, torch.bfloat16],
                [4],  # batch_size
                [977],  # sequence_length
                [(7, 9, 3, 3, 3)],  # problem_shape
                [False, True],  # has_input_state
                [False],  # is_compiling
            )
        )

        args += list(
            product(
                [KernelBackend.triton],  # kernel_backend
                [torch.float32, torch.bfloat16],
                [4],  # batch_size
                [1024],  # sequence_length
                _get_problem_shapes(),  # problem_shape
                [False, True],  # has_input_state
                [False],  # is_compiling
            )
        )

        return args

    @pytest.mark.parametrize(
        "kernel_backend,dtype,batch_size,sequence_length,problem_shape,has_input_state,is_compiling", _generate_args()
    )
    @torch._dynamo.config.patch(recompile_limit=1024)
    def test_linear_attention(
        kernel_backend: KernelBackend,
        dtype: torch.dtype,
        batch_size: int,
        sequence_length: int,
        problem_shape: tuple[int, int, int, int, int, int, int],
        has_input_state: bool,
        is_compiling: bool,
    ) -> None:
        skip_if_incompatible_kernel_backend(kernel_backend)
        device = kernel_backend.get_compatible_accelerator().get_current_device()

        set_seed(_SEED)

        key_head_dim, value_head_dim, num_query_heads, num_key_heads, num_value_heads = problem_shape
        num_heads = max(num_query_heads, num_key_heads, num_value_heads)
        state_size = num_heads * key_head_dim * value_head_dim

        x_kernel, x_torch, input_state_kernel, input_state_torch = _get_packed_tensor_inputs(
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

        assert_equal_tensors(y_kernel, y_torch, False)
        assert_equal_tensors(output_state_kernel, output_state_torch, False)
