# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import pytest
import torch
import torch.nn as nn

from xma import GRU, Accelerator, KernelBackend, set_seed

from ..utils import (
    assert_equal_tensors,
    collect_gradients_from_module_and_zero_grads,
    skip_if_incompatible_kernel_backend,
)
from .rnn_test import _get_packed_tensor_inputs


_SEED = 42


def _get_problem_shapes() -> list[tuple[int, int, int, int, int, int, int]]:
    result = [(9, 7, 7, 7, 7, 7, 7)]

    base = [8, 8, 8, 8, 8, 8, 8]
    for i in range(1, len(base)):
        t = base.copy()
        t[i] = 4
        result.append(tuple(t))

    return result


@pytest.mark.parametrize("kernel_backend", [KernelBackend.triton])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("input_shape", [(4, 1024, None), (None, None, [0, 7, 19, 27, 93])])
@pytest.mark.parametrize("problem_shape", _get_problem_shapes())
@pytest.mark.parametrize("has_input_state", [False, True])
@pytest.mark.parametrize("is_compiling", [False, True])
@torch._dynamo.config.patch(recompile_limit=1024)
def test_gru(
    kernel_backend: KernelBackend,
    dtype: torch.dtype,
    input_shape: tuple[int, int, list[int]],
    problem_shape: tuple[int, int, int, int, int, int, int],
    has_input_state: bool,
    is_compiling: bool,
) -> None:
    skip_if_incompatible_kernel_backend(kernel_backend)
    device = kernel_backend.get_compatible_accelerator().get_current_device()

    set_seed(_SEED)

    (
        state_head_dim,
        num_input_heads,
        num_forget_input_heads,
        num_reset_input_heads,
        num_weight_heads,
        num_forget_weight_heads,
        num_reset_weight_heads,
    ) = problem_shape

    num_heads = max(*problem_shape[1:])
    state_size = num_heads * state_head_dim

    B, S, cu_seqlens = input_shape
    max_seqlen = None

    if B is None:
        cu_seqlens = torch.tensor(cu_seqlens, device=device)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        B = cu_seqlens.size(0) - 1

    x_kernel, x_torch, input_state_kernel, input_state_torch = _get_packed_tensor_inputs(
        batch_size=B,
        sequence_length=S if cu_seqlens is None else None,
        total_tokens=None if cu_seqlens is None else cu_seqlens[-1],
        state_size=state_size,
        has_input_state=has_input_state,
        dtype=dtype,
        device=device,
    )

    with torch.device(device):
        gru = GRU(
            input_size=state_size,
            state_head_dim=state_head_dim,
            output_size=state_size,
            num_input_heads=num_input_heads,
            num_forget_input_heads=num_forget_input_heads,
            num_reset_input_heads=num_reset_input_heads,
            num_weight_heads=num_weight_heads,
            num_forget_weight_heads=num_forget_weight_heads,
            num_reset_weight_heads=num_reset_weight_heads,
            add_bias=False,
            gradient_clipping=None,
        ).to(dtype)

        nn.init.normal_(gru.state_weight, std=0.1)

    gru_torch = gru
    gru_kernel = gru

    if is_compiling:
        gru_kernel = torch.compile(gru_kernel, fullgraph=True)

    y_kernel, output_state_kernel = gru_kernel(
        input=x_kernel,
        input_state=input_state_kernel,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        kernel_backend=kernel_backend,
    )

    y_torch, output_state_torch = gru_torch(
        input=x_torch,
        input_state=input_state_torch,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        kernel_backend=KernelBackend.torch,
    )

    assert_equal_tensors(y_kernel, y_torch, False)
    assert_equal_tensors(output_state_kernel, output_state_torch, False)

    y_kernel.sum().backward()
    weight_kernel_grads = collect_gradients_from_module_and_zero_grads(gru)

    y_torch.sum().backward()
    weight_torch_grads = collect_gradients_from_module_and_zero_grads(gru)

    assert_equal_tensors(x_kernel.grad, x_torch.grad, False, atol_float16=1e-3, rtol_float16=0)

    if has_input_state:
        assert_equal_tensors(input_state_kernel.grad, input_state_torch.grad, False, atol_float16=2e-3, rtol_float16=0)

    for weight_name in weight_kernel_grads:
        assert_equal_tensors(
            weight_kernel_grads[weight_name],
            weight_torch_grads[weight_name],
            False,
            atol_float16=5.2e-3,
            rtol_float16=0,
        )


# @pytest.mark.parametrize("kernel_backend", [KernelBackend.torch])
# @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("cu_seqlens", [[0, 7, 19, 27, 93]])
# @pytest.mark.parametrize("problem_shape", _get_problem_shapes())
# @pytest.mark.parametrize("has_input_state", [False, True])
# @torch._dynamo.config.patch(recompile_limit=1024)
# def test_gru_varlen_torch(
#     kernel_backend: KernelBackend,
#     dtype: torch.dtype,
#     cu_seqlens: list[int],
#     problem_shape: tuple[int, int, int, int, int, int, int],
#     has_input_state: bool,
# ) -> None:
#     if Accelerator.get_accelerator() != Accelerator.cuda:
#         pytest.skip("Sufficient to run on CUDA device")

#     skip_if_incompatible_kernel_backend(kernel_backend)
#     device = kernel_backend.get_compatible_accelerator().get_current_device()

#     set_seed(_SEED)

#     batch_size = len(cu_seqlens) - 1
#     cu_seqlens = torch.tensor(cu_seqlens, device=device)
#     max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

#     (
#         state_head_dim,
#         num_input_heads,
#         num_forget_input_heads,
#         num_reset_input_heads,
#         num_weight_heads,
#         num_forget_weight_heads,
#         num_reset_weight_heads,
#     ) = problem_shape

#     num_heads = max(*problem_shape[1:])
#     state_size = num_heads * state_head_dim

#     x_packed_kernel, x_packed_torch, input_state_kernel, input_state_torch = _get_packed_tensor_inputs(
#         batch_size=batch_size,
#         sequence_length=None,
#         total_tokens=cu_seqlens[-1],
#         state_size=state_size,
#         has_input_state=has_input_state,
#         dtype=dtype,
#         device=device,
#     )

#     with torch.device(device):
#         gru = GRU(
#             input_size=state_size,
#             state_head_dim=state_head_dim,
#             output_size=state_size,
#             num_input_heads=num_input_heads,
#             num_forget_input_heads=num_forget_input_heads,
#             num_reset_input_heads=num_reset_input_heads,
#             num_weight_heads=num_weight_heads,
#             num_forget_weight_heads=num_forget_weight_heads,
#             num_reset_weight_heads=num_reset_weight_heads,
#             add_bias=False,
#             gradient_clipping=None,
#         ).to(dtype)

#         nn.init.normal_(gru.state_weight, std=0.01)

#     y_kernel, _ = gru(
#         input=x_packed_kernel,
#         input_state=input_state_kernel,
#         cu_seqlens=cu_seqlens,
#         max_seqlen=max_seqlen,
#         kernel_backend=KernelBackend.torch,
#     )

#     y_torch = []
#     for i in range(batch_size):
#         y, _ = gru(
#             input=x_packed_torch[cu_seqlens[i] : cu_seqlens[i + 1]].unsqueeze(0),
#             input_state=input_state_torch[i].unsqueeze(0) if has_input_state else None,
#             kernel_backend=KernelBackend.torch,
#         )
#         y_torch.append(y.squeeze(0))
#     y_torch = torch.cat(y_torch)

#     assert_equal_tensors(y_kernel, y_torch, False, atol_bfloat16=3.1e-5, rtol_bfloat16=0)

#     y_kernel.sum().backward()
#     weight_kernel_grads = collect_gradients_from_module_and_zero_grads(gru)

#     y_torch.sum().backward()
#     weight_torch_grads = collect_gradients_from_module_and_zero_grads(gru)

#     for weight_name in weight_kernel_grads:
#         assert_equal_tensors(
#             weight_kernel_grads[weight_name],
#             weight_torch_grads[weight_name],
#             False,
#             atol_float32=2.4e-7,
#             rtol_float32=0,
#             atol_float16=1.5e-3,
#             rtol_float16=0,
#             atol_bfloat16=6e-3,
#             rtol_bfloat16=0,
#         )
