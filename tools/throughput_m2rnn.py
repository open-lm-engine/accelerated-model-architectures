# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from time import perf_counter

import torch
import torch.nn as nn
from tabulate import tabulate

from xma import Accelerator, KernelBackend, m2rnn


n = 100


def _get_torch_device() -> torch.device:
    accelerator = Accelerator.get_accelerator()
    device = Accelerator.get_current_device()

    if accelerator in [Accelerator.cuda, Accelerator.rocm]:
        return torch.device("cuda", device)

    return torch.device(device)


def _get_dtypes() -> list[torch.dtype]:
    accelerator = Accelerator.get_accelerator()
    if accelerator in [Accelerator.cuda, Accelerator.rocm]:
        return [torch.float16, torch.bfloat16, torch.float32]

    return [torch.float32]


def _tensor_bytes(*tensors: torch.Tensor | None) -> int:
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def _benchmark_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    forget_input: torch.Tensor,
    input_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
    kernel_backend: KernelBackend,
) -> tuple[float, float]:
    with torch.inference_mode():
        y, output_state = m2rnn(
            query=query,
            key=key,
            value=value,
            weight=weight,
            forget_input=forget_input,
            input_state=input_state,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=kernel_backend,
        )

        for _ in range(10):
            m2rnn(
                query=query,
                key=key,
                value=value,
                weight=weight,
                forget_input=forget_input,
                input_state=input_state,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                kernel_backend=kernel_backend,
            )

    Accelerator.synchronize()
    start = perf_counter()
    with torch.inference_mode():
        for _ in range(n):
            m2rnn(
                query=query,
                key=key,
                value=value,
                weight=weight,
                forget_input=forget_input,
                input_state=input_state,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                kernel_backend=kernel_backend,
            )
    Accelerator.synchronize()
    end = perf_counter()

    bytes_per_iter = _tensor_bytes(query, key, value, weight, forget_input, input_state, y, output_state)
    elapsed_s = end - start
    return (elapsed_s * 1e3 / n), (bytes_per_iter * n / elapsed_s / 1e9)


def _benchmark_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    forget_input: torch.Tensor,
    input_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
    kernel_backend: KernelBackend,
) -> tuple[float, float]:
    y, output_state = m2rnn(
        query=query,
        key=key,
        value=value,
        weight=weight,
        forget_input=forget_input,
        input_state=input_state,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        kernel_backend=kernel_backend,
    )

    loss = y.sum() + output_state.sum()

    grad_targets = [query, key, value, weight, forget_input]
    if input_state is not None:
        grad_targets.insert(1, input_state)

    for _ in range(10):
        torch.autograd.grad(loss, grad_targets, retain_graph=True)

    Accelerator.synchronize()
    start = perf_counter()
    for _ in range(n):
        torch.autograd.grad(loss, grad_targets, retain_graph=True)
    Accelerator.synchronize()
    end = perf_counter()

    forward_bytes = _tensor_bytes(query, key, value, weight, forget_input, input_state, y, output_state)
    backward_bytes = forward_bytes + _tensor_bytes(query, key, value, weight, forget_input, input_state)
    elapsed_s = end - start
    return (elapsed_s * 1e3 / n), (backward_bytes * n / elapsed_s / 1e9)


headers = [
    "dtype",
    "torch fwd (ms)",
    "torch fwd (GB/s)",
    "torch bwd (ms)",
    "torch bwd (GB/s)",
    "triton fwd (ms)",
    "triton fwd (GB/s)",
    "triton bwd (ms)",
    "triton bwd (GB/s)",
]
table = []

B = 16
S = 1024
key_head_dim = 64
value_head_dim = 64
num_query_heads = 8
num_key_heads = 8
num_value_heads = 8
num_forget_input_heads = 8
num_weight_heads = 8

device = _get_torch_device()

for dtype in _get_dtypes():
    row = [str(dtype)]

    query = torch.randn(B, S, num_query_heads, key_head_dim, device=device, dtype=dtype, requires_grad=False)
    key = torch.randn(B, S, num_key_heads, key_head_dim, device=device, dtype=dtype, requires_grad=False)
    value = torch.randn(B, S, num_value_heads, value_head_dim, device=device, dtype=dtype, requires_grad=False)
    weight = torch.randn(
        num_weight_heads, value_head_dim, value_head_dim, device=device, dtype=dtype, requires_grad=False
    )
    forget_input = torch.randn(B, S, num_forget_input_heads, device=device, dtype=dtype, requires_grad=False)

    with torch.no_grad():
        nn.init.normal_(weight, std=0.1)

    for kernel_backend in [KernelBackend.torch, KernelBackend.triton]:
        if kernel_backend == KernelBackend.triton and not kernel_backend.verify_accelerator():
            row.extend(["NA", "NA", "NA", "NA"])
            continue

        forward_ms, forward_gbps = _benchmark_forward(
            query=query,
            key=key,
            value=value,
            weight=weight,
            forget_input=forget_input,
            input_state=None,
            cu_seqlens=None,
            max_seqlen=None,
            kernel_backend=kernel_backend,
        )
        # backward_ms, backward_gbps = _benchmark_backward(
        #     query=query,
        #     key=key,
        #     value=value,
        #     weight=weight,
        #     forget_input=forget_input,
        #     input_state=None,
        #     cu_seqlens=None,
        #     max_seqlen=None,
        #     kernel_backend=kernel_backend,
        # )

        row.extend([forward_ms, forward_gbps, 0, 0])

    table.append(row)


print(tabulate(table, headers=headers))
