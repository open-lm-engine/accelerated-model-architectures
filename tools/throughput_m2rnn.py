# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from time import perf_counter

import torch
import torch.nn as nn
from tabulate import tabulate

from xma import M2RNN, Accelerator, KernelBackend


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
    model: M2RNN,
    x: torch.Tensor,
    input_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
    kernel_backend: KernelBackend,
) -> tuple[float, float]:
    with torch.inference_mode():
        y, output_state = model(
            input=x,
            input_state=input_state,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=kernel_backend,
        )

        for _ in range(10):
            model(
                input=x,
                input_state=input_state,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                kernel_backend=kernel_backend,
            )

    Accelerator.synchronize()
    start = perf_counter()
    with torch.inference_mode():
        for _ in range(n):
            model(
                input=x,
                input_state=input_state,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                kernel_backend=kernel_backend,
            )
    Accelerator.synchronize()
    end = perf_counter()

    bytes_per_iter = _tensor_bytes(x, input_state, y, output_state)
    elapsed_s = end - start
    return (elapsed_s * 1e3 / n), (bytes_per_iter * n / elapsed_s / 1e9)


def _make_inputs(
    *,
    batch_size: int,
    sequence_length: int,
    state_size: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch_size, sequence_length, state_size, device=device, dtype=dtype, requires_grad=requires_grad)
    input_state = torch.randn(batch_size, state_size, device=device, dtype=dtype, requires_grad=requires_grad)
    return x, input_state


def _benchmark_backward(
    model: M2RNN,
    x: torch.Tensor,
    input_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
    kernel_backend: KernelBackend,
) -> tuple[float, float]:
    y, output_state = model(
        input=x,
        input_state=input_state,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        kernel_backend=kernel_backend,
    )

    loss = y.sum() + output_state.sum()

    grad_targets = [x, *model.parameters()]
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

    forward_bytes = _tensor_bytes(x, input_state, y, output_state, *model.parameters())
    backward_bytes = forward_bytes + _tensor_bytes(x, input_state, *model.parameters())
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

num_heads = max(num_query_heads, num_key_heads, num_value_heads, num_forget_input_heads, num_weight_heads)
state_size = num_heads * value_head_dim

device = _get_torch_device()

for dtype in _get_dtypes():
    row = [str(dtype)]

    x_fwd, input_state_fwd = _make_inputs(
        batch_size=B,
        sequence_length=S,
        state_size=state_size,
        dtype=dtype,
        device=device,
        requires_grad=False,
    )
    x_bwd, input_state_bwd = _make_inputs(
        batch_size=B,
        sequence_length=S,
        state_size=state_size,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )

    with torch.no_grad():
        m2rnn = M2RNN(
            input_size=state_size,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            output_size=state_size,
            num_query_heads=num_query_heads,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            num_forget_input_heads=num_forget_input_heads,
            num_weight_heads=num_weight_heads,
            add_bias=False,
            gradient_clipping=None,
        ).to(device=device, dtype=dtype)

        nn.init.normal_(m2rnn.state_weight, std=0.1)

    for kernel_backend in [KernelBackend.torch, KernelBackend.triton]:
        if kernel_backend == KernelBackend.triton and not kernel_backend.verify_accelerator():
            row.extend(["NA", "NA", "NA", "NA"])
            continue

        forward_ms, forward_gbps = _benchmark_forward(
            model=m2rnn,
            x=x_fwd,
            input_state=input_state_fwd,
            cu_seqlens=None,
            max_seqlen=None,
            kernel_backend=kernel_backend,
        )
        backward_ms, backward_gbps = _benchmark_backward(
            model=m2rnn,
            x=x_bwd,
            input_state=input_state_bwd,
            cu_seqlens=None,
            max_seqlen=None,
            kernel_backend=kernel_backend,
        )

        row.extend([forward_ms, forward_gbps, backward_ms, backward_gbps])

    table.append(row)


print(tabulate(table, headers=headers))
