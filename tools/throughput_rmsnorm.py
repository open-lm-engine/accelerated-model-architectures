# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from tabulate import tabulate

from xma import Accelerator, KernelBackend, rmsnorm


torch._functorch.config.donated_buffer = False


n = 100

kernels = [
    (rmsnorm, KernelBackend.triton, "triton"),
    (rmsnorm, KernelBackend.torch, "torch"),
    (torch.compile(rmsnorm), KernelBackend.torch, "torch compile static"),
    (torch.compile(rmsnorm, dynamic=True), KernelBackend.torch, "torch compile dynamic"),
]
dtypes = [torch.float32, torch.bfloat16, torch.float16]
headers = ["kernel"] + dtypes
table = []

B = 16 * 4096
H = 4096

run_forward = True

for kernel, kernel_backend, row_header in kernels:
    row = [row_header]

    if not kernel_backend.verify_accelerator():
        for _ in range(len(dtypes)):
            row.append("NA")
        table.append(row)
        continue

    device = kernel_backend.get_compatible_accelerator().get_current_device()

    for dtype in dtypes:
        x = torch.randn(B, H, device=device, dtype=dtype, requires_grad=not run_forward)
        W = torch.randn(H, device=device, dtype=dtype, requires_grad=not run_forward)

        if not run_forward:
            dy = torch.randn(B, H, device=device, dtype=dtype, requires_grad=not run_forward)
            z = kernel(x=x, weight=W, eps=None, kernel_backend=kernel_backend)

        for i in range(n):
            if run_forward:
                z = kernel(x=x, weight=W, eps=None)
            else:
                torch.autograd.grad(z, (x, W), grad_outputs=dy, retain_graph=True)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        s.record()
        for i in range(n):
            if run_forward:
                z = kernel(x=x, weight=W, eps=None, kernel_backend=kernel_backend)
            else:
                torch.autograd.grad(z, (x, W), grad_outputs=dy, retain_graph=True)
        e.record()

        Accelerator.synchronize()

        t = s.elapsed_time(e) / n / 1e3

        elements = (B + 1) * H
        if not run_forward:
            elements += B * H

        row.append(elements * dtype.itemsize / t / 1e12)

    table.append(row)


print(tabulate(table, headers=headers))
