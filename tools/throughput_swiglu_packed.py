# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import torch
from tabulate import tabulate

from xma import Accelerator, KernelBackend, swiglu_packed


torch._inductor.config.max_autotune_gemm_backends = "TRITON"
torch.backends.cuda.matmul.allow_tf32 = True

n = 100

headers = ["dtype", "torch BW", "torch compile BW", "CUDA BW"]
kernels = [
    partial(swiglu_packed, kernel_backend=KernelBackend.torch),
    partial(torch.compile(swiglu_packed, dynamic=True), kernel_backend=KernelBackend.torch),
    partial(swiglu_packed, kernel_backend=KernelBackend.cuda),
]

table = []
B = 16 * 4096
H = 4096

run_forward = False

for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    row = [str(dtype)]
    x = torch.randn(B, 2 * H, device=torch.cuda.current_device(), dtype=dtype, requires_grad=not run_forward)

    if not run_forward:
        dy = torch.randn(B, H, device=torch.cuda.current_device(), dtype=dtype)

    for kernel in kernels:
        if not run_forward:
            z = kernel(x)

        for i in range(n):
            if run_forward:
                z = kernel(x)
            else:
                torch.autograd.grad(z, x, grad_outputs=dy, retain_graph=True)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        s.record()
        for i in range(n):
            if run_forward:
                z = kernel(x)
            else:
                torch.autograd.grad(z, x, grad_outputs=dy, retain_graph=True)
        e.record()

        Accelerator.synchronize()

        t = s.elapsed_time(e) / n / 1e3
        row.append((3 if run_forward else 5) * B * H * dtype.itemsize / t / 1e12)

    table.append(row)


print(tabulate(table, headers=headers))
