# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from functools import partial

import torch
from tabulate import tabulate

from xma import Accelerator, KernelBackend, bmm


torch._inductor.config.max_autotune_gemm_backends = "TRITON"
torch.backends.cuda.matmul.allow_tf32 = True

n = 100

headers = ["dtype", "torch TFLOPs", "torch compile TFLOPs", "triton TFLOPs"]
kernels = [
    partial(bmm, kernel_backend=KernelBackend.torch),
    partial(torch.compile(bmm, mode="max-autotune"), kernel_backend=KernelBackend.torch),
    partial(bmm, kernel_backend=KernelBackend.triton),
]

table = []
L = 1
M = 4096
N = 4096
K = 4096

for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    row = [str(dtype)]
    for kernel in kernels:
        x = torch.randn(L, M, K, device=torch.cuda.current_device(), dtype=dtype)
        w = torch.randn(L, K, N, device=torch.cuda.current_device(), dtype=dtype)

        for i in range(n):
            z = kernel(x, w, C=None, beta=0)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        s.record()
        for i in range(n):
            z = kernel(x, w, C=None, beta=0)
        e.record()

        Accelerator.synchronize()

        t = s.elapsed_time(e) / n / 1e3
        row.append(2 * L * M * N * K / t / 1e12)

    table.append(row)


print(tabulate(table, headers=headers))
