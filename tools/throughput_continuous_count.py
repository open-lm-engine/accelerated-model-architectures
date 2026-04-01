# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from tabulate import tabulate

from xma import Accelerator, KernelBackend, continuous_count


torch._functorch.config.donated_buffer = False


n = 100

kernels = [
    (continuous_count, KernelBackend.cuda, "cuda"),
    (continuous_count, KernelBackend.torch, "torch"),
]
dtypes = [torch.int32, torch.int64]
headers = ["kernel"] + dtypes
table = []

B = 16 * 4096
C = 64

for kernel, kernel_backend, row_header in kernels:
    row = [row_header]

    if not kernel_backend.verify_accelerator():
        for _ in range(len(dtypes)):
            row.append("NA")
        table.append(row)
        continue

    device = kernel_backend.get_compatible_accelerator().get_current_device()

    for dtype in dtypes:
        x = torch.randint(0, C, (B,), device=device, dtype=dtype)
        y = kernel(x=x, bins=C, kernel_backend=kernel_backend)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        s.record()
        for i in range(n):
            y = kernel(x=x, bins=C, kernel_backend=kernel_backend)
        e.record()

        Accelerator.synchronize()

        t = s.elapsed_time(e) / n / 1e3

        io = B * dtype.itemsize + C * torch.uint32.itemsize
        row.append(io / t / 1e12)

    table.append(row)


print(tabulate(table, headers=headers))
