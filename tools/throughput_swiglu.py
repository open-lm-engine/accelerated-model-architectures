# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
from tabulate import tabulate

from xma import Accelerator, KernelBackend, swiglu
from xma.cute_dsl_utils import enable_cute_ptx_dump, get_ptx_from_cute_op
from xma.functional.swiglu.cuda_implementation.forward import _swiglu_forward_cuda


torch._inductor.config.max_autotune_gemm_backends = "TRITON"
torch.backends.cuda.matmul.allow_tf32 = True

n = 100

kernels = [
    (swiglu, KernelBackend.cuda, "CUDA"),
    (swiglu, KernelBackend.pallas, "pallas"),
    (swiglu, KernelBackend.nki, "nki"),
    (swiglu, KernelBackend.torch, "torch"),
    (torch.compile(swiglu, dynamic=True), KernelBackend.torch, "torch compile"),
    (swiglu, KernelBackend.triton, "triton"),
    (swiglu, KernelBackend.mps, "mps"),
]
dtypes = [torch.float32, torch.bfloat16, torch.float16]
headers = ["kernel"] + dtypes
table = []

B = 16 * 4096
H = 4096

run_forward = False

for kernel, kernel_backend, row_header in kernels:
    row = [row_header]

    if not kernel_backend.verify_accelerator():
        for _ in range(len(dtypes)):
            row.append("NA")
        table.append(row)
        continue

    device = kernel_backend.get_compatible_accelerator().get_current_device()

    for dtype in dtypes:
        u = torch.randn(B, H, device=device, dtype=dtype, requires_grad=not run_forward)
        g = torch.randn(B, H, device=device, dtype=dtype, requires_grad=not run_forward)

        if not run_forward:
            dy = torch.randn(B, H, device=device, dtype=dtype)
            z = kernel(g, u, kernel_backend=kernel_backend)

        for i in range(n):
            if run_forward:
                z = kernel(g, u, kernel_backend=kernel_backend)
            else:
                torch.autograd.grad(z, (g, u), grad_outputs=dy, retain_graph=True)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        s.record()
        for i in range(n):
            if run_forward:
                z = kernel(g, u, kernel_backend=kernel_backend)
            else:
                torch.autograd.grad(z, (g, u), grad_outputs=dy, retain_graph=True)
        e.record()

        Accelerator.synchronize()

        t = s.elapsed_time(e) / n / 1e3
        row.append((3 if run_forward else 5) * B * H * dtype.itemsize / t / 1e12)

    table.append(row)


print(tabulate(table, headers=headers))


ptx_output_directory = "dump_swiglu_ptx"
enable_cute_ptx_dump(ptx_output_directory)
_swiglu_forward_cuda.cache = {}

device = KernelBackend.cuda.get_compatible_accelerator().get_current_device()
for dtype in dtypes:
    g = torch.randn(B, H, device=device, dtype=dtype)
    u = torch.randn(B, H, device=device, dtype=dtype)
    swiglu(g, u, kernel_backend=KernelBackend.cuda)

get_ptx_from_cute_op(_swiglu_forward_cuda, ptx_output_directory)
print(f"dumped PTX for _swiglu_forward_cuda to {ptx_output_directory}")
