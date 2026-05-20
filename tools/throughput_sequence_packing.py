# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F
from tabulate import tabulate

from xma import Accelerator, pack_sequence


n = 100
B = 7
S = 4096
T = 691
INNER = (32, 128)

cu_seqlens = torch.tensor([0, 70, 170, 295, 393, 412, 515, T], device=torch.cuda.current_device(), dtype=torch.uint32)
attention_mask = [
    torch.cat([torch.zeros(S - i), torch.ones(i)], dim=-1) for i in cu_seqlens[1:].int() - cu_seqlens[:-1].int()
]
attention_mask = torch.stack(attention_mask, dim=0).to(torch.cuda.current_device()).to(torch.bool)


def _hf_compatible_pack(x, attention_mask: torch.Tensor):
    seqlens: torch.Tensor = attention_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return pack_sequence([x], cu_seqlens=cu_seqlens, total_tokens=T)[0]


headers = ["dtype", "pack_sequence (GB/s)"]
kernels = [_hf_compatible_pack]

table = []

for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    x = torch.randn(B, S, *INNER, device=torch.cuda.current_device(), dtype=dtype)
    # read T tokens + write T tokens
    bytes_moved = 2 * T * x[0, 0].numel() * x.element_size()

    row = [str(dtype)]
    for kernel in kernels:
        for _ in range(n):
            z = kernel(x, attention_mask)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        s.record()
        for _ in range(n):
            z = kernel(x, attention_mask)
        e.record()

        Accelerator.synchronize()

        time_ms = s.elapsed_time(e) / n
        row.append(f"{bytes_moved / (time_ms * 1e-3) / 1e9:.2f}")
    table.append(row)


print(tabulate(table, headers=headers))
