# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F
from tabulate import tabulate

from xma import Accelerator, pack_sequence


n = 100
B = 8
S = 4096
T = 691
x = torch.randn(B, S, 32, 128, device=torch.cuda.current_device(), dtype=torch.float32)
cu_seqlens = torch.tensor([0, 70, 170, 295, 393, 412, 515, T], device=torch.cuda.current_device(), dtype=torch.uint32)
attention_mask = [
    torch.cat([torch.zeros(S - i), torch.ones(i)], dim=-1) for i in cu_seqlens[1:].int() - cu_seqlens[:-1].int()
]
attention_mask = torch.stack(attention_mask, dim=0).to(torch.cuda.current_device()).to(torch.bool)


def _hf_compatible_pack(x, attention_mask: torch.Tensor):
    seqlens: torch.Tensor = attention_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return pack_sequence(x, cu_seqlens, T)


headers = ["dtype", "pack_sequence"]
kernels = [_hf_compatible_pack]


table = []

for dtype in [torch.float32]:
    row = [str(dtype)]
    for kernel in kernels:
        for i in range(n):
            z = kernel(x, attention_mask)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)

        s.record()
        for i in range(n):
            z = kernel(x, attention_mask)
        e.record()

        Accelerator.synchronize()

        row.append(s.elapsed_time(e) / n)
    table.append(row)


print(tabulate(table, headers=headers))
