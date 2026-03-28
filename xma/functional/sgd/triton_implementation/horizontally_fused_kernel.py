# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl

from ....math import get_powers_of_2
from .kernel import _sgd_step


_TORCH_TO_TRITON_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for num_warps in get_powers_of_2(4, 32):
        for BLOCK_SIZE in get_powers_of_2(64, 16384):
            configs.append(triton.Config({"BLOCK_SIZE": BLOCK_SIZE}, num_warps=num_warps))

    return configs


@triton.autotune(configs=_get_autotune_configs(), key=[])
@triton.jit
def sgd_horizontally_fused_kernel(
    W_ptr_ptr, dW_ptr_ptr, N_ptr, lr, BLOCK_SIZE: tl.constexpr, MAXIMIZE: tl.constexpr, DTYPE: tl.constexpr
):
    i = tl.program_id(0)

    W_ptr = tl.load(W_ptr_ptr + i).to(tl.pointer_type(DTYPE))
    dW_ptr = tl.load(dW_ptr_ptr + i).to(tl.pointer_type(DTYPE))
    N = tl.load(N_ptr + i)

    for START in range(0, N, BLOCK_SIZE):
        BLOCK = START + tl.arange(0, BLOCK_SIZE)
        MASK = BLOCK < N

        W = tl.load(W_ptr + BLOCK, mask=MASK)
        dW = tl.load(dW_ptr + BLOCK, mask=MASK)

        W = _sgd_step(W=W, dW=dW, lr=lr, MAXIMIZE=MAXIMIZE)
        tl.store(W_ptr + BLOCK, W, mask=MASK)


def sgd_horizontally_fused_triton(Ws: list[torch.Tensor], dWs: list[torch.Tensor], lr: float, maximize: bool) -> None:
    device = Ws[0].device

    W_ptr_ptr = torch.tensor([W.data_ptr() for W in Ws], dtype=torch.int64, device=device)
    dW_ptr_ptr = torch.tensor([dW.data_ptr() for dW in dWs], dtype=torch.int64, device=device)
    N_ptr = torch.tensor([W.numel() for W in Ws], dtype=torch.int64, device=device)

    sgd_horizontally_fused_kernel[(len(Ws),)](
        W_ptr_ptr=W_ptr_ptr,
        dW_ptr_ptr=dW_ptr_ptr,
        N_ptr=N_ptr,
        lr=lr,
        MAXIMIZE=maximize,
        DTYPE=_TORCH_TO_TRITON_DTYPE[Ws[0].dtype],
    )
