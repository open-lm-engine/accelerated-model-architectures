# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import triton
import triton.language as tl


_TORCH_TO_TRITON_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def sgd_horizontally_fused_kernel(
    W_ptr_list, dW_ptr_list, N_list, lr, BLOCK_SIZE: tl.constexpr, MAXIMIZE: tl.constexpr, DTYPE: tl.constexpr
):
    i = tl.program_id(0)

    W_base = tl.load(W_ptr_list + i).to(tl.pointer_type(DTYPE))
    dW_base = tl.load(dW_ptr_list + i).to(tl.pointer_type(DTYPE))
    N = tl.load(N_list + i)

    for block_start in range(0, N, BLOCK_SIZE):
        BLOCK = block_start + tl.arange(0, BLOCK_SIZE)
        MASK = BLOCK < N

        W = tl.load(W_base + BLOCK, mask=MASK)
        dW = tl.load(dW_base + BLOCK, mask=MASK)

        if MAXIMIZE:
            dW = -dW

        W -= lr * dW
        tl.store(W_base + BLOCK, W, mask=MASK)


def sgd_horizontally_fused_triton(Ws: list[torch.Tensor], dWs: list[torch.Tensor], lr: float, maximize: bool) -> None:
    num_tensors = len(Ws)
    device = Ws[0].device
    dtype = Ws[0].dtype

    W_ptrs = torch.tensor([W.data_ptr() for W in Ws], dtype=torch.int64, device=device)
    dW_ptrs = torch.tensor([dW.data_ptr() for dW in dWs], dtype=torch.int64, device=device)
    Ns = torch.tensor([W.numel() for W in Ws], dtype=torch.int64, device=device)

    sgd_horizontally_fused_kernel[(num_tensors,)](
        W_ptr_list=W_ptrs,
        dW_ptr_list=dW_ptrs,
        N_list=Ns,
        lr=lr,
        BLOCK_SIZE=block_size,
        MAXIMIZE=maximize,
        DTYPE=_TORCH_TO_TRITON_DTYPE[dtype],
    )
