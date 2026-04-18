# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
from torch.distributed.tensor import DTensor
from torch.optim.sgd import _multi_tensor_sgd, _single_tensor_sgd

from ...accelerator import Accelerator, KernelBackend
from ...constants import LOG_WARP_SIZE
from ...utils import is_triton_available


if is_triton_available():
    import triton.language as tl

    from .triton_implementation import _multi_tensor_sgd_triton_kernel, _single_tensor_sgd_triton

    _TORCH_TO_TRITON_DTYPE = {torch.float32: tl.float32, torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}


@torch.no_grad()
def sgd(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    momentum_buffer_list: list[torch.Tensor],
    lr: float,
    maximize: bool,
    horizontal_fusion: bool,
    weight_decay: float,
    momentum: float,
    dampening: float,
    nesterov: bool,
    *,
    kernel_backend: KernelBackend | None = None,
) -> None:
    if kernel_backend is None:
        kernel_backend = Accelerator.get_kernel_backend()
    else:
        assert kernel_backend.verify_accelerator()

    if kernel_backend in [KernelBackend.cuda, KernelBackend.triton]:
        if isinstance(W, DTensor):
            assert isinstance(dW, DTensor)
            assert W.placements == dW.placements
            W = W._local_tensor
            dW = dW._local_tensor

        is_first_step = False
        if momentum == 0:
            assert len(momentum_buffer_list) == 0
            momentum_buffer_list = None
        elif momentum_buffer_list[0] is None:
            assert all([m is None for m in momentum_buffer_list])
            is_first_step = True

            for i, p in enumerate(params):
                momentum_buffer_list[i] = torch.empty_like(p, dtype=torch.float32)

        if horizontal_fusion:
            device = params[0].device
            NUM_WARPS = 8

            _multi_tensor_sgd_triton_kernel[len(params),](
                W_ptr_ptr=torch.tensor([W.data_ptr() for W in params], dtype=torch.int64, device=device),
                W_dtype=_TORCH_TO_TRITON_DTYPE[params[0].dtype],
                dW_ptr_ptr=torch.tensor([dW.data_ptr() for dW in grads], dtype=torch.int64, device=device),
                dW_dtype=_TORCH_TO_TRITON_DTYPE[grads[0].dtype],
                M_ptr_ptr=(
                    None
                    if momentum == 0
                    else torch.tensor([M.data_ptr() for M in momentum_buffer_list], dtype=torch.int64, device=device)
                ),
                M_dtype=None if momentum == 0 else _TORCH_TO_TRITON_DTYPE[momentum_buffer_list[0].dtype],
                N_ptr=torch.tensor([W.numel() for W in params], dtype=torch.int64, device=device),
                lr=lr,
                weight_decay=None if weight_decay == 0 else weight_decay,
                momentum=None if momentum == 0 else momentum,
                dampening=None if dampening == 0 else dampening,
                NESTEROV=nesterov,
                MAXIMIZE=maximize,
                IS_FIRST_STEP=is_first_step,
                BLOCK_SIZE=(NUM_WARPS << LOG_WARP_SIZE) * (16 // params[0].dtype.itemsize),
                num_warps=NUM_WARPS,
            )
        else:
            if momentum_buffer_list is None:
                momentum_buffer_list = [None] * len(params)

            for W, dW, M in zip(params, grads, momentum_buffer_list):
                assert W.is_contiguous()
                dW = dW.contiguous()

                if M is not None:
                    assert M.is_contiguous()

                _single_tensor_sgd_triton(
                    W=W,
                    dW=dW,
                    M=M,
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    dampening=dampening,
                    nesterov=nesterov,
                    maximize=maximize,
                    is_first_step=is_first_step,
                )
    elif kernel_backend == KernelBackend.torch:
        (_multi_tensor_sgd if horizontal_fusion else _single_tensor_sgd)(
            params=params,
            grads=grads,
            momentum_buffer_list=momentum_buffer_list,
            grad_scale=None,
            found_inf=None,
            weight_decay=weight_decay,
            momentum=momentum,
            lr=lr,
            dampening=dampening,
            nesterov=nesterov,
            maximize=maximize,
            has_sparse_grad=False,
        )
    else:
        raise ValueError(f"unexpected kernel_backend ({kernel_backend})")
