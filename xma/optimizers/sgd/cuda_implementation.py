# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from functools import partial

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Float32, const_expr

from ...autotuner import AutotuneConfig, autotune
from ...cute_dsl_utils import MultiTensorApplyCUDAKernel, multi_tensor_apply


class _SGDCUDAKernel(MultiTensorApplyCUDAKernel):
    def __init__(
        self,
        BLOCK_SIZE: int,
        M: int,
        num_tensors: int,
        has_momentum: bool,
        is_first_step: bool,
        nesterov: bool,
        maximize: bool,
    ) -> _SGDCUDAKernel:
        depth_x = 3 if (has_momentum and not is_first_step) else 2
        depth_y = 2 if has_momentum else 1

        super().__init__(BLOCK_SIZE=BLOCK_SIZE, M=M, depth_x=depth_x, depth_y=depth_y, num_tensors=num_tensors)

        self.has_momentum = has_momentum
        self.is_first_step = is_first_step
        self.nesterov = nesterov
        self.maximize = maximize

    @cute.jit
    def compute(self, xs: list[cute.TensorSSA], scalars: list[Float32]) -> list[cute.TensorSSA | None]:
        lr = scalars[0]
        weight_decay = scalars[1]
        momentum = scalars[2]
        dampening = scalars[3]

        if const_expr(self.has_momentum and not self.is_first_step):
            W, dW, Mbuf = xs
        else:
            W, dW = xs

        dtype = W.dtype
        W = W.to(Float32)
        dW = dW.to(Float32)

        if const_expr(self.maximize):
            dW = -dW

        dW = dW + weight_decay * W

        if const_expr(self.has_momentum):
            if const_expr(self.is_first_step):
                M = dW
            else:
                M = Mbuf.to(Float32) * momentum + dW * (1.0 - dampening)

            dW = dW + M * momentum if const_expr(self.nesterov) else M

            W = W - lr * dW

            return [W.to(dtype), M.to(dtype)]
        else:
            W = W - lr * dW

            return [W.to(dtype)]


def _get_autotune_configs() -> list[AutotuneConfig]:
    # single config on purpose: SGD's update isn't idempotent, so benchmarking multiple configs would apply it repeatedly
    return [AutotuneConfig({"BLOCK_SIZE": 256, "M": 4})]


@autotune(configs=_get_autotune_configs())
def _sgd_multi_tensor_cuda(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    momentum_buffer_list: list[torch.Tensor],
    lr: float,
    weight_decay: float,
    momentum: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    is_first_step: bool,
    BLOCK_SIZE: int,
    M: int,
) -> None:
    has_momentum = momentum != 0

    scalars = [lr, weight_decay, momentum, dampening]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if not has_momentum or is_first_step:
        x_tensor_lists = [params, grads]
    else:
        x_tensor_lists = [params, grads, momentum_buffer_list]

    y_tensor_lists = [params, momentum_buffer_list] if has_momentum else [params]

    multi_tensor_apply(
        caller_op=_sgd_multi_tensor_cuda,
        key=(params[0].dtype, len(params), has_momentum, is_first_step, nesterov, maximize, BLOCK_SIZE, M),
        kernel_class=partial(
            _SGDCUDAKernel,
            BLOCK_SIZE=BLOCK_SIZE,
            M=M,
            has_momentum=has_momentum,
            is_first_step=is_first_step,
            nesterov=nesterov,
            maximize=maximize,
        ),
        x_tensor_lists=x_tensor_lists,
        y_tensor_lists=y_tensor_lists,
        divisibility_list=[1] * len(params),
        scalars=scalars,
        stream=stream,
    )
