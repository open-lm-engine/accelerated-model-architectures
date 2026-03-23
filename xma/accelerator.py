# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from enum import Enum
from functools import lru_cache

import torch

from .utils import is_torch_neuronx_available, is_torch_xla_available


if is_torch_xla_available():
    from torch_xla.core.xla_model import wait_device_ops as xla_wait_device_ops
    from torch_xla.core.xla_model import xla_device


_IS_ROCM_AVAILABLE = torch.version.hip is not None


class KernelBackend(Enum):
    cuda = "cuda"
    mps = "mps"
    nki = "nki"
    pallas = "pallas"
    rocm = "rocm"
    torch = "torch"
    triton = "triton"

    def get_compatible_accelerator(self) -> Accelerator:
        found_accelerator = Accelerator.get_accelerator()

        if self == KernelBackend.torch or (
            self == KernelBackend.triton and found_accelerator in [Accelerator.cuda, Accelerator.rocm]
        ):
            return found_accelerator

        mapping = {
            KernelBackend.cuda: Accelerator.cuda,
            KernelBackend.mps: Accelerator.mps,
            KernelBackend.nki: Accelerator.trainium,
            KernelBackend.pallas: Accelerator.tpu,
            KernelBackend.rocm: Accelerator.rocm,
        }

        return mapping.get(self, None)

    def verify_accelerator(self) -> bool:
        expected_accelerator = self.get_compatible_accelerator()
        found_accelerator = Accelerator.get_accelerator()
        return expected_accelerator == found_accelerator


class Accelerator(Enum):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"
    rocm = "rocm"
    tpu = "tpu"
    trainium = "trainium"

    @staticmethod
    @lru_cache
    def get_accelerator() -> Accelerator:
        if is_torch_xla_available():
            accelerator = Accelerator.tpu
        elif is_torch_neuronx_available():
            accelerator = Accelerator.trainium
        elif torch.cuda.is_available():
            accelerator = Accelerator.rocm if _IS_ROCM_AVAILABLE else Accelerator.cuda
        elif torch.mps.is_available():
            accelerator = Accelerator.mps
        else:
            accelerator = Accelerator.cpu

        return accelerator

    @staticmethod
    def get_current_device() -> int | str:
        accelerator = Accelerator.get_accelerator()

        if accelerator in [Accelerator.cuda, Accelerator.rocm]:
            device = torch.cuda.current_device()
        elif accelerator == Accelerator.mps:
            device = "mps"
        elif accelerator == Accelerator.tpu:
            device = xla_device()
        elif accelerator == Accelerator.trainium:
            device = torch.neuron.current_device()
        elif accelerator == Accelerator.cpu:
            device = "cpu"

        return device

    @staticmethod
    @lru_cache
    def get_kernel_backend() -> KernelBackend:
        accelerator = Accelerator.get_accelerator()

        if accelerator == Accelerator.cuda:
            kernel_backend = KernelBackend.rocm if _IS_ROCM_AVAILABLE else KernelBackend.cuda
        elif accelerator == Accelerator.mps:
            kernel_backend = KernelBackend.mps
        elif accelerator == Accelerator.tpu:
            kernel_backend = KernelBackend.pallas
        elif accelerator == Accelerator.trainium:
            kernel_backend = KernelBackend.nki
        else:
            kernel_backend = KernelBackend.triton

        return kernel_backend

    @staticmethod
    def synchronize() -> None:
        accelerator = Accelerator.get_accelerator()

        if accelerator == Accelerator.cuda:
            torch.cuda.synchronize()
        elif accelerator == Accelerator.mps:
            torch.mps.synchronize()
        elif accelerator == Accelerator.tpu:
            xla_wait_device_ops()

    @staticmethod
    def get_sm_count(device: torch.device | None = None) -> int:
        if device is None:
            accelerator = Accelerator.get_accelerator()
        else:
            accelerator = Accelerator(device.type)

        # TODO clean this up
        if accelerator == Accelerator.cuda:
            sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        elif device.type == "xpu":
            sm_count = torch.xpu.get_device_properties(device).gpu_subslice_count

        return sm_count
