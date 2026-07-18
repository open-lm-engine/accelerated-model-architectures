# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import os

from xma import Accelerator, is_torch_available


def pytest_configure(config) -> None:
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is None:
        return

    if not is_torch_available() or Accelerator.get_accelerator() not in [Accelerator.cuda, Accelerator.rocm]:
        return

    gpu_count = Accelerator.device_count()
    if gpu_count == 0:
        return

    gpu_id = int(worker_id.replace("gw", "")) % gpu_count
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
