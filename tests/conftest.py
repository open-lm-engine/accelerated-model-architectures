# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import os


def pytest_configure(config) -> None:
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is None:
        return

    import torch

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return

    gpu_id = int(worker_id.replace("gw", "")) % gpu_count
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
