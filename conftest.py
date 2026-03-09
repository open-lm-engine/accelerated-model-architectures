# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os


def pytest_configure(config) -> None:
    """Assign each xdist worker to a dedicated GPU.

    When running with `pytest -n <N>`, xdist spawns N worker sub-processes and
    sets the PYTEST_XDIST_WORKER env-var to "gw0", "gw1", … before the worker
    starts executing tests.  We map worker index → GPU index (round-robin) by
    setting CUDA_VISIBLE_DEVICES *before* any CUDA context is created, so every
    worker sees exactly one logical GPU (device 0).
    """
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is None:
        return  # single-process run – leave CUDA_VISIBLE_DEVICES unchanged

    import torch

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return  # CPU-only environment

    worker_num = int(worker_id[2:])  # "gw3" → 3
    gpu_id = worker_num % num_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
