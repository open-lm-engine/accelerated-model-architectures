# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import os

from xma.utils import is_torch_available


def pytest_configure(config) -> None:
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is None:
        return

    if not is_torch_available():
        return

    import torch

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return

    gpu_id = int(worker_id.replace("gw", "")) % gpu_count
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


# test modules that don't require torch to even be importable, so they stay collectible in a torch-free
# (e.g. jax-only TPU) environment
_TORCH_FREE_TEST_FILES = {
    os.path.join("functional", "swiglu_jax_test.py"),
}


if not is_torch_available():
    collect_ignore = []
    _tests_dir = os.path.dirname(__file__)

    for _root, _, _files in os.walk(_tests_dir):
        for _name in _files:
            if not _name.endswith("_test.py"):
                continue

            _rel_path = os.path.relpath(os.path.join(_root, _name), _tests_dir)
            if _rel_path not in _TORCH_FREE_TEST_FILES:
                collect_ignore.append(_rel_path)
