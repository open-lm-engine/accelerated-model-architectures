# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import subprocess

import pytest
import torch


def test_delta_rule_context_parallel() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    gpus_per_node = torch.cuda.device_count()

    if gpus_per_node < 2:
        pytest.skip("context parallel requires at least 2 GPUs")

    command = [
        "torchrun",
        "--nproc_per_node",
        str(gpus_per_node),
        "-m",
        "tests.functional.delta_test",
    ]

    subprocess.run(command, check=True)
