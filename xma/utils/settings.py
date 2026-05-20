# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch


def get_triton_num_warps(BLOCK_SIZE: int) -> int:
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if torch.version.hip is None else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    return num_warps
