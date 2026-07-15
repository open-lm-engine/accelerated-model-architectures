# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import os
from typing import Callable


def get_ptx_from_cute_op(op: Callable, output_directory: str) -> None:
    cache = getattr(op, "cache", None)
    assert cache is not None, f"{op} has no compile cache, has it been called (and autotuned) at least once?"

    os.makedirs(output_directory, exist_ok=True)

    for key, kernel in cache.items():
        filename = str(key)
        for ch in " :,()/\\":
            filename = filename.replace(ch, "_")

        with open(os.path.join(output_directory, f"{filename}.ptx"), "w") as a:
            print(kernel.__ptx__, file=a)
