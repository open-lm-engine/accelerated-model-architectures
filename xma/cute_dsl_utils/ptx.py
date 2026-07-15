# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable


def _sanitize_cache_key(key: object) -> str:
    filename = str(key)
    for ch in " :,()/\\":
        filename = filename.replace(ch, "_")

    return filename


def _find_nvdisasm() -> str | None:
    path = shutil.which("nvdisasm")
    if path is not None:
        return path

    for cuda_dir in sorted(Path("/usr/local").glob("cuda*"), reverse=True):
        candidate = cuda_dir / "bin" / "nvdisasm"
        if candidate.is_file():
            return str(candidate)

    return None


def get_ptx_from_cute_op(op: Callable, output_directory: str) -> None:
    os.makedirs(output_directory, exist_ok=True)

    for key, kernel in op.cache.items():
        filename = _sanitize_cache_key(key)

        with open(os.path.join(output_directory, f"{filename}.ptx"), "w") as a:
            print(kernel.__ptx__, file=a)


def get_sass_from_cute_op(op: Callable, output_directory: str) -> None:
    nvdisasm = _find_nvdisasm()
    assert nvdisasm is not None, "nvdisasm not found on PATH or under /usr/local/cuda*/bin"

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    for key, kernel in op.cache.items():
        filename = _sanitize_cache_key(key)

        cubin_path = output_directory / f"{filename}.cubin"
        cubin_path.write_bytes(kernel.__cubin__)

        result = subprocess.run([nvdisasm, str(cubin_path)], capture_output=True, text=True)
        assert result.returncode == 0, f"nvdisasm failed for {filename}: {result.stderr.strip()}"

        (output_directory / f"{filename}.sass").write_text(result.stdout)
