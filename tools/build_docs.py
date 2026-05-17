# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import subprocess
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_DOCS_DIR = _ROOT / "docs"


def _run(*cmd: str) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    _run("make", "-C", str(_DOCS_DIR), "clean")
    _run(
        "sphinx-apidoc",
        "--force",
        "--separate",
        "--output-dir",
        str(_DOCS_DIR),
        str(_ROOT),
        str(_ROOT / "tests"),
    )
    _run(sys.executable, str(_ROOT / "tools" / "clean_rst_headings.py"))
    _run("make", "-C", str(_DOCS_DIR), "html")


if __name__ == "__main__":
    main()
