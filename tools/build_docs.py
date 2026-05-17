# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import re
import subprocess
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_DOCS_DIR = _ROOT / "docs"


def _run(*cmd: str) -> None:
    subprocess.run(cmd, check=True)


def _is_package(dotted_name: str) -> bool:
    parts = dotted_name.split(".")
    return (_ROOT.joinpath(*parts) / "__init__.py").exists()


_AUTOMODULE_OPTIONS = "   :members:\n   :undoc-members:\n   :show-inheritance:\n"
_KEEP_FILES = {"conf.py", "index.rst", "Makefile", "make.bat"}


def _clean_generated_rsts() -> None:
    """Delete all sphinx-apidoc-generated RSTs, keeping hand-crafted files."""
    for f in _DOCS_DIR.glob("*.rst"):
        if f.name not in _KEEP_FILES:
            f.unlink()


def _flatten_to_entry_points() -> None:
    """
    For each package RST:
      - Remove toctree entries for plain modules (not sub-packages).
      - For each removed entry whose RST still exists (real content, not already
        deleted as an implementation file), inline an automodule directive so
        those symbols appear directly on the parent page.
    Then delete the now-orphaned plain-module RSTs.

    This avoids :imported-members: which would also pull in external symbols
    (e.g. Accelerator, KernelBackend) from helper imports and cause duplicates.
    """
    orphaned: set[str] = set()

    for rst_path in sorted(_DOCS_DIR.glob("*.rst")):
        stem = rst_path.stem
        if stem in ("index", "modules"):
            continue
        if not _is_package(stem):
            continue

        lines = rst_path.read_text().splitlines(keepends=True)
        out: list[str] = []
        i = 0
        inlined: list[str] = []  # plain-module stems whose content to inline

        while i < len(lines):
            line = lines[i]

            if re.match(r"^\.\. toctree::", line):
                block: list[str] = [line]
                i += 1
                while i < len(lines):
                    l = lines[i]
                    if l.strip() and not l[0].isspace():
                        break
                    block.append(l)
                    i += 1

                filtered: list[str] = []
                for bl in block:
                    bs = bl.strip()
                    if bs and not bs.startswith(":") and not bs.startswith(".."):
                        if _is_package(bs):
                            filtered.append(bl)
                        else:
                            orphaned.add(bs)
                            if (_DOCS_DIR / f"{bs}.rst").exists():
                                inlined.append(bs)
                    else:
                        filtered.append(bl)

                has_entries = any(
                    fl.strip() and not fl.strip().startswith(":") and not fl.strip().startswith("..")
                    for fl in filtered
                )
                if has_entries:
                    out.extend(filtered)

                # Inline content from removed plain-module RSTs before automodule
                for mod in inlined:
                    out.append(f"\n.. automodule:: {mod}\n{_AUTOMODULE_OPTIONS}")
                inlined.clear()
                continue

            out.append(line)
            i += 1

        rst_path.write_text("".join(out))

    for stem in orphaned:
        p = _DOCS_DIR / f"{stem}.rst"
        if p.exists():
            p.unlink()


def main() -> None:
    _run("make", "-C", str(_DOCS_DIR), "clean")
    _clean_generated_rsts()
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
    _flatten_to_entry_points()
    _run("make", "-C", str(_DOCS_DIR), "html")


if __name__ == "__main__":
    main()
