# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

"""Remove 'module', 'package', 'Submodules', 'Subpackages' from sphinx-apidoc generated RST files."""

import re
from pathlib import Path


def clean_rst_file(filepath: Path) -> None:
    content = filepath.read_text()
    original = content

    # Remove " package" and " module" from headings (title lines followed by === or ---)
    # Pattern: line ending with " package" or " module", followed by a line of = or -
    content = re.sub(
        r"^(.+) (package|module)\n(=+|-+)$",
        lambda m: f"{m.group(1)}\n{m.group(3)[:len(m.group(1))]}",
        content,
        flags=re.MULTILINE,
    )

    # Remove "Subpackages" and "Submodules" section headers
    content = re.sub(r"^Subpackages\n-+\n+", "", content, flags=re.MULTILINE)
    content = re.sub(r"^Submodules\n-+\n+", "", content, flags=re.MULTILINE)
    content = re.sub(r"^Module contents\n-+\n+", "", content, flags=re.MULTILINE)

    if content != original:
        filepath.write_text(content)
        print(f"Cleaned: {filepath}")


def main():
    docs_dir = Path(__file__).parent.parent / "docs"
    for rst_file in docs_dir.glob("*.rst"):
        if rst_file.name not in ("index.rst", "conf.py"):
            clean_rst_file(rst_file)


if __name__ == "__main__":
    main()
