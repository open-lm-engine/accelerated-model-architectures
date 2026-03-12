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


_EXCLUDE_SUFFIXES = {"utils", "helpers", "helper", "constants", "common"}


def _is_excluded(module_name: str) -> bool:
    """Return True if any component of the dotted module name is a utility/helper."""
    for part in module_name.split("."):
        if part in _EXCLUDE_SUFFIXES or any(part.endswith(f"_{s}") for s in _EXCLUDE_SUFFIXES):
            return True
    return False


def delete_unwanted_files(docs_dir: Path) -> None:
    for rst_file in docs_dir.glob("*.rst"):
        if rst_file.name == "index.rst":
            continue
        module_name = rst_file.stem
        # Delete implementation files (triton/cuda/nki/pallas kernels)
        if any(f"{impl}_implementation" in module_name for impl in ["triton", "cuda", "nki", "pallas"]):
            rst_file.unlink()
            print(f"Deleted: {rst_file}")
        # Delete utility/helper files
        elif _is_excluded(module_name):
            rst_file.unlink()
            print(f"Deleted: {rst_file}")


def _find_toctree_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """Return list of (start, end) line index pairs for each toctree block."""
    blocks = []
    i = 0
    while i < len(lines):
        if re.match(r"^\.\. toctree::", lines[i]):
            start = i
            j = i + 1
            while j < len(lines):
                # Toctree block ends at a non-indented, non-empty line
                if lines[j] and not lines[j][0].isspace():
                    break
                j += 1
            blocks.append((start, j))
            i = j
        else:
            i += 1
    return blocks


def _get_toctree_entries(lines: list[str], start: int, end: int) -> list[int]:
    """Return line indices of entry lines within a toctree block."""
    entry_lines = []
    for i in range(start + 1, end):
        stripped = lines[i].strip()
        if stripped and not stripped.startswith(":"):
            entry_lines.append(i)
    return entry_lines


def _insert_into_toctree(content: str, new_entries: list[str], toctree_index: int) -> str:
    """Insert new_entries into the toctree at toctree_index (0=first, -1=last), sorted."""
    lines = content.splitlines(keepends=True)
    blocks = _find_toctree_blocks([l.rstrip("\n") for l in lines])

    if not blocks:
        return content

    start, end = blocks[toctree_index % len(blocks)]
    entry_indices = _get_toctree_entries([l.rstrip("\n") for l in lines], start, end)

    existing = [lines[i].strip() for i in entry_indices]
    merged = sorted(set(existing) | set(new_entries))
    new_lines = ["   " + e + "\n" for e in merged]

    if entry_indices:
        lines[entry_indices[0] : entry_indices[-1] + 1] = new_lines
    else:
        # No existing entries: insert after options block (lines starting with ":")
        insert_at = start + 1
        while insert_at < end:
            s = lines[insert_at].strip()
            if s.startswith(":") or not s:
                insert_at += 1
            else:
                break
        lines[insert_at:insert_at] = new_lines

    return "".join(lines)


def remove_dangling_toctree_entries(docs_dir: Path) -> None:
    """Remove toctree entries that reference rst files that no longer exist."""
    for rst_file in sorted(docs_dir.glob("*.rst")):
        content = rst_file.read_text()
        if ".. toctree::" not in content:
            continue

        lines = content.splitlines(keepends=True)
        flat = [l.rstrip("\n") for l in lines]
        blocks = _find_toctree_blocks(flat)
        removed = []

        for start, end in blocks:
            for i in _get_toctree_entries(flat, start, end):
                entry = flat[i].strip()
                if not (docs_dir / f"{entry}.rst").exists():
                    lines[i] = ""
                    removed.append(entry)

        if removed:
            rst_file.write_text("".join(lines))
            print(f"Removed dangling entries from {rst_file.name}: {removed}")


def update_parent_toctrees(docs_dir: Path, source_root: Path) -> None:
    """Add newly generated rst entries to parent toctrees if not already present."""
    for parent_rst in sorted(docs_dir.glob("*.rst")):
        if parent_rst.name == "index.rst":
            continue

        content = parent_rst.read_text()
        if ".. toctree::" not in content:
            continue

        module_name = parent_rst.stem  # e.g. "xma.functional"
        prefix = module_name + "."

        referenced = set(re.findall(r"^ {3}(\S+)$", content, re.MULTILINE))

        new_packages: list[str] = []
        new_modules: list[str] = []

        for rst_file in sorted(docs_dir.glob(f"{prefix}*.rst")):
            child_name = rst_file.stem  # e.g. "xma.functional.gru"
            suffix = child_name[len(prefix) :]
            if "." in suffix:
                continue  # grandchild or deeper, skip
            if child_name in referenced:
                continue
            if _is_excluded(suffix):
                continue

            # Determine package vs module by checking source filesystem
            parts = child_name.split(".")
            if (source_root.joinpath(*parts) / "__init__.py").exists():
                new_packages.append(child_name)
            else:
                new_modules.append(child_name)

        if not new_packages and not new_modules:
            continue

        if new_packages:
            content = _insert_into_toctree(content, new_packages, toctree_index=0)
        if new_modules:
            content = _insert_into_toctree(content, new_modules, toctree_index=-1)

        parent_rst.write_text(content)
        added = new_packages + new_modules
        print(f"Updated {parent_rst.name}: added {added}")


def main():
    docs_dir = Path(__file__).parent.parent / "docs"
    source_root = Path(__file__).parent.parent
    delete_unwanted_files(docs_dir)
    remove_dangling_toctree_entries(docs_dir)
    update_parent_toctrees(docs_dir, source_root)
    for rst_file in docs_dir.glob("*.rst"):
        if rst_file.name not in ("index.rst", "conf.py"):
            clean_rst_file(rst_file)


if __name__ == "__main__":
    main()
