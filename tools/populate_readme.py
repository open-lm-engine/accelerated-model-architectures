# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from pathlib import Path

import yaml


YES = "✅"
NO = "❌"

TOOLS_DIR = Path(__file__).parent
ROOT_DIR = TOOLS_DIR.parent

kernels = yaml.full_load(open(TOOLS_DIR / "kernels.yml"))
backends = [("CUDA", "cuda"), ("Pallas", "pallas"), ("NKI", "nki"), ("ROCm", "rocm"), ("Triton", "triton")]


def get_md_table(key: str) -> str:
    """Generate markdown table rows."""
    rows = ""
    for kernel, implementations in kernels[key].items():
        rows += f"| {kernel} |"
        for backend in backends:
            rows += f" {YES if backend[1] in implementations else NO} |"
        rows += "\n"
    return rows.rstrip()


def get_rst_table(key: str, name_col: str, widths: str) -> str:
    """Generate RST list-table."""
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        f"   :widths: {widths}",
        "",
        f"   * - {name_col}",
    ]
    for backend in backends:
        lines.append(f"     - {backend[0]}")

    for kernel, implementations in kernels[key].items():
        lines.append(f"   * - {kernel}")
        for backend in backends:
            lines.append(f"     - {YES if backend[1] in implementations else NO}")

    return "\n".join(lines)


# Generate README.md
readme = f"""<!-- **************************************************
Copyright (c) 2025, Mayank Mishra
************************************************** -->

# <img src="assets/xma.png" width="90px" height="30px" style="vertical-align: middle;"> (Accelerated Model Architectures)

XMA is a repository comprising of fast kernels for model training.  
We are planning on adding lots of experimental and fun model architectures with support for multiple accelerators like NVIDIA, AMD GPUs, Google TPUs and Amazon Trainiums.

## layers

| functional | {' | '.join([i[0] for i in backends])} |
|-| {' | '.join(['-' for _ in range(len(backends))])} |
{get_md_table('layers')}

## functional

| functional | {' | '.join([i[0] for i in backends])} |
|-| {' | '.join(['-' for _ in range(len(backends))])} |
{get_md_table('functional')}

# Discord Server
Join the [discord server](https://discord.gg/AFDxmjH5RV) if you are interested in LLM architecture or distributed training/inference research.
"""

(ROOT_DIR / "README.md").write_text(readme)
print("Updated README.md")


# Generate docs/index.rst
index_rst = f"""XMA (Accelerated Model Architectures)
=====================================

XMA is a repository comprising of fast kernels for model training.
We are planning on adding lots of experimental and fun model architectures with support for multiple accelerators like NVIDIA, AMD GPUs, Google TPUs and Amazon Trainiums.

Installation
------------

.. code-block:: bash

   git clone https://github.com/open-lm-engine/accelerated-model-architectures
   cd accelerated-model-architectures
   pip install .
   cd ..

Layers
------

{get_rst_table('layers', 'Layer', '20 16 16 16 16 16')}

Functional
----------

{get_rst_table('functional', 'Function', '30 14 14 14 14 14')}

Community
---------

Join the `Discord server <https://discord.gg/AFDxmjH5RV>`_ if you are interested in LLM architecture or distributed training/inference research.

.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: API Reference

   xma.functional
   xma.layers

.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: Utilities

   xma.accelerator
   xma.counters
"""

(ROOT_DIR / "docs" / "index.rst").write_text(index_rst)
print("Updated docs/index.rst")
