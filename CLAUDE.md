<!-- **************************************************
Copyright (c) 2026, Mayank Mishra
************************************************** -->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Always activate the uv virtual environment before running any Python or shell commands:

```bash
source ./venv/bin/activate
```

All `python`, `pytest`, and `uv` commands must be run inside this environment.

## Commands

```bash
# Install dependencies (cuda extras by default)
uv sync --extra cuda

# Run all tests
make test

# Run a single test file
pytest tests/functional/test_swiglu.py -v

# Run a single test by name
pytest tests/functional/test_swiglu.py::test_swiglu_forward -v

# Lint / style check
make style

# Build docs
make website

# Serve docs locally (port 8001)
make host-website
```

`make test` calls pytest with accelerator-specific extras; the default accelerator is `cuda`. To override: `make test ACCELERATOR=cpu`.

## Architecture

**XMA** is a multi-backend kernel library for PyTorch. It provides optimized implementations of common deep learning operations across CUDA, ROCm, Apple MPS, Google TPU (Pallas), AWS Trainium (NKI), and Triton.

### Key abstractions

- **`xma/accelerator.py`** — `Accelerator` enum detects the runtime hardware (CUDA, ROCm, MPS, TPU, Trainium, CPU). `KernelBackend` enum maps each operation to the right backend. All multi-backend kernels branch on these at call time.
- **`xma/custom_op.py`** — registers custom autograd functions so kernels integrate cleanly with `torch.compile`.
- **`xma/inductor.py`** — PyTorch Inductor lowering passes (e.g., fusing RMSNorm into a single kernel).
- **`xma/jit.py`** — JIT/ahead-of-time compilation helpers shared across backends.

### Layout convention

Each operation lives under either `xma/functional/<op>/` or `xma/layers/<op>/`:

```
xma/functional/<op>/
    __init__.py        # public API (e.g. swiglu())
    triton_impl.py     # Triton kernel
    cuda_impl.py       # CUDA/C++ kernel (optional)
    backward.py        # backward pass (if non-trivial)

xma/layers/<op>/
    __init__.py        # nn.Module wrapper
```

When adding a new backend for an existing op, follow the existing pattern: add `<backend>_impl.py` and branch inside `__init__.py` using `Accelerator`.

### Multi-accelerator support matrix

Tracked in `tools/kernels.yml` and auto-rendered into `README.md` by `tools/populate_readme.py`. Edit `kernels.yml`, not the README table directly.

### Testing conventions

- Tests live under `tests/` mirroring `xma/` (e.g., `tests/functional/test_swiglu.py`).
- `tests/utils.py` has shared helpers for creating test tensors and shapes.
- `tests/conftest.py` holds pytest fixtures.
- Tolerances vary by dtype and backend; check existing tests before setting `atol`/`rtol`.

### Triton utilities

`xma/triton_utils/` contains shared helpers (e.g., block-pointer helpers, tiling utilities) used across Triton kernels. `xma/autotuner/` is the auto-tuning framework for selecting optimal Triton configs at runtime.

### CUTLASS / cute_dsl kernels

`xma/cute_dsl_utils/` wraps NVIDIA's CUTLASS DSL for writing high-performance GEMM and related kernels directly in Python. Only available with the `cuda` extra.
