# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from unittest.mock import MagicMock


sys.path.insert(0, os.path.abspath(".."))

project = "XMA"
copyright = "2026, Mayank Mishra"
author = "Mayank Mishra"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # supports Google / NumPy style docstrings
]

autosummary_generate = True

# Don't prepend module names to class/function names
add_module_names = False


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


# List of modules to mock
MOCK_MODULES = [
    "cutlass",
    "cutlass._mlir",
    "cutlass._mlir.dialects",
    "cutlass.cute",
    "cutlass.cute.runtime",
    "cutlass.cutlass_dsl",
    "jax",
    "jax.experimental",
    "jax.experimental.pallas",
    "jax.experimental.pallas.tpu",
    "jax.nn",
    "jax.numpy",
    "neuronxcc",
    "neuronxcc.nki",
    "neuronxcc.nki.language",
    "triton",
    "triton.backends",
    "triton.language",
    "torch._inductor.runtime.triton_compat",
    "torch_neuronx",
    "torch_xla",
    "torch_xla.core",
    "torch_xla.core.xla_model",
    "torch_xla.experimental",
    "torch_xla.experimental.custom_kernel",
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = Mock()

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "tests/*"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
html_theme = "furo"
html_static_path = ["_static"]
