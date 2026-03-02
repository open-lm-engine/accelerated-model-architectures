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

# Mock modules that aren't installed during doc builds (handles all submodules automatically)
autodoc_mock_imports = [
    "cutlass",
    "jax",
    "neuronxcc",
    "torch._dynamo",
    "torch._inductor",
    "torch.compiler",
    "triton",
    "torch_neuronx",
    "torch_xla",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "tests/*"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
html_theme = "furo"
html_static_path = ["_static"]
