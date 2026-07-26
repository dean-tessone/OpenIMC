# Configuration file for the Sphinx documentation builder.

import os
import sys

# Make the top-level project directory importable so `openimc` can be found
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "OpenIMC"
copyright = "2025, Dean Tessone"
author = "Dean Tessone"
release = "1.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True

# If heavy deps cause import errors when building docs, mock them here
autodoc_mock_imports = [
    "torch",
    "cellpose",
    "squidpy",
    "readimc",
    "tifffile",
    "numpy_groupies",
    "yaml",
    "anndata",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# Use a compatible theme - sphinx_rtd_theme needs to be updated for Sphinx 9+
# Using 'alabaster' as default fallback, or install sphinx_rtd_theme>=2.0.0
html_theme = "alabaster"
html_static_path = ["_static"]
