# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath("_ext"))

project = "onnx-light-kernel-images"
copyright = "2025, xadupre"
author = "xadupre"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "onnx_kernels",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
