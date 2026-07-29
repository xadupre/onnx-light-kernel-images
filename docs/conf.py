# Configuration file for the Sphinx documentation builder.

project = "onnx-light-kernel-images"
copyright = "2025, xadupre"
author = "xadupre"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
