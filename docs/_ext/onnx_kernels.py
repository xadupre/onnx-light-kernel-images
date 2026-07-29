# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Sphinx extension rendering the repository's registered kernels as a table.

Provides the ``onnx-light-kernels`` directive which scans the C++ registration
sources (see :mod:`kernel_scan`) and emits a table with one row per registered
kernel. Because the table is generated at build time, it always reflects the
kernels the repository actually registers.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from docutils import nodes
from docutils.parsers.rst import Directive, directives

from kernel_scan import DEFAULT_GLOBS, Kernel, find_registered_kernels, iter_source_files

_HEADERS = ("Domain", "Operator", "Device")


def _row(cells: Sequence[str]) -> nodes.row:
    row = nodes.row()
    for cell in cells:
        entry = nodes.entry()
        entry += nodes.paragraph(text=str(cell))
        row += entry
    return row


def _build_table(kernels: List[Kernel]) -> nodes.table:
    table = nodes.table()
    table["classes"].append("onnx-light-kernels")
    tgroup = nodes.tgroup(cols=len(_HEADERS))
    table += tgroup
    for _ in _HEADERS:
        tgroup += nodes.colspec(colwidth=1)

    thead = nodes.thead()
    thead += _row(_HEADERS)
    tgroup += thead

    tbody = nodes.tbody()
    for domain, op, device in kernels:
        tbody += _row((domain, op, device))
    tgroup += tbody
    return table


class OnnxLightKernelsDirective(Directive):
    """Render a table of the kernels registered by this repository."""

    has_content = False
    option_spec = {"glob": directives.unchanged}

    def run(self) -> List[nodes.Node]:
        env = self.state.document.settings.env
        root = Path(env.srcdir).parent
        globs = tuple(self.options["glob"].split()) if "glob" in self.options else DEFAULT_GLOBS

        # Rebuild the page whenever a scanned source file changes.
        for path in iter_source_files(root, globs):
            env.note_dependency(str(path))

        kernels = find_registered_kernels(root, globs)
        if not kernels:
            warning = self.state_machine.reporter.warning(
                "onnx-light-kernels: no registered kernels found under "
                f"{root} for globs {globs}.",
                line=self.lineno,
            )
            paragraph = nodes.paragraph(text="No kernels are currently registered.")
            return [paragraph, warning]

        return [_build_table(kernels)]


def setup(app):
    app.add_directive("onnx-light-kernels", OnnxLightKernelsDirective)
    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
