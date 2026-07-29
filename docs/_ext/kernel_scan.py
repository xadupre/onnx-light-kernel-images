# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Scan the C++ sources for the kernels registered by this repository.

The kernels are registered at runtime through ``RegisterKernelFn`` calls of the
form::

    RegisterKernelFn("ai.onnx", "ImageDecoder",
                     onnx_light::core::symbolic::Device::kCPU,
                     MakeKernel<ImageDecoder>());

This module extracts ``(domain, operator, device)`` triples from those calls so
the documentation can present an always up-to-date table of available kernels
without any manual bookkeeping.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

# Matches ``RegisterKernelFn("domain", "Operator", ...Device::kDevice`` and
# captures the domain, operator name and device (without the ``k`` prefix).
_REGISTER_RE = re.compile(
    r"RegisterKernelFn\(\s*"
    r'"(?P<domain>[^"]+)"\s*,\s*'
    r'"(?P<op>[^"]+)"\s*,\s*'
    r"[\w:]*Device::k(?P<device>\w+)",
)

# Source globs (relative to the repository root) that may register kernels.
DEFAULT_GLOBS: Tuple[str, ...] = ("onnx_light_kernel_images/**/*.cc",)

Kernel = Tuple[str, str, str]


def scan_source(text: str) -> List[Kernel]:
    """Return the ``(domain, operator, device)`` triples found in ``text``."""
    return [
        (m.group("domain"), m.group("op"), m.group("device")) for m in _REGISTER_RE.finditer(text)
    ]


def iter_source_files(root: Path, globs: Tuple[str, ...] = DEFAULT_GLOBS) -> List[Path]:
    """Return the sorted list of source files matched by ``globs`` under ``root``."""
    files: List[Path] = []
    for pattern in globs:
        files.extend(root.glob(pattern))
    return sorted(set(files))


def find_registered_kernels(root: Path, globs: Tuple[str, ...] = DEFAULT_GLOBS) -> List[Kernel]:
    """Return the sorted, de-duplicated kernels registered under ``root``."""
    kernels: List[Kernel] = []
    seen = set()
    for path in iter_source_files(root, globs):
        for kernel in scan_source(path.read_text(encoding="utf-8")):
            if kernel not in seen:
                seen.add(kernel)
                kernels.append(kernel)
    kernels.sort()
    return kernels
