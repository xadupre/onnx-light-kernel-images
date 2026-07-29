# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the automated kernels documentation scanner."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "docs" / "_ext"))

import kernel_scan  # noqa: E402


class TestKernelScan(unittest.TestCase):
    """Tests for scanning registered kernels from the C++ sources."""

    def test_scan_source_extracts_triple(self):
        """A RegisterKernelFn call yields a (domain, operator, device) triple."""
        text = (
            'RegisterKernelFn("ai.onnx", "ImageDecoder", '
            "onnx_light::core::symbolic::Device::kCPU, MakeKernel<ImageDecoder>());"
        )
        self.assertEqual(
            kernel_scan.scan_source(text),
            [("ai.onnx", "ImageDecoder", "CPU")],
        )

    def test_scan_source_ignores_unrelated_code(self):
        """Lines without a registration call yield nothing."""
        self.assertEqual(kernel_scan.scan_source("int main() { return 0; }"), [])

    def test_find_registered_kernels_includes_image_decoder(self):
        """Scanning the repository finds the ImageDecoder kernel."""
        kernels = kernel_scan.find_registered_kernels(_REPO_ROOT)
        self.assertIn(("ai.onnx", "ImageDecoder", "CPU"), kernels)

    def test_find_registered_kernels_is_sorted_and_unique(self):
        """The returned kernels are sorted and contain no duplicates."""
        kernels = kernel_scan.find_registered_kernels(_REPO_ROOT)
        self.assertEqual(kernels, sorted(set(kernels)))


if __name__ == "__main__":
    unittest.main()
