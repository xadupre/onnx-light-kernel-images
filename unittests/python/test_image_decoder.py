# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the onnx-light-kernel-images Python extension."""

from __future__ import annotations

import unittest


class TestImageKernelBindings(unittest.TestCase):
    """Tests for the _imgpykernels nanobind module."""

    def test_import(self):
        """The extension module is importable."""
        from onnx_light_kernel_images.onnx_py import _imgpykernels  # noqa: F401

    def test_has_image_kernels(self):
        """has_image_kernels() returns True."""
        from onnx_light_kernel_images.onnx_py._imgpykernels import has_image_kernels

        self.assertTrue(has_image_kernels())

    def test_register_image_kernels_is_callable(self):
        """register_image_kernels() is callable and does not raise."""
        from onnx_light_kernel_images.onnx_py._imgpykernels import register_image_kernels

        register_image_kernels()

    def test_register_image_kernels_idempotent(self):
        """Calling register_image_kernels() twice does not raise."""
        from onnx_light_kernel_images.onnx_py._imgpykernels import register_image_kernels

        register_image_kernels()
        register_image_kernels()

    def test_module_docstring(self):
        """The module has a docstring mentioning ImageDecoder."""
        from onnx_light_kernel_images.onnx_py import _imgpykernels

        self.assertIn("ImageDecoder", _imgpykernels.__doc__)

    def test_decode_image_is_callable(self):
        """decode_image() is exposed and returns a (shape, bytes) pair."""
        from onnx_light_kernel_images.onnx_py._imgpykernels import decode_image

        shape, data = decode_image(b"", "RGB")
        self.assertEqual(shape, [0, 0, 3])
        self.assertEqual(data, b"")


if __name__ == "__main__":
    unittest.main()
