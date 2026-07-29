# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Round-trip tests for the ``decode_image`` binding (PIL save -> ONNX reload)."""

from __future__ import annotations

import io
import unittest

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is an optional test dependency
    Image = None

from onnx_light_kernel_images.onnx_py._imgpykernels import (
    decode_image,
    register_image_kernels,
)


def _make_image() -> np.ndarray:
    """Builds a small deterministic RGB image with saturated corner colors."""
    arr = np.zeros((5, 7, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, 7, dtype=np.uint8)
    arr[:, :, 2] = np.linspace(0, 255, 5, dtype=np.uint8)[:, None]
    arr[0, 0] = (255, 0, 0)
    arr[0, -1] = (0, 255, 0)
    arr[-1, 0] = (0, 0, 255)
    arr[-1, -1] = (255, 255, 255)
    return arr


@unittest.skipIf(Image is None, "Pillow is not installed")
class TestDecodeImageRoundTrip(unittest.TestCase):
    """Every lossless format must decode back to the exact original pixels."""

    # (name, PIL save kwargs)
    LOSSLESS_CASES = [
        ("BMP", {"format": "BMP"}),
        ("PNG", {"format": "PNG"}),
        ("PPM", {"format": "PPM"}),
        ("TIFF-raw", {"format": "TIFF", "compression": "raw"}),
        ("TIFF-packbits", {"format": "TIFF", "compression": "packbits"}),
        ("TIFF-lzw", {"format": "TIFF", "compression": "tiff_lzw"}),
        ("TIFF-deflate", {"format": "TIFF", "compression": "tiff_adobe_deflate"}),
    ]

    def setUp(self):
        register_image_kernels()
        self.original = _make_image()
        self.pil_image = Image.fromarray(self.original, "RGB")

    def _encode(self, save_kwargs) -> bytes:
        buffer = io.BytesIO()
        self.pil_image.save(buffer, **save_kwargs)
        return buffer.getvalue()

    def test_lossless_round_trip(self):
        for name, save_kwargs in self.LOSSLESS_CASES:
            with self.subTest(fmt=name):
                decoded = decode_image(self._encode(save_kwargs), "RGB")
                self.assertEqual(decoded.dtype, np.uint8)
                self.assertEqual(decoded.shape, self.original.shape)
                np.testing.assert_array_equal(decoded, self.original)

    def test_bgr_swaps_channels(self):
        decoded = decode_image(self._encode({"format": "PNG"}), "BGR")
        np.testing.assert_array_equal(decoded, self.original[:, :, ::-1])

    def test_grayscale_channel_count(self):
        decoded = decode_image(self._encode({"format": "PNG"}), "Grayscale")
        self.assertEqual(decoded.shape, (5, 7, 1))

    def test_jpeg_shape_matches(self):
        decoded = decode_image(self._encode({"format": "JPEG"}), "RGB")
        # JPEG is lossy, so only the shape is guaranteed.
        self.assertEqual(decoded.shape, self.original.shape)


class TestDecodeImageInvalid(unittest.TestCase):
    """Undecodable inputs fall back to the schema-mandated empty matrix."""

    def setUp(self):
        register_image_kernels()

    def test_garbage_returns_empty_matrix(self):
        decoded = decode_image(b"\xde\xad\xbe\xef not an image", "RGB")
        self.assertEqual(decoded.shape, (0, 0, 3))

    def test_empty_input_returns_empty_matrix(self):
        decoded = decode_image(b"", "Grayscale")
        self.assertEqual(decoded.shape, (0, 0, 1))


if __name__ == "__main__":
    unittest.main()
