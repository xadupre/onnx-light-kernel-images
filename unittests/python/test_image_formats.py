# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end ImageDecoder tests for every supported image format.

These build a one-node ``ImageDecoder`` ONNX model, register this package's
image kernels with the onnx-light dispatch table (``register_image_kernels``)
and run the model through onnx-light's :class:`ReferenceEvaluator`, feeding the
encoded bytestream as a ``uint8`` input tensor and checking the decoded
``(H, W, C)`` output. They mirror the C++ ``unittests/cc/test_image_decoder.cc``
cases (BMP, PNM, uncompressed TIFF and the compressed TIFF variants handled by
this package) but exercise the full model-execution path rather than calling the
kernel directly.

Running a model requires the onnx-light Python package; when it is not
importable the whole module is skipped.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from onnx_light.onnx import TensorProto, helper
    from onnx_light.onnx.reference import ReferenceEvaluator

    from onnx_light_kernel_images.onnx_py._imgpykernels import register_image_kernels

    _IMPORT_ERROR: str | None = None
except ImportError as exc:  # pragma: no cover - exercised only without onnx-light
    _IMPORT_ERROR = str(exc)


# Minimal 2x2 24-bit uncompressed BMP (BI_RGB). Bottom-up rows:
# row0=[Blue, White], row1=[Red, Green].
BMP_DATA = bytes.fromhex(
    "424d460000000000000036000000280000000200000002000000010018000000"
    "000010000000130b0000130b00000000000000000000ff0000ffffff00000000"
    "ff00ff000000"
)

# Minimal PNM P6 (binary RGB) 2x1 image: Red, Green.
PNM_DATA = b"P6\n2 1\n255\n" + bytes([0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00])

# Minimal little-endian baseline uncompressed RGB TIFF, 2x1: Red, Green.
TIFF_DATA = bytes.fromhex(
    "49492a0008000000090000010300010000000200000001010300010000000100"
    "000002010300030000007a0000000301030001000000010000000601030001000"
    "0000200000011010400010000008000000015010300010000000300000016010"
    "300010000000100000017010400010000000600000000000000080008000800ff"
    "000000ff00"
)

# Compressed variants of the 2x1 RGB TIFF above (Red, Green). Same IFD layout
# as ``TIFF_DATA`` but with a non-trivial Compression tag; this package's kernel
# rewrites them into an uncompressed baseline TIFF before decoding.

# Compression = PackBits (32773). Strip = [0x05, FF,00,00,00,FF,00].
TIFF_PACKBITS_DATA = bytes.fromhex(
    "49492a0008000000090000010300010000000200000001010300010000000100"
    "000002010300030000007a00000003010300010000000580000006010300010000"
    "000200000011010400010000008000000015010300010000000300000016010300"
    "0100000001000000170104000100000007000000000000000800080008000"
    "5ff000000ff00"
)

# Compression = LZW (5).
TIFF_LZW_DATA = bytes.fromhex(
    "49492a0008000000090000010300010000000200000001010300010000000100"
    "000002010300030000007a00000003010300010000000500000006010300010000"
    "000200000011010400010000008000000015010300010000000300000016010300"
    "01000000010000001701040001000000070000000000000008000800080080"
    "3fc010381404"
)

# Compression = Deflate/ZIP (8). Strip is a zlib-wrapped DEFLATE stream.
TIFF_DEFLATE_DATA = bytes.fromhex(
    "49492a0008000000090000010300010000000200000001010300010000000100"
    "000002010300030000007a00000003010300010000000800000006010300010000"
    "000200000011010400010000008000000015010300010000000300000016010300"
    "010000000100000017010400010000000e00000000000000080008000800"
    "78dafbcfc0c0f09f010007fe01ff"
)


def _make_image_decoder_model(pixel_format: str):
    """Builds a single-node ``ImageDecoder`` model for ``pixel_format``."""
    node = helper.make_node("ImageDecoder", ["encoded"], ["image"], pixel_format=pixel_format)
    graph = helper.make_graph(
        [node],
        "image_decoder",
        [helper.make_tensor_value_info("encoded", TensorProto.UINT8, [None])],
        [helper.make_tensor_value_info("image", TensorProto.UINT8, [None, None, None])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


@unittest.skipIf(_IMPORT_ERROR is not None, f"onnx-light not available: {_IMPORT_ERROR}")
class TestImageDecoderModel(unittest.TestCase):
    """Runs an ImageDecoder model for every supported image format."""

    @classmethod
    def setUpClass(cls):
        # Register this package's image kernels once with the onnx-light
        # dispatch table so the ImageDecoder node (including the compressed
        # TIFF support added here) is resolved when the model runs.
        register_image_kernels()

    def _decode(self, data: bytes, pixel_format: str) -> np.ndarray:
        model = _make_image_decoder_model(pixel_format)
        sess = ReferenceEvaluator(model)
        encoded = np.frombuffer(data, dtype=np.uint8)
        (image,) = sess.run(None, {"encoded": encoded})
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image.ndim, 3)
        return image

    def test_decode_bmp_rgb(self):
        image = self._decode(BMP_DATA, "RGB")
        self.assertEqual(image.shape, (2, 2, 3))
        # Row 0 (display top) = BMP row 1: Red, Green
        np.testing.assert_array_equal(image[0, 0], [255, 0, 0])
        np.testing.assert_array_equal(image[0, 1], [0, 255, 0])
        # Row 1 (display bottom) = BMP row 0: Blue, White
        np.testing.assert_array_equal(image[1, 0], [0, 0, 255])
        np.testing.assert_array_equal(image[1, 1], [255, 255, 255])

    def test_decode_bmp_grayscale(self):
        image = self._decode(BMP_DATA, "Grayscale")
        self.assertEqual(image.shape, (2, 2, 1))

    def test_decode_bmp_bgr(self):
        image = self._decode(BMP_DATA, "BGR")
        self.assertEqual(image.shape[2], 3)
        # Row 0 pixel 0 is Red in RGB => BGR = (0, 0, 255)
        np.testing.assert_array_equal(image[0, 0], [0, 0, 255])

    def test_decode_pnm_rgb(self):
        image = self._decode(PNM_DATA, "RGB")
        self.assertEqual(image.shape, (1, 2, 3))
        np.testing.assert_array_equal(image[0, 0], [255, 0, 0])  # Red
        np.testing.assert_array_equal(image[0, 1], [0, 255, 0])  # Green

    def test_decode_tiff_rgb(self):
        image = self._decode(TIFF_DATA, "RGB")
        self.assertEqual(image.shape, (1, 2, 3))
        np.testing.assert_array_equal(image[0, 0], [255, 0, 0])  # Red
        np.testing.assert_array_equal(image[0, 1], [0, 255, 0])  # Green

    def test_decode_tiff_bgr(self):
        image = self._decode(TIFF_DATA, "BGR")
        self.assertEqual(image.shape, (1, 2, 3))
        # Pixel 0 is Red in RGB => BGR = (0, 0, 255)
        np.testing.assert_array_equal(image[0, 0], [0, 0, 255])

    def test_decode_tiff_packbits(self):
        self._expect_red_green_rgb(self._decode(TIFF_PACKBITS_DATA, "RGB"))

    def test_decode_tiff_lzw(self):
        self._expect_red_green_rgb(self._decode(TIFF_LZW_DATA, "RGB"))

    def test_decode_tiff_deflate(self):
        self._expect_red_green_rgb(self._decode(TIFF_DEFLATE_DATA, "RGB"))

    def test_decode_tiff_packbits_bgr(self):
        image = self._decode(TIFF_PACKBITS_DATA, "BGR")
        self.assertEqual(image.shape[2], 3)
        # Pixel 0 is Red in RGB => BGR = (0, 0, 255)
        np.testing.assert_array_equal(image[0, 0], [0, 0, 255])

    def test_decode_unrecognized_falls_back_to_empty_matrix(self):
        garbage = bytes([0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04])
        image = self._decode(garbage, "RGB")
        # Undecodable input yields the empty (0, 0, C) matrix per the schema.
        self.assertEqual(image.shape, (0, 0, 3))

    def _expect_red_green_rgb(self, image: np.ndarray):
        self.assertEqual(image.shape, (1, 2, 3))
        np.testing.assert_array_equal(image[0, 0], [255, 0, 0])  # Red
        np.testing.assert_array_equal(image[0, 1], [0, 255, 0])  # Green


if __name__ == "__main__":
    unittest.main()
