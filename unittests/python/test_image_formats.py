# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Decoding tests for every image format supported by the ImageDecoder.

These mirror the C++ ``unittests/cc/test_image_decoder.cc`` cases, exercising
the same minimal 2x2 / 2x1 image bytestreams through the ``decode_image``
Python binding so that BMP, PNM, JPEG2000 and (compressed) TIFF are all covered
from Python as well. The byte-streams are annotated in the C++ file; here they
are kept as compact hex literals.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import unittest

from onnx_light_kernel_images.onnx_py._imgpykernels import decode_image

# Minimal 2x2 24-bit uncompressed BMP (BI_RGB). Bottom-up rows:
# row0=[Blue, White], row1=[Red, Green].
BMP_DATA = bytes.fromhex(
    "424d460000000000000036000000280000000200000002000000010018000000"
    "000010000000130b0000130b00000000000000000000ff0000ffffff00000000"
    "ff00ff000000"
)

# Minimal PNM P6 (binary RGB) 2x1 image: Red, Green.
PNM_DATA = b"P6\n2 1\n255\n" + bytes([0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00])

# Minimal lossless JPEG2000 (JP2 file format) 2x1 image: Red, Green.
JP2_DATA = bytes.fromhex(
    "0000000c6a5020200d0a870a00000014667479706a703220000000006a703220"
    "0000002d6a703268000000166968647200000001000000020003070700000000"
    "000f636f6c7201000000000010000000986a703263ff4fff51002f0000000000"
    "02000000010000000000000000000000020000000100000000000000000003070"
    "101070101070101ff52000c00000001000004040001ff5c00044040ff640025"
    "000143726561746564206279204f70656e4a5045472076657273696f6e20322e"
    "352e34ff90000a0000000000200001ff93df80200bb28a7fdf801805a2dddf80"
    "10093fffd9"
)

# Minimal little-endian baseline uncompressed RGB TIFF, 2x1: Red, Green.
TIFF_DATA = bytes.fromhex(
    "49492a0008000000090000010300010000000200000001010300010000000100"
    "000002010300030000007a0000000301030001000000010000000601030001000"
    "0000200000011010400010000008000000015010300010000000300000016010"
    "300010000000100000017010400010000000600000000000000080008000800ff"
    "000000ff00"
)

# Compressed variants of the 2x1 RGB TIFF above (Red, Green). Same IFD layout
# as ``TIFF_DATA`` but with a non-trivial Compression tag; ``decode_image``
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


def _openjpeg_runtime_available() -> bool:
    """Returns True when the OpenJPEG runtime (libopenjp2) can be loaded.

    Mirrors the kernel's own runtime gating so the JPEG2000 test can assert
    full decoding only when the optional dependency is present.
    """
    names = ["libopenjp2.so.7", "libopenjp2.so", "libopenjp2.dll", "openjp2.dll"]
    found = ctypes.util.find_library("openjp2")
    if found:
        names.insert(0, found)
    for name in names:
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


class TestImageFormats(unittest.TestCase):
    """Decoding tests for every supported image format."""

    def test_decode_bmp_rgb(self):
        shape, data = decode_image(BMP_DATA, "RGB")
        self.assertEqual(shape, [2, 2, 3])
        px = list(data)
        # Row 0 (display top) = BMP row 1: Red, Green
        self.assertEqual(px[0:3], [255, 0, 0])
        self.assertEqual(px[3:6], [0, 255, 0])
        # Row 1 (display bottom) = BMP row 0: Blue, White
        self.assertEqual(px[6:9], [0, 0, 255])
        self.assertEqual(px[9:12], [255, 255, 255])

    def test_decode_bmp_grayscale(self):
        shape, _ = decode_image(BMP_DATA, "Grayscale")
        self.assertEqual(shape, [2, 2, 1])

    def test_decode_bmp_bgr(self):
        shape, data = decode_image(BMP_DATA, "BGR")
        self.assertEqual(shape[2], 3)
        # Row 0 pixel 0 is Red in RGB => BGR = (0, 0, 255)
        self.assertEqual(list(data)[0:3], [0, 0, 255])

    def test_decode_pnm_rgb(self):
        shape, data = decode_image(PNM_DATA, "RGB")
        self.assertEqual(shape, [1, 2, 3])
        px = list(data)
        self.assertEqual(px[0:3], [255, 0, 0])  # Red
        self.assertEqual(px[3:6], [0, 255, 0])  # Green

    def test_decode_jpeg2000_rgb(self):
        shape, data = decode_image(JP2_DATA, "RGB")
        self.assertEqual(shape[2], 3)  # channels
        if _openjpeg_runtime_available():
            self.assertEqual(shape[0], 1)  # height
            self.assertEqual(shape[1], 2)  # width
            px = list(data)
            self.assertEqual(px[0:3], [255, 0, 0])  # Red
            self.assertEqual(px[3:6], [0, 255, 0])  # Green
        else:
            # Without the runtime library the kernel returns an empty matrix.
            self.assertEqual(shape[0], 0)
            self.assertEqual(shape[1], 0)

    def test_decode_tiff_rgb(self):
        shape, data = decode_image(TIFF_DATA, "RGB")
        self.assertEqual(shape, [1, 2, 3])
        px = list(data)
        self.assertEqual(px[0:3], [255, 0, 0])  # Red
        self.assertEqual(px[3:6], [0, 255, 0])  # Green

    def test_decode_tiff_bgr(self):
        shape, data = decode_image(TIFF_DATA, "BGR")
        self.assertEqual(shape, [1, 2, 3])
        # Pixel 0 is Red in RGB => BGR = (0, 0, 255)
        self.assertEqual(list(data)[0:3], [0, 0, 255])

    def test_decode_tiff_packbits(self):
        self._expect_red_green_rgb(decode_image(TIFF_PACKBITS_DATA, "RGB"))

    def test_decode_tiff_lzw(self):
        self._expect_red_green_rgb(decode_image(TIFF_LZW_DATA, "RGB"))

    def test_decode_tiff_deflate(self):
        self._expect_red_green_rgb(decode_image(TIFF_DEFLATE_DATA, "RGB"))

    def test_decode_tiff_packbits_bgr(self):
        shape, data = decode_image(TIFF_PACKBITS_DATA, "BGR")
        self.assertEqual(shape[2], 3)
        # Pixel 0 is Red in RGB => BGR = (0, 0, 255)
        self.assertEqual(list(data)[0:3], [0, 0, 255])

    def test_invalid_empty_input(self):
        # Empty input falls back to an empty matrix (0, 0, 3).
        shape, data = decode_image(b"", "RGB")
        self.assertEqual(shape, [0, 0, 3])
        self.assertEqual(len(data), 0)

    def test_unrecognized_format_falls_back_to_empty_matrix(self):
        garbage = bytes([0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04])
        shape, data = decode_image(garbage, "RGB")
        self.assertEqual(shape, [0, 0, 3])
        self.assertEqual(len(data), 0)

    def test_default_pixel_format_is_rgb(self):
        shape_default, data_default = decode_image(TIFF_DATA)
        shape_rgb, data_rgb = decode_image(TIFF_DATA, "RGB")
        self.assertEqual(shape_default, shape_rgb)
        self.assertEqual(data_default, data_rgb)

    def _expect_red_green_rgb(self, decoded):
        shape, data = decoded
        self.assertEqual(shape, [1, 2, 3])
        px = list(data)
        self.assertEqual(px[0:3], [255, 0, 0])  # Red
        self.assertEqual(px[3:6], [0, 255, 0])  # Green


if __name__ == "__main__":
    unittest.main()
