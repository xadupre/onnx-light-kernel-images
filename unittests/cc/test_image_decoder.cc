// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_kernel_images/kernels/image/legacy_image_codecs.h"
#include "onnx_light_kernel_images/register_image_kernels.h"
#include "onnx_light_kernel_images/tiff_compression.h"

#include "onnx.h"
#include "onnx_core/compute/raw_buffer_allocator.h"
#if __has_include("onnx_core/runtime/kernels/kernel_context.h")
#define ONNX_LIGHT_KERNEL_IMAGES_HAS_CURRENT_BACKEND_TEST_API 1
#include "onnx_core/backend_test/expect.h"
#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/kernels/kernel_context.h"
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/kernels/tensor_compare.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#else
#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/kernel_dispatch_table.h"
#include "onnx_core/runtime/run_nodes.h"
#include "onnx_core/runtime/simple_tensor.h"
#endif
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernels/image/include_image_kernels.h"
#include "onnx_threads/cpu_executor.h"
#include "onnx_threads/parallel_for.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

using namespace onnx_light;
using core::runtime::DataType;
using core::runtime::KernelDispatchTable;
using core::runtime::Tensor;
using onnx_kernels::kernel::ImageDecoder;
using onnx_kernels::kernel::KernelContext;

namespace {

#if ONNX_LIGHT_KERNEL_IMAGES_HAS_CURRENT_BACKEND_TEST_API
using core::backend_test::CollectTestCases;
using core::backend_test::DataSet;
using core::backend_test::TestCase;
using core::backend_test::TestCaseKind;
using core::backend_test::TestCaseTag;
using core::runtime::CompareTensors;
using core::runtime::RunModel;
using core::runtime::TensorComparison;
using core::runtime::Tensors;

class TestCaseUnloadGuard {
public:
  explicit TestCaseUnloadGuard(TestCase &test_case) : test_case_(test_case) {}

  ~TestCaseUnloadGuard() {
    if (test_case_.materialized()) {
      test_case_.unload();
    }
  }

  TestCaseUnloadGuard(const TestCaseUnloadGuard &) = delete;
  TestCaseUnloadGuard &operator=(const TestCaseUnloadGuard &) = delete;

private:
  TestCase &test_case_;
};
#endif

// Minimal 2x2 24-bit uncompressed BMP (BI_RGB, BITMAPINFOHEADER).
// Pixel layout (bottom-up): row0=[Red, Green], row1=[Blue, White].
// clang-format off
const unsigned char kBmpData[] = {
    // File header (14 bytes)
    0x42, 0x4D,             // "BM"
    0x46, 0x00, 0x00, 0x00, // file size = 70
    0x00, 0x00, 0x00, 0x00, // reserved
    0x36, 0x00, 0x00, 0x00, // pixel offset = 54
    // DIB header (BITMAPINFOHEADER, 40 bytes)
    0x28, 0x00, 0x00, 0x00, // header size = 40
    0x02, 0x00, 0x00, 0x00, // width = 2
    0x02, 0x00, 0x00, 0x00, // height = 2 (bottom-up)
    0x01, 0x00,             // planes = 1
    0x18, 0x00,             // bpp = 24
    0x00, 0x00, 0x00, 0x00, // compression = BI_RGB
    0x10, 0x00, 0x00, 0x00, // image size = 16
    0x13, 0x0B, 0x00, 0x00, // X pix/m
    0x13, 0x0B, 0x00, 0x00, // Y pix/m
    0x00, 0x00, 0x00, 0x00, // colors in table
    0x00, 0x00, 0x00, 0x00, // important colors
    // Pixel data (bottom-up, padded rows to 4-byte boundary)
    // Row 0 (bottom): Blue(BGR=255,0,0), White(BGR=255,255,255)
    0xFF, 0x00, 0x00,       // pixel (0,0) BGR = Blue
    0xFF, 0xFF, 0xFF,       // pixel (0,1) BGR = White
    0x00, 0x00,             // padding to 4-byte boundary
    // Row 1 (top): Red(BGR=0,0,255), Green(BGR=0,255,0)
    0x00, 0x00, 0xFF,       // pixel (1,0) BGR = Red
    0x00, 0xFF, 0x00,       // pixel (1,1) BGR = Green
    0x00, 0x00,             // padding
};
// clang-format on

// Minimal PNM P6 (binary RGB) 2x1 image.
const unsigned char kPnmData[] = {
    'P',
    '6',
    '\n',
    '2',
    ' ',
    '1',
    '\n',
    '2',
    '5',
    '5',
    '\n',
    // pixel (0,0) RGB = (255, 0, 0) = Red
    0xFF,
    0x00,
    0x00,
    // pixel (0,1) RGB = (0, 255, 0) = Green
    0x00,
    0xFF,
    0x00,
};

// Minimal lossless JPEG2000 (JP2 file format) 2x1 image, generated with
// OpenJPEG (reversible 5/3 wavelet). Pixel (0,0) RGB = Red, pixel (0,1) = Green.
// clang-format off
const unsigned char kJp2Data[] = {
    0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50, 0x20, 0x20, 0x0D, 0x0A, 0x87, 0x0A,
    0x00, 0x00, 0x00, 0x14, 0x66, 0x74, 0x79, 0x70, 0x6A, 0x70, 0x32, 0x20,
    0x00, 0x00, 0x00, 0x00, 0x6A, 0x70, 0x32, 0x20, 0x00, 0x00, 0x00, 0x2D,
    0x6A, 0x70, 0x32, 0x68, 0x00, 0x00, 0x00, 0x16, 0x69, 0x68, 0x64, 0x72,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x03, 0x07, 0x07,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x0F, 0x63, 0x6F, 0x6C, 0x72, 0x01, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x98, 0x6A, 0x70, 0x32,
    0x63, 0xFF, 0x4F, 0xFF, 0x51, 0x00, 0x2F, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x07, 0x01, 0x01, 0x07, 0x01,
    0x01, 0x07, 0x01, 0x01, 0xFF, 0x52, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x01,
    0x00, 0x00, 0x04, 0x04, 0x00, 0x01, 0xFF, 0x5C, 0x00, 0x04, 0x40, 0x40,
    0xFF, 0x64, 0x00, 0x25, 0x00, 0x01, 0x43, 0x72, 0x65, 0x61, 0x74, 0x65,
    0x64, 0x20, 0x62, 0x79, 0x20, 0x4F, 0x70, 0x65, 0x6E, 0x4A, 0x50, 0x45,
    0x47, 0x20, 0x76, 0x65, 0x72, 0x73, 0x69, 0x6F, 0x6E, 0x20, 0x32, 0x2E,
    0x35, 0x2E, 0x34, 0xFF, 0x90, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x20, 0x00, 0x01, 0xFF, 0x93, 0xDF, 0x80, 0x20, 0x0B, 0xB2, 0x8A, 0x7F,
    0xDF, 0x80, 0x18, 0x05, 0xA2, 0xDD, 0xDF, 0x80, 0x10, 0x09, 0x3F, 0xFF,
    0xD9,
};
// clang-format on

// Minimal lossless WebP (VP8L) 2x1 image. Pixel (0,0) RGB = Red,
// pixel (0,1) = Green. Decoded by the ImageDecoder through libwebp.
// clang-format off
const unsigned char kWebpData[] = {
    0x52, 0x49, 0x46, 0x46, 0x18, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
    0x56, 0x50, 0x38, 0x4C, 0x0C, 0x00, 0x00, 0x00, 0x2F, 0x01, 0x00, 0x00,
    0x00, 0x98, 0xFF, 0xF9, 0x8F, 0xFE, 0x87, 0x01,
};
// clang-format on

// Minimal little-endian baseline uncompressed RGB TIFF, 2x1 (chunky,
// 8 bits per sample). Pixel (0,0) = Red (255,0,0), pixel (0,1) = Green
// (0,255,0). BitsPerSample is stored out-of-line as [8, 8, 8].
// clang-format off
const unsigned char kTiffData[] = {
    0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, // header: "II", 42, IFD@8
    0x09, 0x00,                                     // 9 IFD entries
    0x00, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, // ImageWidth=2
    0x01, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, // ImageLength=1
    0x02, 0x01, 0x03, 0x00, 0x03, 0x00, 0x00, 0x00, 0x7A, 0x00, 0x00, 0x00, // BitsPerSample@122
    0x03, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, // Compression=none
    0x06, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, // Photometric=RGB
    0x11, 0x01, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, // StripOffsets@128
    0x15, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, // SamplesPerPixel=3
    0x16, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, // RowsPerStrip=1
    0x17, 0x01, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, // StripByteCounts=6
    0x00, 0x00, 0x00, 0x00,                         // next IFD = 0
    0x08, 0x00, 0x08, 0x00, 0x08, 0x00,             // BitsPerSample = [8, 8, 8]
    0xFF, 0x00, 0x00,                               // pixel (0,0) RGB = Red
    0x00, 0xFF, 0x00,                               // pixel (0,1) RGB = Green
};
// clang-format on

// Compressed variants of the 2x1 RGB TIFF above (pixel (0,0) = Red,
// pixel (0,1) = Green). Same IFD layout as ``kTiffData`` but with a
// non-trivial Compression tag and a compressed single strip. The onnx-light
// ImageDecoder only handles uncompressed TIFF, so these are exercised through
// ``RewriteCompressedTiff`` which rewrites them into an uncompressed baseline
// TIFF before decoding.
// clang-format off

// Compression = PackBits (32773). Strip = [0x05, FF,00,00,00,FF,00].
const unsigned char kTiffPackBitsData[] = {
    0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x01,
    0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x01,
    0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x01,
    0x03, 0x00, 0x03, 0x00, 0x00, 0x00, 0x7A, 0x00, 0x00, 0x00, 0x03, 0x01,
    0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x80, 0x00, 0x00, 0x06, 0x01,
    0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x11, 0x01,
    0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x15, 0x01,
    0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x16, 0x01,
    0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x17, 0x01,
    0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x08, 0x00, 0x08, 0x00, 0x08, 0x00, 0x05, 0xFF, 0x00, 0x00,
    0x00, 0xFF, 0x00,
};

// Compression = LZW (5).
const unsigned char kTiffLzwData[] = {
    0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x01,
    0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x01,
    0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x01,
    0x03, 0x00, 0x03, 0x00, 0x00, 0x00, 0x7A, 0x00, 0x00, 0x00, 0x03, 0x01,
    0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x06, 0x01,
    0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x11, 0x01,
    0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x15, 0x01,
    0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x16, 0x01,
    0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x17, 0x01,
    0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x08, 0x00, 0x08, 0x00, 0x08, 0x00, 0x80, 0x3F, 0xC0, 0x10,
    0x38, 0x14, 0x04,
};

// Compression = Deflate/ZIP (8). Strip is a zlib-wrapped DEFLATE stream.
const unsigned char kTiffDeflateData[] = {
    0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x01,
    0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x01,
    0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x01,
    0x03, 0x00, 0x03, 0x00, 0x00, 0x00, 0x7A, 0x00, 0x00, 0x00, 0x03, 0x01,
    0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x06, 0x01,
    0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x11, 0x01,
    0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x15, 0x01,
    0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x16, 0x01,
    0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x17, 0x01,
    0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0E, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x08, 0x00, 0x08, 0x00, 0x08, 0x00, 0x78, 0xDA, 0xFB, 0xCF,
    0xC0, 0xC0, 0xF0, 0x9F, 0x01, 0x00, 0x07, 0xFE, 0x01, 0xFF,
};
// clang-format on

// Lossless 512x128 red/green quadrants, generated with Pillow (WebP and
// JPEG2000 with irreversible=False, num_resolutions=1, mct=1).
// clang-format off
const unsigned char kLargeWebpData[] = {
    0x52, 0x49, 0x46, 0x46, 0x2E, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
    0x56, 0x50, 0x38, 0x4C, 0x21, 0x00, 0x00, 0x00, 0x2F, 0xFF, 0xC1, 0x1F,
    0x00, 0x0F, 0xB0, 0xFF, 0xF3, 0x1F, 0xF3, 0x1F, 0x15, 0xC8, 0xA4, 0x6D,
    0xE6, 0xDF, 0x74, 0xEF, 0x28, 0x58, 0x44, 0xFF, 0x27, 0x20, 0x3E, 0xFE,
    0xDF, 0xF9, 0xF8, 0x7F, 0x07, 0x00,
};
const unsigned char kLargeJp2Data[] = {
    0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50, 0x20, 0x20, 0x0D, 0x0A, 0x87, 0x0A,
    0x00, 0x00, 0x00, 0x14, 0x66, 0x74, 0x79, 0x70, 0x6A, 0x70, 0x32, 0x20,
    0x00, 0x00, 0x00, 0x00, 0x6A, 0x70, 0x32, 0x20, 0x00, 0x00, 0x00, 0x2D,
    0x6A, 0x70, 0x32, 0x68, 0x00, 0x00, 0x00, 0x16, 0x69, 0x68, 0x64, 0x72,
    0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x02, 0x00, 0x00, 0x03, 0x07, 0x07,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x0F, 0x63, 0x6F, 0x6C, 0x72, 0x01, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x02, 0xB9, 0x6A, 0x70, 0x32,
    0x63, 0xFF, 0x4F, 0xFF, 0x51, 0x00, 0x2F, 0x00, 0x00, 0x00, 0x00, 0x02,
    0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x07, 0x01, 0x01, 0x07, 0x01,
    0x01, 0x07, 0x01, 0x01, 0xFF, 0x52, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x01,
    0x01, 0x00, 0x04, 0x04, 0x00, 0x01, 0xFF, 0x5C, 0x00, 0x04, 0x40, 0x40,
    0xFF, 0x64, 0x00, 0x25, 0x00, 0x01, 0x43, 0x72, 0x65, 0x61, 0x74, 0x65,
    0x64, 0x20, 0x62, 0x79, 0x20, 0x4F, 0x70, 0x65, 0x6E, 0x4A, 0x50, 0x45,
    0x47, 0x20, 0x76, 0x65, 0x72, 0x73, 0x69, 0x6F, 0x6E, 0x20, 0x32, 0x2E,
    0x35, 0x2E, 0x34, 0xFF, 0x90, 0x00, 0x0A, 0x00, 0x00, 0x00, 0x00, 0x02,
    0x41, 0x00, 0x01, 0xFF, 0x93, 0xF9, 0xFE, 0xD1, 0x9F, 0xDA, 0x33, 0xFE,
    0xD1, 0x9F, 0xDA, 0x33, 0xF0, 0x26, 0x81, 0x37, 0x02, 0x68, 0x13, 0x40,
    0x9A, 0x04, 0xD0, 0x26, 0x81, 0x37, 0xED, 0x19, 0xFD, 0xA3, 0x3F, 0xB4,
    0x67, 0xF6, 0x8C, 0x80, 0x11, 0x50, 0x54, 0xAF, 0xFF, 0x7F, 0xE9, 0x89,
    0x60, 0x15, 0x8E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
    0x84, 0x84, 0xBF, 0xFF, 0x7F, 0x11, 0x50, 0x54, 0xAF, 0xFF, 0x7F, 0xE9,
    0x89, 0x60, 0x15, 0x8E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x84, 0x84, 0xBF, 0xFF, 0x7F, 0x11, 0x50, 0x54, 0xAF, 0xFF, 0x7F,
    0xE9, 0x89, 0x60, 0x15, 0x8E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x01, 0x84, 0x84, 0xBF, 0xFF, 0x7F, 0x11, 0x50, 0x54, 0xAF, 0xFF,
    0x7F, 0xE9, 0x89, 0x60, 0x15, 0x8E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x01, 0x84, 0x84, 0xBF, 0xFF, 0x7F, 0x11, 0x50, 0x54, 0xAF,
    0xFF, 0x7F, 0x11, 0x50, 0x54, 0xAF, 0xFF, 0x7F, 0x11, 0x50, 0x54, 0xAF,
    0xFF, 0x7F, 0x11, 0x50, 0x54, 0xAF, 0xFF, 0x7F, 0x11, 0x50, 0x54, 0xAF,
    0xFF, 0x7F, 0x11, 0x50, 0x54, 0xAF, 0xFF, 0x7F, 0x11, 0x50, 0x54, 0xAF,
    0xFF, 0x7F, 0x11, 0x50, 0x54, 0xAF, 0xFF, 0x7F, 0x11, 0x50, 0x54, 0xAF,
    0xFF, 0x7F, 0xE9, 0x89, 0x60, 0x15, 0x8E, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x01, 0x84, 0x84, 0xBF, 0xFF, 0x7F, 0x11, 0x50, 0x54,
    0xAF, 0xFF, 0x7F, 0xE9, 0x89, 0x60, 0x15, 0x8E, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x84, 0x84, 0xBF, 0xFF, 0x7F, 0x11, 0x50,
    0x54, 0xAF, 0xFF, 0x7F, 0xE9, 0x89, 0x60, 0x15, 0x8E, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x84, 0x84, 0xBF, 0xFF, 0x7F, 0x11,
    0x50, 0x54, 0xAF, 0xFF, 0x7F, 0xE9, 0x89, 0x60, 0x15, 0x8E, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x84, 0x84, 0xBF, 0xFF, 0x7F,
    0xF2, 0x77, 0xFC, 0x02, 0x7F, 0x80, 0x4F, 0xFC, 0x02, 0x7F, 0x80, 0x4F,
    0xFC, 0x02, 0x7F, 0x80, 0x4F, 0xF8, 0x04, 0xFF, 0x00, 0x48, 0x00, 0x11,
    0x50, 0x54, 0xAF, 0xFF, 0x7F, 0xBE, 0x83, 0x80, 0x11, 0x50, 0x54, 0xAF,
    0xFF, 0x7F, 0xBE, 0x83, 0x80, 0x11, 0x50, 0x54, 0xAF, 0xFF, 0x7F, 0xBE,
    0x83, 0x80, 0x11, 0x50, 0x54, 0xAF, 0xFF, 0x7F, 0xBE, 0x83, 0x80, 0x11,
    0x50, 0x54, 0xAF, 0xFF, 0x7F, 0xBE, 0x83, 0x80, 0x11, 0x50, 0x54, 0xAF,
    0xFF, 0x7F, 0xBE, 0x83, 0x80, 0x11, 0x50, 0x54, 0xAF, 0xFF, 0x7F, 0xBE,
    0x83, 0x80, 0x11, 0x50, 0x54, 0xAF, 0xFF, 0x7F, 0xBE, 0x83, 0x80, 0xFB,
    0xFE, 0x01, 0x3F, 0xC0, 0x27, 0xFE, 0x01, 0x3F, 0xC0, 0x27, 0xFF, 0x40,
    0x27, 0xF8, 0x04, 0xFF, 0x60, 0x13, 0xFC, 0x02, 0x7F, 0x80, 0x4F, 0xF0,
    0x09, 0xFE, 0x01, 0x3F, 0xC0, 0x27, 0xF8, 0x04, 0xFF, 0x00, 0x4F, 0xF0,
    0x09, 0xFE, 0x01, 0x20, 0x14, 0x00, 0x5C, 0xAF, 0xFF, 0x7F, 0xBE, 0x83,
    0x80, 0x14, 0x00, 0x5C, 0xAF, 0xFF, 0x7F, 0xBE, 0x83, 0x80, 0x14, 0x00,
    0x5C, 0xAF, 0xFF, 0x7F, 0xBE, 0x83, 0x80, 0x14, 0x00, 0x5C, 0xAF, 0xFF,
    0x7F, 0xBE, 0x83, 0x80, 0x11, 0x50, 0x54, 0xAF, 0xFF, 0x7F, 0xBE, 0x83,
    0x80, 0x11, 0x50, 0x54, 0xAF, 0xFF, 0x7F, 0xBE, 0x83, 0x80, 0x11, 0x50,
    0x54, 0xAF, 0xFF, 0x7F, 0xBE, 0x83, 0x80, 0x11, 0x50, 0x54, 0xAF, 0xFF,
    0x7F, 0xBE, 0x83, 0x80, 0x11, 0x50, 0x54, 0xAF, 0xFF, 0x7F, 0xBE, 0x83,
    0x80, 0x11, 0x50, 0x54, 0xAF, 0xFF, 0x7F, 0xBE, 0x83, 0x80, 0x11, 0x50,
    0x54, 0xAF, 0xFF, 0x7F, 0xBE, 0x83, 0x80, 0x11, 0x50, 0x54, 0xAF, 0xFF,
    0x7F, 0xBE, 0x83, 0x80, 0x14, 0x00, 0x5C, 0xAF, 0xFF, 0x7F, 0xBE, 0x83,
    0x80, 0x14, 0x00, 0x5C, 0xAF, 0xFF, 0x7F, 0xBE, 0x83, 0x80, 0x14, 0x00,
    0x5C, 0xAF, 0xFF, 0x7F, 0xBE, 0x83, 0x80, 0x14, 0x00, 0x5C, 0xAF, 0xFF,
    0x7F, 0xBE, 0x83, 0x80, 0xFF, 0xD9,
};
// clang-format on

// Attempts to load the OpenJPEG runtime library the ImageDecoder uses to
// decode JPEG2000. Returns true when it is available, mirroring the kernel's
// own runtime gating so the test can assert full decoding only when the
// dependency is present.
bool OpenJpegRuntimeAvailable() {
#if defined(_WIN32)
  for (const char *name : {"libopenjp2.dll", "openjp2.dll"}) {
    HMODULE handle = LoadLibraryA(name);
    if (handle != nullptr) {
      FreeLibrary(handle);
      return true;
    }
  }
  return false;
#else
  for (const char *name : {"libopenjp2.so.7", "libopenjp2.so"}) {
    void *handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
    if (handle != nullptr) {
      dlclose(handle);
      return true;
    }
  }
  return false;
#endif
}

// Attempts to load the libwebp runtime library the ImageDecoder uses to decode
// WebP. Returns true when it is available, mirroring the kernel's own runtime
// gating so the test can assert full decoding only when the dependency is
// present.
bool LibWebpRuntimeAvailable() {
#if defined(_WIN32)
  for (const char *name : {"libwebp.dll", "webp.dll"}) {
    HMODULE handle = LoadLibraryA(name);
    if (handle != nullptr) {
      FreeLibrary(handle);
      return true;
    }
  }
  return false;
#else
  for (const char *name : {"libwebp.so.7", "libwebp.so"}) {
    void *handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
    if (handle != nullptr) {
      dlclose(handle);
      return true;
    }
  }
  return false;
#endif
}

Tensor MakeEncodedTensor(const unsigned char *data, size_t size) {
#if ONNX_LIGHT_KERNEL_IMAGES_HAS_CURRENT_BACKEND_TEST_API
  return Tensor::FromUint8("", {static_cast<int64_t>(size)},
                           std::vector<uint8_t>(data, data + size));
#else
  Tensor t;
  t.data_type = static_cast<int32_t>(DataType::UINT8);
  t.shape = {static_cast<int64_t>(size)};
  t.data.assign(data, data + size);
  return t;
#endif
}

// Three strips in reverse storage order, with a partial final strip.
std::vector<uint8_t> MakeStripTiff(uint32_t width, bool grayscale) {
  const uint32_t height = 5;
  const uint32_t samples = grayscale ? 1 : 3;
  const uint32_t row_bytes = width * samples;
  const uint32_t pixel_offset = 152;
  std::vector<uint8_t> data(kTiffData, kTiffData + 128);
  data.resize(pixel_offset + row_bytes * height);
  const auto put32 = [&](size_t offset, uint32_t value) {
    for (size_t byte = 0; byte < 4; ++byte) {
      data[offset + byte] = static_cast<uint8_t>(value >> (byte * 8));
    }
  };
  data[12] = 4; // ImageWidth uses LONG to include widths above 65535.
  put32(18, width);
  put32(30, height);
  if (grayscale) {
    put32(38, 1); // BitsPerSample count
    put32(42, 8);
    put32(66, 1); // BlackIsZero
    put32(90, 1); // SamplesPerPixel
  }
  put32(74, 3); // StripOffsets count
  put32(78, 128);
  put32(102, 2); // RowsPerStrip
  put32(110, 3); // StripByteCounts count
  put32(114, 140);
  for (uint32_t strip = 0; strip < 3; ++strip) {
    const uint32_t rows = strip == 2 ? 1 : 2;
    const uint32_t offset = pixel_offset + (strip == 0 ? 3 : strip == 1 ? 1 : 0) * row_bytes;
    put32(128 + strip * 4, offset);
    put32(140 + strip * 4, rows * row_bytes);
    for (uint32_t row = 0; row < rows; ++row) {
      for (uint32_t column = 0; column < width; ++column) {
        const size_t pixel = static_cast<size_t>(strip * 2 + row) * width + column;
        for (uint32_t channel = 0; channel < samples; ++channel) {
          data[offset + row * row_bytes + column * samples + channel] =
              static_cast<uint8_t>(pixel * 17 + channel * 73);
        }
      }
    }
  }
  return data;
}

template <typename Runtime = core::runtime::RuntimeContext>
Tensor DecodeWithRegisteredKernel(const unsigned char *data, size_t size,
                                  const std::string &pixel_format) {
  const auto opset = core::runtime::DefaultOpset(20);
  const KernelContext ctx{opset};
  Runtime rt(ctx);
  rt.tensors()["encoded"] = MakeEncodedTensor(data, size);

  NodeProto node;
  node.set_op_type("ImageDecoder");
  node.add_input("encoded");
  node.add_output("image");
  AttributeProto *attribute = node.add_attribute();
  attribute->set_name("pixel_format");
  attribute->set_type(AttributeProto::AttributeType::STRING);
  attribute->set_s(pixel_format);

  core::runtime::RunNode(node, rt);
  return rt.tensors().at("image");
}

#if ONNX_LIGHT_KERNEL_IMAGES_HAS_CURRENT_BACKEND_TEST_API
template <typename Runtime = core::runtime::RuntimeContext>
std::vector<TestCase> CollectLocalImageDecoderCases(core::backend_test::TestMode mode) {
  struct Image {
    const char *name;
    const unsigned char *data;
    size_t size;
  };
  const Image images[] = {
      {"bmp", kBmpData, sizeof(kBmpData)},
      {"pnm", kPnmData, sizeof(kPnmData)},
      {"jpeg2000", kJp2Data, sizeof(kJp2Data)},
      {"webp", kWebpData, sizeof(kWebpData)},
      {"tiff", kTiffData, sizeof(kTiffData)},
      {"tiff_packbits", kTiffPackBitsData, sizeof(kTiffPackBitsData)},
      {"tiff_lzw", kTiffLzwData, sizeof(kTiffLzwData)},
      {"tiff_deflate", kTiffDeflateData, sizeof(kTiffDeflateData)},
  };
  std::vector<TestCase> cases;
  for (const Image &image : images) {
    for (const char *pixel_format : {"RGB", "BGR", "Grayscale"}) {
      NodeProto node;
      node.set_op_type("ImageDecoder");
      node.add_input("encoded");
      node.add_output("image");
      AttributeProto *attribute = node.add_attribute();
      attribute->set_name("pixel_format");
      attribute->set_type(AttributeProto::AttributeType::STRING);
      attribute->set_s(pixel_format);
      std::string name = std::string("test_local_image_decoder_") + image.name + "_" + pixel_format;
      if (mode == core::backend_test::TestMode::BENCHMARK) {
        name += "_benchmark";
      }
      auto make_io = [image, pixel_format]() -> core::backend_test::IoData {
        Tensor input = MakeEncodedTensor(image.data, image.size);
        Tensor output = DecodeWithRegisteredKernel<Runtime>(image.data, image.size, pixel_format);
        return {{std::move(input)}, {std::move(output)}};
      };
      // Builders retain only static image metadata, never a kernel or runtime context.
      static_assert(std::is_trivially_copyable_v<decltype(make_io)>);
      core::backend_test::Expect(cases, std::move(node), std::move(name),
                                 {core::runtime::DefaultOpset(20)}, std::move(make_io));
    }
  }
  return cases;
}

// Exercise real kernel resolution while tracking the lifetime of its owning runtime.
class TrackedImageRuntime : public core::runtime::RuntimeContext {
public:
  explicit TrackedImageRuntime(const KernelContext &ctx) : RuntimeContext(ctx) {
    ++constructed;
    ++alive;
  }
  ~TrackedImageRuntime() { --alive; }
  TrackedImageRuntime(const TrackedImageRuntime &) = delete;
  TrackedImageRuntime &operator=(const TrackedImageRuntime &) = delete;

  static inline size_t constructed = 0;
  static inline size_t alive = 0;
};
#endif

} // namespace

class ImageDecoderTest : public ::testing::Test {
protected:
  void SetUp() override { onnx_light_kernel_images::RegisterImageKernels(); }

  void CheckLargeLegacyConversion(const unsigned char *data, size_t size, const char *label) {
    core::runtime::CpuExecutorRegistry registry(2);
    for (const std::string format : {"RGB", "BGR", "Grayscale"}) {
      const size_t channels = ImageDecoder::ChannelCount(format);
      std::vector<uint8_t> expected(512 * 128 * channels);
      for (size_t row = 0; row < 128; ++row) {
        for (size_t column = 0; column < 512; ++column) {
          const bool red = (column < 256) == (row < 64);
          const size_t offset = (row * 512 + column) * channels;
          if (channels == 1) {
            expected[offset] = red ? 76 : 150;
          } else {
            expected[offset] = format == "RGB" && red ? 255 : 0;
            expected[offset + 1] = red ? 0 : 255;
            expected[offset + 2] = format == "BGR" && red ? 255 : 0;
          }
        }
      }
      std::vector<uint8_t> serial;
      for (int32_t threads : {1, 2}) {
        SCOPED_TRACE(::testing::Message() << label << " " << format << " " << threads);
        core::runtime::CpuExecutionPolicy policy;
        policy.num_threads = threads;
        policy.spin_policy = core::runtime::CpuSpinPolicy::kParkImmediately;
        auto executor = registry.Acquire(policy);
        core::runtime::CpuExecutorScope executor_scope(executor.get());
        core::runtime::ParallelRegionCollector collector(4);
        core::runtime::ParallelRegionCollectorScope collector_scope(&collector);
        const Tensor result = DecodeWithRegisteredKernel(data, size, format);
        ASSERT_EQ(result.shape, (std::vector<int64_t>{128, 512, static_cast<int64_t>(channels)}));
        const std::vector<uint8_t> pixels(result.bytes(), result.bytes() + result.size_bytes());
        EXPECT_EQ(pixels, expected);
        if (threads == 1) {
          serial = pixels;
        } else {
          EXPECT_EQ(pixels, serial);
        }
        const auto events = collector.events();
        ASSERT_EQ(events.size(), 1u);
        EXPECT_EQ(events[0].label, label);
        EXPECT_EQ(events[0].executor_instance_id, executor->instance_id());
        EXPECT_EQ(events[0].total_iterations, 512 * 128);
        EXPECT_EQ(events[0].observed_threads, threads);
      }
    }
  }
};

TEST_F(ImageDecoderTest, RegistrationIsIdempotent) {
  // Calling twice must not crash.
  onnx_light_kernel_images::RegisterImageKernels();
  onnx_light_kernel_images::RegisterImageKernels();
}

TEST_F(ImageDecoderTest, KernelRegisteredInDispatchTable) {
  const auto &table = KernelDispatchTable();
  EXPECT_NE(table.find("ai.onnx:ImageDecoder"), table.end());
}

TEST(ImageKernelRegistrationTest, ContextRegistrationIsLocalAndIdempotent) {
  core::runtime::RuntimeContext local(KernelContext(core::runtime::DefaultOpset(20)));
  core::runtime::RuntimeContext other(KernelContext(core::runtime::DefaultOpset(20)));
  const auto before = KernelDispatchTable();

  onnx_light_kernel_images::RegisterImageKernels(local);
  onnx_light_kernel_images::RegisterImageKernels(local);
  onnx_light_kernel_images::RegisterImageKernels(other);

  ASSERT_EQ(local.custom_kernels().size(), 1u);
  EXPECT_EQ(local.custom_kernels().count("ai.onnx:ImageDecoder"), 1u);
  EXPECT_EQ(other.custom_kernels().count("ai.onnx:ImageDecoder"), 1u);
  local.ClearCustomKernels();
  EXPECT_EQ(other.custom_kernels().count("ai.onnx:ImageDecoder"), 1u);
  ASSERT_EQ(KernelDispatchTable().size(), before.size());
  for (const auto &[key, factory] : before) {
    ASSERT_NE(KernelDispatchTable().find(key), KernelDispatchTable().end());
    EXPECT_EQ(KernelDispatchTable().at(key).target_type(), factory.target_type());
  }

  other.tensors()["encoded"] = MakeEncodedTensor(kTiffPackBitsData, sizeof(kTiffPackBitsData));
  NodeProto node;
  node.set_op_type("ImageDecoder");
  node.add_input("encoded");
  node.add_output("image");
  core::runtime::RunNode(node, other);
  const Tensor &image = other.tensors().at("image");
  ASSERT_EQ(image.shape, (std::vector<int64_t>{1, 2, 3}));
  EXPECT_EQ((std::vector<uint8_t>(image.bytes(), image.bytes() + image.size_bytes())),
            (std::vector<uint8_t>{255, 0, 0, 0, 255, 0}));
}

TEST_F(ImageDecoderTest, TiffConversionUsesOnnxLightExecutor) {
  core::runtime::CpuExecutorRegistry registry(2);
  for (bool grayscale : {false, true}) {
    for (uint32_t width : {2u, 65536u, 65537u}) {
      const auto encoded = MakeStripTiff(width, grayscale);
      for (const char *format : {"RGB", "BGR", "Grayscale"}) {
        const size_t channels = ImageDecoder::ChannelCount(format);
        const bool is_rgb = std::string(format) == "RGB";
        std::vector<uint8_t> expected(width * 5 * channels);
        for (size_t i = 0; i < width * 5; ++i) {
          const uint8_t red = static_cast<uint8_t>(i * 17);
          const uint8_t green = grayscale ? red : static_cast<uint8_t>(i * 17 + 73);
          const uint8_t blue = grayscale ? red : static_cast<uint8_t>(i * 17 + 146);
          if (channels == 1) {
            expected[i] = static_cast<uint8_t>((299 * red + 587 * green + 114 * blue + 500) / 1000);
          } else {
            expected[i * 3] = is_rgb ? red : blue;
            expected[i * 3 + 1] = green;
            expected[i * 3 + 2] = is_rgb ? blue : red;
          }
        }
        for (int32_t threads : {1, 2}) {
          SCOPED_TRACE(::testing::Message()
                       << width << " " << grayscale << " " << format << " " << threads);
          core::runtime::CpuExecutionPolicy policy;
          policy.num_threads = threads;
          policy.spin_policy = core::runtime::CpuSpinPolicy::kParkImmediately;
          auto executor = registry.Acquire(policy);
          core::runtime::CpuExecutorScope executor_scope(executor.get());
          core::runtime::ParallelRegionCollector collector(4);
          core::runtime::ParallelRegionCollectorScope collector_scope(&collector);
          const Tensor result = DecodeWithRegisteredKernel(encoded.data(), encoded.size(), format);
          ASSERT_EQ(result.shape, (std::vector<int64_t>{5, width, static_cast<int64_t>(channels)}));
          EXPECT_EQ(std::vector<uint8_t>(result.bytes(), result.bytes() + result.size_bytes()),
                    expected);
          const auto events = collector.events();
          ASSERT_EQ(events.size(), 1u);
          EXPECT_EQ(events[0].label, "ImageDecoder/TIFF");
          EXPECT_EQ(events[0].executor_instance_id, executor->instance_id());
          EXPECT_EQ(events[0].total_iterations, width * 5);
          EXPECT_EQ(events[0].observed_threads, width == 2 ? 1 : threads);
        }
      }
    }
  }
}

TEST_F(ImageDecoderTest, LargeWebpConversionUsesWorkerThreads) {
  if (!LibWebpRuntimeAvailable()) {
    GTEST_SKIP() << "Optional WebP runtime is unavailable.";
  }
  CheckLargeLegacyConversion(kLargeWebpData, sizeof(kLargeWebpData), "ImageDecoder/WebP");
}

TEST_F(ImageDecoderTest, LargeJpeg2000ConversionUsesWorkerThreads) {
  if (!OpenJpegRuntimeAvailable()) {
    GTEST_SKIP() << "Optional OpenJPEG runtime is unavailable.";
  }
  CheckLargeLegacyConversion(kLargeJp2Data, sizeof(kLargeJp2Data), "ImageDecoder/JPEG2000");
}

TEST_F(ImageDecoderTest, SmallLegacyConversionsUseOnnxLightExecutorInline) {
  core::runtime::CpuExecutorRegistry registry(1);
  core::runtime::CpuExecutionPolicy policy;
  policy.num_threads = 2;
  policy.spin_policy = core::runtime::CpuSpinPolicy::kParkImmediately;
  auto executor = registry.Acquire(policy);
  ASSERT_EQ(executor->policy().effective_threads, 2);
  core::runtime::CpuExecutorScope executor_scope(executor.get());
  const bool webp_available = LibWebpRuntimeAvailable();
  const bool openjpeg_available = OpenJpegRuntimeAvailable();
  for (bool webp : {true, false}) {
    if (webp ? !webp_available : !openjpeg_available) {
      continue;
    }
    for (const char *format : {"RGB", "BGR", "Grayscale"}) {
      core::runtime::ParallelRegionCollector collector(4);
      core::runtime::ParallelRegionCollectorScope collector_scope(&collector);
      const Tensor result = DecodeWithRegisteredKernel(
          webp ? kWebpData : kJp2Data, webp ? sizeof(kWebpData) : sizeof(kJp2Data), format);
      ASSERT_EQ(result.shape, (std::vector<int64_t>{1, 2, ImageDecoder::ChannelCount(format)}));
      const std::vector<uint8_t> expected =
          std::string(format) == "RGB"   ? std::vector<uint8_t>{255, 0, 0, 0, 255, 0}
          : std::string(format) == "BGR" ? std::vector<uint8_t>{0, 0, 255, 0, 255, 0}
                                         : std::vector<uint8_t>{76, 150};
      EXPECT_EQ(std::vector<uint8_t>(result.bytes(), result.bytes() + result.size_bytes()),
                expected);
      const auto events = collector.events();
      ASSERT_EQ(events.size(), 1u);
      EXPECT_EQ(events[0].label, webp ? "ImageDecoder/WebP" : "ImageDecoder/JPEG2000");
      EXPECT_EQ(events[0].executor_instance_id, executor->instance_id());
      EXPECT_EQ(events[0].total_iterations, 2);
      EXPECT_EQ(events[0].observed_threads, 1);
    }
  }
}

TEST_F(ImageDecoderTest, DecodeBmpRgb) {
  KernelContext ctx(core::runtime::DefaultOpset(20));
  ImageDecoder decoder(ctx);
  Tensor encoded = MakeEncodedTensor(kBmpData, sizeof(kBmpData));
  Tensor result = decoder(encoded, "RGB");

  ASSERT_EQ(result.shape.size(), 3u);
  EXPECT_EQ(result.shape[0], 2); // height
  EXPECT_EQ(result.shape[1], 2); // width
  EXPECT_EQ(result.shape[2], 3); // channels

  const uint8_t *p = result.bytes();
  // Row 0 (display top) = BMP row 1: Red, Green
  EXPECT_EQ(p[0], 255); // R
  EXPECT_EQ(p[1], 0);   // G
  EXPECT_EQ(p[2], 0);   // B

  EXPECT_EQ(p[3], 0);   // R
  EXPECT_EQ(p[4], 255); // G
  EXPECT_EQ(p[5], 0);   // B

  // Row 1 (display bottom) = BMP row 0: Blue, White
  EXPECT_EQ(p[6], 0);   // R
  EXPECT_EQ(p[7], 0);   // G
  EXPECT_EQ(p[8], 255); // B

  EXPECT_EQ(p[9], 255);  // R
  EXPECT_EQ(p[10], 255); // G
  EXPECT_EQ(p[11], 255); // B
}

TEST_F(ImageDecoderTest, DecodeBmpGrayscale) {
  KernelContext ctx(core::runtime::DefaultOpset(20));
  ImageDecoder decoder(ctx);
  Tensor encoded = MakeEncodedTensor(kBmpData, sizeof(kBmpData));
  Tensor result = decoder(encoded, "Grayscale");

  ASSERT_EQ(result.shape.size(), 3u);
  EXPECT_EQ(result.shape[0], 2);
  EXPECT_EQ(result.shape[1], 2);
  EXPECT_EQ(result.shape[2], 1);
}

TEST_F(ImageDecoderTest, DecodeBmpBgr) {
  KernelContext ctx(core::runtime::DefaultOpset(20));
  ImageDecoder decoder(ctx);
  Tensor encoded = MakeEncodedTensor(kBmpData, sizeof(kBmpData));
  Tensor result = decoder(encoded, "BGR");

  ASSERT_EQ(result.shape.size(), 3u);
  EXPECT_EQ(result.shape[2], 3);

  const uint8_t *p = result.bytes();
  // Row 0 pixel 0 is Red in RGB => BGR = (0, 0, 255)
  EXPECT_EQ(p[0], 0);   // B
  EXPECT_EQ(p[1], 0);   // G
  EXPECT_EQ(p[2], 255); // R
}

TEST_F(ImageDecoderTest, DecodePnmRgb) {
  KernelContext ctx(core::runtime::DefaultOpset(20));
  ImageDecoder decoder(ctx);
  Tensor encoded = MakeEncodedTensor(kPnmData, sizeof(kPnmData));
  Tensor result = decoder(encoded, "RGB");

  ASSERT_EQ(result.shape.size(), 3u);
  EXPECT_EQ(result.shape[0], 1); // height
  EXPECT_EQ(result.shape[1], 2); // width
  EXPECT_EQ(result.shape[2], 3); // channels

  const uint8_t *p = result.bytes();
  // Pixel 0: Red
  EXPECT_EQ(p[0], 255);
  EXPECT_EQ(p[1], 0);
  EXPECT_EQ(p[2], 0);
  // Pixel 1: Green
  EXPECT_EQ(p[3], 0);
  EXPECT_EQ(p[4], 255);
  EXPECT_EQ(p[5], 0);
}

TEST_F(ImageDecoderTest, DecodeJpeg2000Rgb) {
  Tensor result = DecodeWithRegisteredKernel(kJp2Data, sizeof(kJp2Data), "RGB");

  ASSERT_EQ(result.shape.size(), 3u);
  EXPECT_EQ(result.shape[2], 3); // channels

  if (OpenJpegRuntimeAvailable()) {
    // The kernel decodes JPEG2000 through libopenjp2 when it is present.
    EXPECT_EQ(result.shape[0], 1); // height
    EXPECT_EQ(result.shape[1], 2); // width

    const uint8_t *p = result.bytes();
    // Pixel 0: Red
    EXPECT_EQ(p[0], 255);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 0);
    // Pixel 1: Green
    EXPECT_EQ(p[3], 0);
    EXPECT_EQ(p[4], 255);
    EXPECT_EQ(p[5], 0);
  } else {
    // Without the runtime library the kernel falls back to an empty matrix.
    EXPECT_EQ(result.shape[0], 0);
    EXPECT_EQ(result.shape[1], 0);
  }
}

TEST_F(ImageDecoderTest, DecodeWebpRgb) {
  Tensor result = DecodeWithRegisteredKernel(kWebpData, sizeof(kWebpData), "RGB");

  ASSERT_EQ(result.shape.size(), 3u);
  EXPECT_EQ(result.shape[2], 3); // channels

  if (LibWebpRuntimeAvailable()) {
    // The kernel decodes WebP through libwebp when it is present.
    EXPECT_EQ(result.shape[0], 1); // height
    EXPECT_EQ(result.shape[1], 2); // width

    const uint8_t *p = result.bytes();
    // Pixel 0: Red
    EXPECT_EQ(p[0], 255);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 0);
    // Pixel 1: Green
    EXPECT_EQ(p[3], 0);
    EXPECT_EQ(p[4], 255);
    EXPECT_EQ(p[5], 0);
  } else {
    // Without the runtime library the kernel falls back to an empty matrix.
    EXPECT_EQ(result.shape[0], 0);
    EXPECT_EQ(result.shape[1], 0);
  }
}

TEST_F(ImageDecoderTest, DecodeWebpBgr) {
  Tensor result = DecodeWithRegisteredKernel(kWebpData, sizeof(kWebpData), "BGR");

  ASSERT_EQ(result.shape.size(), 3u);
  EXPECT_EQ(result.shape[2], 3); // channels

  if (LibWebpRuntimeAvailable()) {
    EXPECT_EQ(result.shape[0], 1); // height
    EXPECT_EQ(result.shape[1], 2); // width

    const uint8_t *p = result.bytes();
    // Pixel 0 is Red in RGB => BGR = (0, 0, 255)
    EXPECT_EQ(p[0], 0);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 255);
  } else {
    EXPECT_EQ(result.shape[0], 0);
    EXPECT_EQ(result.shape[1], 0);
  }
}

TEST_F(ImageDecoderTest, DecodeTiffRgb) {
  Tensor result = DecodeWithRegisteredKernel(kTiffData, sizeof(kTiffData), "RGB");

  ASSERT_EQ(result.shape.size(), 3u);
  EXPECT_EQ(result.shape[0], 1); // height
  EXPECT_EQ(result.shape[1], 2); // width
  EXPECT_EQ(result.shape[2], 3); // channels

  const uint8_t *p = result.bytes();
  // Pixel 0: Red
  EXPECT_EQ(p[0], 255);
  EXPECT_EQ(p[1], 0);
  EXPECT_EQ(p[2], 0);
  // Pixel 1: Green
  EXPECT_EQ(p[3], 0);
  EXPECT_EQ(p[4], 255);
  EXPECT_EQ(p[5], 0);
}

TEST_F(ImageDecoderTest, DecodeTiffBgr) {
  Tensor result = DecodeWithRegisteredKernel(kTiffData, sizeof(kTiffData), "BGR");

  ASSERT_EQ(result.shape.size(), 3u);
  EXPECT_EQ(result.shape[0], 1);
  EXPECT_EQ(result.shape[1], 2);
  EXPECT_EQ(result.shape[2], 3);

  const uint8_t *p = result.bytes();
  // Pixel 0 is Red in RGB => BGR = (0, 0, 255)
  EXPECT_EQ(p[0], 0);   // B
  EXPECT_EQ(p[1], 0);   // G
  EXPECT_EQ(p[2], 255); // R
}

TEST_F(ImageDecoderTest, InvalidInputThrows) {
  KernelContext ctx(core::runtime::DefaultOpset(20));
  ImageDecoder decoder(ctx);

  // Empty input
  Tensor empty;
  empty.data_type = static_cast<int32_t>(DataType::UINT8);
  empty.shape = {0};
  Tensor result = decoder(empty, "RGB");
  // Falls back to empty matrix (0, 0, 3)
  EXPECT_EQ(result.shape[0], 0);
  EXPECT_EQ(result.shape[1], 0);
  EXPECT_EQ(result.shape[2], 3);
}

TEST_F(ImageDecoderTest, UnrecognizedFormatFallsBackToEmptyMatrix) {
  KernelContext ctx(core::runtime::DefaultOpset(20));
  ImageDecoder decoder(ctx);

  // Random garbage that is not a valid image.
  const unsigned char garbage[] = {0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04};
  Tensor encoded = MakeEncodedTensor(garbage, sizeof(garbage));
  Tensor result = decoder(encoded, "RGB");

  EXPECT_EQ(result.shape[0], 0);
  EXPECT_EQ(result.shape[1], 0);
  EXPECT_EQ(result.shape[2], 3);
}

// ---------------------------------------------------------------------------
// TIFF compression support (PackBits / LZW / Deflate).
//
// The onnx-light ImageDecoder only decodes uncompressed TIFF, so compressed
// inputs are rewritten into an uncompressed baseline TIFF by the registered
// ``TiffAwareImageDecoder`` and then decoded by this package's TIFF decoder.
// ---------------------------------------------------------------------------
namespace {

// Rewrites a compressed TIFF and decodes it as RGB. Fails the calling test
// when the rewrite does not apply.
Tensor DecodeCompressedTiff(const unsigned char *data, size_t size,
                            const std::string &pixel_format) {
  return DecodeWithRegisteredKernel(data, size, pixel_format);
}

void ExpectRedGreenRgb(const Tensor &result) {
  ASSERT_EQ(result.shape.size(), 3u);
  EXPECT_EQ(result.shape[0], 1); // height
  EXPECT_EQ(result.shape[1], 2); // width
  EXPECT_EQ(result.shape[2], 3); // channels
  const uint8_t *p = result.bytes();
  EXPECT_EQ(p[0], 255); // pixel 0: Red
  EXPECT_EQ(p[1], 0);
  EXPECT_EQ(p[2], 0);
  EXPECT_EQ(p[3], 0); // pixel 1: Green
  EXPECT_EQ(p[4], 255);
  EXPECT_EQ(p[5], 0);
}

} // namespace

TEST_F(ImageDecoderTest, DecodeTiffPackBits) {
  ExpectRedGreenRgb(DecodeCompressedTiff(kTiffPackBitsData, sizeof(kTiffPackBitsData), "RGB"));
}

TEST_F(ImageDecoderTest, DecodeTiffLzw) {
  ExpectRedGreenRgb(DecodeCompressedTiff(kTiffLzwData, sizeof(kTiffLzwData), "RGB"));
}

TEST_F(ImageDecoderTest, DecodeTiffDeflate) {
  ExpectRedGreenRgb(DecodeCompressedTiff(kTiffDeflateData, sizeof(kTiffDeflateData), "RGB"));
}

TEST_F(ImageDecoderTest, DecodeTiffPackBitsBgr) {
  Tensor result = DecodeCompressedTiff(kTiffPackBitsData, sizeof(kTiffPackBitsData), "BGR");
  ASSERT_EQ(result.shape.size(), 3u);
  const uint8_t *p = result.bytes();
  // Pixel 0 is Red in RGB => BGR = (0, 0, 255).
  EXPECT_EQ(p[0], 0);
  EXPECT_EQ(p[1], 0);
  EXPECT_EQ(p[2], 255);
}

TEST_F(ImageDecoderTest, RewriteCompressedTiffRejectsUncompressed) {
  // Uncompressed TIFF is left for the SDK to decode directly.
  std::vector<uint8_t> rewritten;
  EXPECT_FALSE(
      onnx_light_kernel_images::RewriteCompressedTiff(kTiffData, sizeof(kTiffData), rewritten));

  // Non-TIFF inputs (BMP) are rejected too.
  EXPECT_FALSE(
      onnx_light_kernel_images::RewriteCompressedTiff(kBmpData, sizeof(kBmpData), rewritten));
}

TEST_F(ImageDecoderTest, MalformedTiffDimensionsAreRejectedBeforeAllocation) {
  std::vector<uint8_t> malformed(kTiffData, kTiffData + sizeof(kTiffData));
  std::fill(malformed.begin() + 18, malformed.begin() + 22, 0xFF);
  std::fill(malformed.begin() + 30, malformed.begin() + 34, 0xFF);

  int64_t height = 0;
  int64_t width = 0;
  std::vector<uint8_t> pixels;
  EXPECT_FALSE(onnx_light_kernel_images::TryDecodeLegacyCodecImage(
      malformed.data(), malformed.size(), "RGB", height, width, pixels, nullptr));
  EXPECT_TRUE(pixels.empty());
}

// ---------------------------------------------------------------------------
// End-to-end run of the registered ``TiffAwareImageDecoder`` kernel through
// ``RunNode`` with a runtime allocator attached to the ``RuntimeContext``.
//
// This exercises the ``TiffAwareImageDecoder::Run`` path (the compressed-TIFF
// rewrite injected before the base decoder), which acquires its rewritten
// byte-stream tensor from ``rt.allocator()`` via ``MakeOutputTensor`` instead
// of an inline ``std::vector``. The decoded output must still be correct and,
// because an allocator is attached, allocator-backed.
// ---------------------------------------------------------------------------
TEST_F(ImageDecoderTest, RunNodeDecodesCompressedTiffThroughAllocator) {
  core::runtime::SimpleRawBufferAllocator pool(16);
  core::runtime::RuntimeContext rt(KernelContext(core::runtime::DefaultOpset(20)),
                                   core::runtime::RuntimeContextOptions{.allocator = &pool});

  rt.tensors()["encoded"] = MakeEncodedTensor(kTiffPackBitsData, sizeof(kTiffPackBitsData));

  NodeProto node;
  node.set_op_type("ImageDecoder");
  node.add_input("encoded");
  node.add_output("image");
  AttributeProto *pixel_format = node.add_attribute();
  pixel_format->set_name("pixel_format");
  pixel_format->set_type(AttributeProto::AttributeType::STRING);
  pixel_format->set_s("RGB");

  core::runtime::RunNode(node, rt);

  ASSERT_NE(rt.tensors().find("image"), rt.tensors().end());
  const Tensor &result = rt.tensors()["image"];
  ExpectRedGreenRgb(result);
  // With an allocator attached, the decoded output is acquired from the pool.
  EXPECT_TRUE(result.has_allocation());
  EXPECT_GT(pool.PeakAllocatedSize(), 0u);
}

#if ONNX_LIGHT_KERNEL_IMAGES_HAS_CURRENT_BACKEND_TEST_API
TEST_F(ImageDecoderTest, BackendCaseReferenceRuntimeIsScopedToMaterialization) {
  for (auto mode : {core::backend_test::TestMode::TEST, core::backend_test::TestMode::BENCHMARK}) {
    TrackedImageRuntime::constructed = 0;
    auto cases = CollectLocalImageDecoderCases<TrackedImageRuntime>(mode);
    ASSERT_EQ(cases.size(), 24u);
    EXPECT_EQ(TrackedImageRuntime::constructed, 0u);
    EXPECT_EQ(TrackedImageRuntime::alive, 0u);
    size_t built = 0;
    for (TestCase &test_case : cases) {
      ASSERT_FALSE(test_case.materialized()) << test_case.name;
      ASSERT_TRUE(test_case.build) << test_case.name;
      // Invoke the persistent builder twice without unloading it.
      for (int repeat = 0; repeat < 2; ++repeat) {
        auto data = [](auto &build) {
          if constexpr (std::is_invocable_v<decltype(build), bool>) {
            return build(true);
          } else {
            return build();
          }
        }(test_case.build);
        EXPECT_EQ(TrackedImageRuntime::constructed, ++built) << test_case.name;
        EXPECT_EQ(TrackedImageRuntime::alive, 0u) << test_case.name;
        ASSERT_EQ(data.data_sets.size(), 1u);
        EXPECT_EQ(data.data_sets[0].inputs.size(), 1u);
        EXPECT_EQ(data.data_sets[0].outputs.size(), 1u);
      }
      test_case.Materialize();
      EXPECT_EQ(TrackedImageRuntime::constructed, ++built) << test_case.name;
      EXPECT_EQ(TrackedImageRuntime::alive, 0u) << test_case.name;
      ASSERT_EQ(test_case.data_sets().size(), 1u);
      EXPECT_EQ(TrackedImageRuntime::constructed, built) << test_case.name;
    }
  }
}

TEST_F(ImageDecoderTest, LocalBackendCasesRunThroughRegisteredKernel) {
  for (auto mode : {core::backend_test::TestMode::TEST, core::backend_test::TestMode::BENCHMARK}) {
    auto cases = CollectLocalImageDecoderCases(mode);
    ASSERT_EQ(cases.size(), 24u);
    for (TestCase &test_case : cases) {
      TestCaseUnloadGuard unload_guard(test_case);
      const ModelProto &model = test_case.model();
      for (const DataSet &data_set : test_case.data_sets()) {
        Tensors outputs = RunModel(model, data_set.inputs);
        ASSERT_EQ(outputs.size(), data_set.outputs.size()) << test_case.name;
        for (size_t i = 0; i < outputs.size(); ++i) {
          const auto comparison =
              CompareTensors(outputs[i], data_set.outputs[i], test_case.rtol, test_case.atol);
          EXPECT_TRUE(comparison.close) << test_case.name << ": " << comparison.message;
        }
      }
    }
  }
}

TEST_F(ImageDecoderTest, BackendCasesRunThroughRegisteredKernel) {
  std::vector<TestCase> cases =
      CollectTestCases("ImageDecoder", /*include_big=*/false, core::backend_test::TestMode::TEST);
  ASSERT_FALSE(cases.empty());

  size_t executed = 0;
  for (TestCase &test_case : cases) {
    TestCaseUnloadGuard unload_guard(test_case);
    ASSERT_FALSE(test_case.materialized()) << test_case.name;
    EXPECT_EQ(test_case.kind, TestCaseKind::NODE) << test_case.name;
    EXPECT_EQ(test_case.tag, TestCaseTag::NONE) << test_case.name;
    if (test_case.name.find("_jpeg_") != std::string::npos) {
      continue;
    }

    const ModelProto &model = test_case.model();
    for (const DataSet &data_set : test_case.data_sets()) {
      ASSERT_TRUE(data_set.expected_outputs_generated) << test_case.name;
      Tensors inputs = data_set.inputs;
      Tensors outputs = RunModel(model, std::move(inputs));
      ASSERT_EQ(outputs.size(), data_set.outputs.size()) << test_case.name;
      for (size_t i = 0; i < outputs.size(); ++i) {
        const TensorComparison comparison =
            CompareTensors(outputs[i], data_set.outputs[i], test_case.rtol, test_case.atol);
        EXPECT_TRUE(comparison.close) << test_case.name << ": " << comparison.message;
      }
      ++executed;
    }
  }
  EXPECT_GT(executed, 0u);
}
#endif
