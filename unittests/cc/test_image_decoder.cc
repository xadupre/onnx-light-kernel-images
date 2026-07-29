// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_kernel_images/register_image_kernels.h"

#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/kernel_dispatch_table.h"
#include "onnx_core/runtime/simple_tensor.h"
#include "onnx_extensions/kernels/kernels/image/include_image_kernels.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace onnx_light;
using core::runtime::DataType;
using core::runtime::KernelDispatchTable;
using core::runtime::Tensor;
using onnx_kernels::kernel::ImageDecoder;
using onnx_kernels::kernel::KernelContext;

namespace {

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

Tensor MakeEncodedTensor(const unsigned char *data, size_t size) {
  Tensor t;
  t.data_type = static_cast<int32_t>(DataType::UINT8);
  t.shape = {static_cast<int64_t>(size)};
  t.data.assign(data, data + size);
  return t;
}

} // namespace

class ImageDecoderTest : public ::testing::Test {
protected:
  void SetUp() override { onnx_light_kernel_images::RegisterImageKernels(); }
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

TEST_F(ImageDecoderTest, DecodeTiffRgb) {
  KernelContext ctx(core::runtime::DefaultOpset(20));
  ImageDecoder decoder(ctx);
  Tensor encoded = MakeEncodedTensor(kTiffData, sizeof(kTiffData));
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

TEST_F(ImageDecoderTest, DecodeTiffBgr) {
  KernelContext ctx(core::runtime::DefaultOpset(20));
  ImageDecoder decoder(ctx);
  Tensor encoded = MakeEncodedTensor(kTiffData, sizeof(kTiffData));
  Tensor result = decoder(encoded, "BGR");

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
