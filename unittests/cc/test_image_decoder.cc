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

TEST_F(ImageDecoderTest, DecodeJpeg2000Rgb) {
  KernelContext ctx(core::runtime::DefaultOpset(20));
  ImageDecoder decoder(ctx);
  Tensor encoded = MakeEncodedTensor(kJp2Data, sizeof(kJp2Data));
  Tensor result = decoder(encoded, "RGB");

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
