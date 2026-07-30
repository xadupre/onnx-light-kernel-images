// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// libFuzzer harness for the ImageDecoder kernel registered by
// onnx-light-kernel-images.
//
// Feeds random / malformed bytes to the decoder for every supported
// pixel format (RGB, BGR, Grayscale). The ImageDecoder sniffs the format
// from the byte-stream (BMP, TIFF, JPEG, JPEG2000, PNG, WebP, PNM), so a
// single harness exercises every format's header parser and pixel reader.

#include "onnx_light_kernel_images/register_image_kernels.h"

#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/simple_tensor.h"
#include "onnx_extensions/kernels/kernels/image/include_image_kernels.h"

#include <cstddef>
#include <cstdint>
#include <string>

using onnx_light::core::runtime::DataType;
using onnx_light::core::runtime::DefaultOpset;
using onnx_light::core::runtime::Tensor;
using onnx_light::onnx_kernels::kernel::ImageDecoder;
using onnx_light::onnx_kernels::kernel::KernelContext;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  // Ensure the ImageDecoder kernel is registered once. RegisterImageKernels
  // is idempotent, so calling it on every input is cheap.
  onnx_light_kernel_images::RegisterImageKernels();

  Tensor encoded;
  encoded.data_type = static_cast<int32_t>(DataType::UINT8);
  encoded.shape = {static_cast<int64_t>(size)};
  encoded.data.assign(data, data + size);

  for (const char *pixel_format : {"RGB", "BGR", "Grayscale"}) {
    KernelContext ctx(DefaultOpset(20));
    ImageDecoder decoder(ctx);
    try {
      (void)decoder(encoded, pixel_format);
    } catch (...) {
      // Decode errors are expected for random / malformed inputs.
    }
  }
  return 0;
}
