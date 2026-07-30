// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// libFuzzer harness for the TIFF-compression front-end
// (``RewriteCompressedTiff``) that onnx-light-kernel-images layers on top of
// the onnx-light ImageDecoder.
//
// Feeds random / malformed bytes to the in-memory PackBits / LZW / Deflate
// decompressor and TIFF rewriter. On a successful rewrite the resulting
// (attacker-influenced) uncompressed TIFF is handed to the ImageDecoder,
// mirroring the full ``TiffAwareImageDecoder`` path used at runtime.

#include "onnx_light_kernel_images/register_image_kernels.h"
#include "onnx_light_kernel_images/tiff_compression.h"

#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/simple_tensor.h"
#include "onnx_extensions/kernels/kernels/image/include_image_kernels.h"

#include <cstddef>
#include <cstdint>
#include <vector>

using onnx_light::core::runtime::DataType;
using onnx_light::core::runtime::DefaultOpset;
using onnx_light::core::runtime::Tensor;
using onnx_light::onnx_kernels::kernel::ImageDecoder;
using onnx_light::onnx_kernels::kernel::KernelContext;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  std::vector<uint8_t> rewritten;
  bool did_rewrite = false;
  try {
    did_rewrite = onnx_light_kernel_images::RewriteCompressedTiff(data, size, rewritten);
  } catch (...) {
    // Malformed compressed TIFF inputs are expected to fail; never crash.
    return 0;
  }

  if (!did_rewrite) {
    return 0;
  }

  // The rewrite succeeded: exercise the ImageDecoder on the produced
  // uncompressed TIFF, just like TiffAwareImageDecoder does at runtime.
  onnx_light_kernel_images::RegisterImageKernels();
  Tensor encoded;
  encoded.data_type = static_cast<int32_t>(DataType::UINT8);
  encoded.shape = {static_cast<int64_t>(rewritten.size())};
  encoded.data.assign(rewritten.begin(), rewritten.end());

  KernelContext ctx(DefaultOpset(20));
  ImageDecoder decoder(ctx);
  try {
    (void)decoder(encoded, "RGB");
  } catch (...) {
    // Decode errors are expected for random / malformed inputs.
  }
  return 0;
}
