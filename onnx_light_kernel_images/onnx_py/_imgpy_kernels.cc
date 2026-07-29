// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_kernel_images/register_image_kernels.h"
#include "onnx_light_kernel_images/tiff_compression.h"

#include "onnx_core/runtime/kernel_context.h"
#include "onnx_core/runtime/simple_tensor.h"
#include "onnx_extensions/kernels/kernels/image/include_image_kernels.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

using ::onnx_light::core::runtime::DataType;
using ::onnx_light::core::runtime::DefaultOpset;
using ::onnx_light::core::runtime::Tensor;
using ::onnx_light::onnx_kernels::kernel::ImageDecoder;
using ::onnx_light::onnx_kernels::kernel::KernelContext;

// Decodes an encoded image bytestream into an ``(H, W, C)`` ``uint8`` buffer.
//
// Mirrors the behaviour of the registered ``ImageDecoder`` kernel: compressed
// TIFF inputs are first rewritten into an equivalent uncompressed baseline TIFF
// (PackBits, LZW, Deflate/ZIP) before delegating to the onnx-light decoder,
// which handles the ``pixel_format`` conversion. Unsupported or unrecognised
// bytestreams fall back to an empty ``(0, 0, C)`` matrix, matching the ONNX
// ``ImageDecoder`` schema.
std::pair<std::vector<int64_t>, nb::bytes> DecodeImage(nb::bytes data,
                                                       const std::string &pixel_format) {
  const auto *ptr = reinterpret_cast<const uint8_t *>(data.c_str());
  const size_t size = data.size();

  std::vector<uint8_t> rewritten;
  const uint8_t *encoded_ptr = ptr;
  size_t encoded_size = size;
  if (onnx_light_kernel_images::RewriteCompressedTiff(ptr, size, rewritten)) {
    encoded_ptr = rewritten.data();
    encoded_size = rewritten.size();
  }

  Tensor encoded;
  encoded.data_type = static_cast<int32_t>(DataType::UINT8);
  encoded.shape = {static_cast<int64_t>(encoded_size)};
  encoded.data.assign(encoded_ptr, encoded_ptr + encoded_size);

  KernelContext ctx(DefaultOpset(20));
  ImageDecoder decoder(ctx);
  Tensor result = decoder(encoded, pixel_format);

  nb::bytes out(reinterpret_cast<const char *>(result.bytes()), result.size_bytes());
  return {result.shape, std::move(out)};
}

} // namespace

NB_MODULE(_imgpykernels, m) {
  m.doc() = "Python bindings for onnx-light-kernel-images: "
            "registers the ImageDecoder kernel (BMP, TIFF, JPEG, PNG, PNM) "
            "with the onnx-light kernel dispatch table.";

  m.def("register_image_kernels", &onnx_light_kernel_images::RegisterImageKernels,
        "Registers the ImageDecoder kernel (ai.onnx domain) with the onnx-light "
        "kernel dispatch table.\n\n"
        "Supported formats: BMP, TIFF, JPEG, PNG, PNM.\n"
        "Idempotent: calling more than once is safe and cheap.");

  m.def(
      "has_image_kernels", []() -> bool { return true; },
      "Returns True when the image kernel extension is available.");

  m.def("decode_image", &DecodeImage, nb::arg("data"), nb::arg("pixel_format") = "RGB",
        "Decodes an encoded image bytestream into an (H, W, C) uint8 image.\n\n"
        "Parameters\n"
        "----------\n"
        "data : bytes\n"
        "    The encoded image bytestream (BMP, TIFF, JPEG, JPEG2000, PNG, PNM).\n"
        "    Compressed TIFF (PackBits, LZW, Deflate/ZIP) is supported.\n"
        "pixel_format : str, optional\n"
        "    One of ``\"RGB\"`` (default), ``\"BGR\"`` or ``\"Grayscale\"``.\n\n"
        "Returns\n"
        "-------\n"
        "tuple[list[int], bytes]\n"
        "    The ``(H, W, C)`` shape and the raw row-major uint8 pixel bytes.\n"
        "    Unsupported or unrecognised bytestreams yield an empty ``(0, 0, C)``\n"
        "    matrix, as described by the ONNX ImageDecoder schema.");
}
