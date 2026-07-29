// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_kernel_images/register_image_kernels.h"
#include "onnx_light_kernel_images/tiff_compression.h"

#include "onnx_core/runtime/simple_tensor.h"
#include "onnx_extensions/kernels/kernels/image/include_image_kernels.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace {

using ::onnx_light::core::runtime::DataType;
using ::onnx_light::core::runtime::DefaultOpset;
using ::onnx_light::core::runtime::KernelContext;
using ::onnx_light::core::runtime::Tensor;
using ::onnx_light::onnx_kernels::kernel::ImageDecoder;

// Decodes an encoded image bytestream with the onnx-light ImageDecoder kernel
// and returns the pixels as a channel-last ``(H, W, C)`` uint8 numpy array.
//
// Mirrors the runtime behavior of the registered kernel: a compressed TIFF is
// first rewritten into an equivalent uncompressed baseline TIFF so the
// PackBits / LZW / Deflate variants decode too. Undecodable inputs follow the
// ONNX schema and yield an empty ``(0, 0, C)`` array.
nb::ndarray<nb::numpy, uint8_t, nb::c_contig> DecodeImage(nb::bytes data,
                                                          const std::string &pixel_format) {
  const auto *ptr = reinterpret_cast<const uint8_t *>(data.c_str());
  const size_t size = data.size();

  Tensor encoded;
  encoded.data_type = static_cast<int32_t>(DataType::UINT8);
  std::vector<uint8_t> rewritten;
  if (onnx_light_kernel_images::RewriteCompressedTiff(ptr, size, rewritten)) {
    encoded.data.assign(rewritten.begin(), rewritten.end());
  } else {
    encoded.data.assign(ptr, ptr + size);
  }
  encoded.shape = std::vector<int64_t>{static_cast<int64_t>(encoded.size_bytes())};

  KernelContext ctx(DefaultOpset(20));
  ImageDecoder decoder(ctx);
  Tensor result = decoder(encoded, pixel_format);

  const std::vector<int64_t> result_shape = result.shape;
  std::vector<size_t> shape(result_shape.size());
  for (size_t i = 0; i < result_shape.size(); ++i) {
    shape[i] = static_cast<size_t>(result_shape[i]);
  }

  const size_t nbytes = result.size_bytes();
  auto *buffer = new uint8_t[nbytes == 0 ? 1 : nbytes];
  if (nbytes != 0) {
    std::memcpy(buffer, result.bytes(), nbytes);
  }
  nb::capsule owner(buffer, [](void *p) noexcept { delete[] reinterpret_cast<uint8_t *>(p); });

  return nb::ndarray<nb::numpy, uint8_t, nb::c_contig>(buffer, shape.size(), shape.data(), owner);
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
        "Decodes an encoded image bytestream with the ImageDecoder kernel.\n\n"
        "Parameters\n"
        "----------\n"
        "data : bytes\n"
        "    Encoded image file contents (BMP, TIFF, JPEG, JPEG2000, PNG, WebP "
        "or PNM).\n"
        "pixel_format : str, optional\n"
        "    One of ``\"RGB\"`` (default), ``\"BGR\"`` or ``\"Grayscale\"``.\n\n"
        "Returns\n"
        "-------\n"
        "numpy.ndarray\n"
        "    Decoded pixels as a channel-last ``(H, W, C)`` ``uint8`` array. "
        "Undecodable inputs return an empty ``(0, 0, C)`` array, as described "
        "by the ONNX ImageDecoder schema.");
}
