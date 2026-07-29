// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_kernel_images/register_image_kernels.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

NB_MODULE(_imgpykernels, m) {
  m.doc() = "Python bindings for onnx-light-kernel-images: "
            "registers the ImageDecoder kernel (BMP, TIFF, JPEG, PNG, PNM) "
            "with the onnx-light kernel dispatch table.";

  m.def(
      "register_image_kernels",
      &onnx_light_kernel_images::RegisterImageKernels,
      "Registers the ImageDecoder kernel (ai.onnx domain) with the onnx-light "
      "kernel dispatch table.\n\n"
      "Supported formats: BMP, TIFF, JPEG, PNG, PNM.\n"
      "Idempotent: calling more than once is safe and cheap.");

  m.def(
      "has_image_kernels",
      []() -> bool { return true; },
      "Returns True when the image kernel extension is available.");
}
