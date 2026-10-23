// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/runtime_context.h"
#include "onnx_light_kernel_images/register_image_kernels.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

NB_MODULE(_imgpykernels, m) {
  m.doc() = "Python bindings for onnx-light-kernel-images: "
            "registers the ImageDecoder kernel (BMP, TIFF, JPEG, JPEG2000, PNG, WebP, PNM) "
            "with the onnx-light kernel dispatch table.";

  m.def(
      "register_image_kernels",
      [](nb::object context) {
        if (context.is_none()) {
          onnx_light_kernel_images::RegisterImageKernels();
          return;
        }
        const bool has_runtime_context = nb::hasattr(context, "_ctx");
        if (has_runtime_context) {
          context = context.attr("_ctx");
        }
        using onnx_light::core::runtime::RuntimeContext;
        if (!nb::isinstance<RuntimeContext>(context)) {
          if (has_runtime_context) {
            throw nb::type_error("context._ctx must be a RuntimeContext");
          }
          throw nb::type_error("context must be a RuntimeContext or ReferenceEvaluator");
        }
        onnx_light_kernel_images::RegisterImageKernels(nb::cast<RuntimeContext &>(context));
      },
      nb::arg("context") = nb::none(),
      "Registers the ImageDecoder kernel (ai.onnx domain).\n\n"
      "With no context (or None), updates the global kernel dispatch table.\n"
      "Pass a RuntimeContext or ReferenceEvaluator to register only for that context,\n"
      "before its first run. Already-resolved kernels are not replaced.\n"
      "Supported formats: BMP, TIFF, JPEG, JPEG2000, PNG, WebP, PNM.\n"
      "Repeated registration is safe.");

  m.def(
      "has_image_kernels", []() -> bool { return true; },
      "Returns True when the image kernel extension is available.");
}
