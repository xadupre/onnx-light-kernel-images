// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_kernel_images/register_image_kernels.h"

#include "onnx_core/runtime/kernel_dispatch_table.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_extensions/kernels/kernels/image/include_image_kernels.h"

#include <memory>
#include <string>

namespace onnx_light_kernel_images {

namespace {

using ::onnx_light::core::runtime::KernelBase;
using ::onnx_light::core::runtime::NodeKernelFn;
using ::onnx_light::core::runtime::RegisterKernelFn;
using ::onnx_light::core::runtime::RuntimeContext;
using ::onnx_light::NodeProto;
using ::onnx_light::onnx_kernels::kernel::ImageDecoder;

template <class KernelT> NodeKernelFn MakeKernel() {
  return [](const NodeProto &node, RuntimeContext &rt) -> std::unique_ptr<KernelBase> {
    auto kernel = std::make_unique<KernelT>(rt.kernel_ctx());
    kernel->set_node(node);
    return kernel;
  };
}

} // namespace

void RegisterImageKernels() {
  static bool registered = false;
  if (registered) {
    return;
  }
  registered = true;
  RegisterKernelFn("ai.onnx", "ImageDecoder",
                   onnx_light::core::symbolic::Device::kCPU,
                   MakeKernel<ImageDecoder>());
}

} // namespace onnx_light_kernel_images
