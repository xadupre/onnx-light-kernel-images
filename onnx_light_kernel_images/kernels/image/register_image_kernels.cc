// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_kernel_images/register_image_kernels.h"
#include "onnx_light_kernel_images/kernels/image/legacy_image_codecs.h"
#include "onnx_light_kernel_images/tiff_compression.h"

#if __has_include("onnx_core/runtime/kernels/kernel_dispatch_table.h")
#define ONNX_LIGHT_KERNEL_IMAGES_HAS_CURRENT_RUNTIME_API 1
#include "onnx_core/runtime/kernels/kernel_dispatch_table.h"
#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/memory/simple_tensor.h"
#else
#include "onnx_core/runtime/kernel_dispatch_table.h"
#include "onnx_core/runtime/node_helpers.h"
#include "onnx_core/runtime/simple_tensor.h"
#endif
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/symbolic/sym_tensor.h"
#include "onnx_extensions/kernels/kernels/image/include_image_kernels.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace onnx_light_kernel_images {

namespace {

using ::onnx_light::NodeProto;
using ::onnx_light::core::runtime::DataType;
using ::onnx_light::core::runtime::GetAttributeStringOrDefault;
using ::onnx_light::core::runtime::GetInput;
using ::onnx_light::core::runtime::KernelBase;
using ::onnx_light::core::runtime::MakeOutputTensor;
using ::onnx_light::core::runtime::NodeKernelFn;
using ::onnx_light::core::runtime::RawBufferAllocator;
using ::onnx_light::core::runtime::RegisterKernelFn;
using ::onnx_light::core::runtime::RequireInputCount;
using ::onnx_light::core::runtime::RequireOutputCount;
using ::onnx_light::core::runtime::RuntimeContext;
using ::onnx_light::core::runtime::SetOutput;
using ::onnx_light::core::runtime::Tensor;
using ::onnx_light::onnx_kernels::kernel::ImageDecoder;

// Restores the TIFF, WebP and JPEG2000 formats removed from onnx-light's
// ImageDecoder. BMP, JPEG, PNG and PNM continue to use the upstream decoder.
class TiffAwareImageDecoder : public ImageDecoder {
public:
  using ImageDecoder::ImageDecoder;

  void Run(RuntimeContext &rt) override {
    const NodeProto &node = *node_;
    RequireInputCount(node, 1);
    RequireOutputCount(node, 1);
    const Tensor &input = GetInput(node, 0, rt.tensors());
    if (!IsLegacyCodecImage(input.bytes(), input.size_bytes())) {
      ImageDecoder::Run(rt);
      return;
    }

    EXT_ENFORCE_INVALID(input.data_type == static_cast<int32_t>(DataType::UINT8),
                        "kernel::ImageDecoder only supports UINT8 input tensors.");
    EXT_ENFORCE_INVALID(input.shape.size() == 1u,
                        "kernel::ImageDecoder input ``encoded_stream`` must be a 1-D tensor.");
    EXT_ENFORCE_INVALID(static_cast<int64_t>(input.size_bytes()) == input.element_count(),
                        "kernel::ImageDecoder input data size does not match its shape.");
    const std::string pixel_format = GetAttributeStringOrDefault(node, "pixel_format", "RGB");
    const int64_t channels = ImageDecoder::ChannelCount(pixel_format);
    int64_t height = 0;
    int64_t width = 0;
    std::vector<uint8_t> pixels;
#if ONNX_LIGHT_KERNEL_IMAGES_HAS_CURRENT_RUNTIME_API
    RawBufferAllocator *workspace_allocator = rt.execution_allocator();
#else
    RawBufferAllocator *workspace_allocator = rt.allocator();
#endif
    const bool decoded = TryDecodeLegacyCodecImage(input.bytes(), input.size_bytes(), pixel_format,
                                                   height, width, pixels, workspace_allocator);
    if (!decoded) {
      height = 0;
      width = 0;
      pixels.clear();
    }

#if ONNX_LIGHT_KERNEL_IMAGES_HAS_CURRENT_RUNTIME_API
    Tensor output = rt.MakeOutputTensor(0, static_cast<int32_t>(DataType::UINT8),
                                        {height, width, channels}, pixels.size());
#else
    Tensor output = MakeOutputTensor(static_cast<int32_t>(DataType::UINT8),
                                     {height, width, channels}, pixels.size(), rt.allocator());
#endif
    if (!pixels.empty()) {
      std::memcpy(output.mutable_bytes(), pixels.data(), pixels.size());
    }
#if ONNX_LIGHT_KERNEL_IMAGES_HAS_CURRENT_RUNTIME_API
    SetOutput(node, 0, std::move(output), rt);
#else
    SetOutput(node, 0, std::move(output), rt.tensors());
#endif
  }
};

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
  RegisterKernelFn("ai.onnx", "ImageDecoder", onnx_light::core::symbolic::Device::kCPU,
                   MakeKernel<TiffAwareImageDecoder>());
}

} // namespace onnx_light_kernel_images
