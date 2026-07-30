// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_kernel_images/register_image_kernels.h"
#include "onnx_light_kernel_images/tiff_compression.h"

#include "onnx_core/runtime/kernel_dispatch_table.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/simple_tensor.h"
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
using ::onnx_light::core::runtime::KernelBase;
using ::onnx_light::core::runtime::MakeOutputTensor;
using ::onnx_light::core::runtime::NodeKernelFn;
using ::onnx_light::core::runtime::RawBufferAllocator;
using ::onnx_light::core::runtime::RegisterKernelFn;
using ::onnx_light::core::runtime::RuntimeContext;
using ::onnx_light::core::runtime::Tensor;
using ::onnx_light::onnx_kernels::kernel::ImageDecoder;

// ImageDecoder variant that adds TIFF compression support (PackBits, LZW,
// Deflate). The onnx-light reference ImageDecoder only decodes uncompressed
// baseline TIFF; here we transparently rewrite a compressed TIFF input into
// an equivalent uncompressed baseline TIFF before delegating to the base
// kernel, which handles the pixel-format conversion. Non-TIFF and
// already-uncompressed inputs are passed through untouched.
//
// The rewritten byte-stream is stored in a result tensor acquired from the
// runtime allocator (:cpp:func:`RuntimeContext::allocator`) via
// :cpp:func:`MakeOutputTensor`, so this kernel never allocates its result
// buffer outside the allocator. When no allocator is attached to the context
// ``MakeOutputTensor`` falls back to inline ``std::vector`` storage.
class TiffAwareImageDecoder : public ImageDecoder {
public:
  using ImageDecoder::ImageDecoder;

  void Run(RuntimeContext &rt) override {
    if (node_ != nullptr && node_->input_size() > 0) {
      const std::string &name = node_->input(0);
      if (!name.empty() && rt.Has(name)) {
        const Tensor &input = rt.Get(name);
        const int32_t data_type = input.data_type;
        std::vector<uint8_t> rewritten;
        if (RewriteCompressedTiff(input.bytes(), input.size_bytes(), rewritten)) {
          RawBufferAllocator *allocator = rt.allocator();
          Tensor decoded = MakeOutputTensor(data_type, {static_cast<int64_t>(rewritten.size())},
                                            rewritten.size(), allocator);
          if (!rewritten.empty()) {
            std::memcpy(decoded.mutable_bytes(), rewritten.data(), rewritten.size());
          }
          rt.Put(name, std::move(decoded));
        }
      }
    }
    ImageDecoder::Run(rt);
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
