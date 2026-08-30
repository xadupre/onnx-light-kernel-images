// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/compute/raw_buffer_allocator.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace onnx_light_kernel_images {

bool IsLegacyCodecImage(const uint8_t *data, size_t size);

bool TryDecodeLegacyCodecImage(const uint8_t *data, size_t size, const std::string &pixel_format,
                               int64_t &height, int64_t &width, std::vector<uint8_t> &pixels,
                               onnx_light::core::runtime::RawBufferAllocator *allocator);

} // namespace onnx_light_kernel_images
