// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file register_image_kernels.h
 * @brief Public API for registering the ImageDecoder kernel (BMP, TIFF, JPEG,
 *        PNG, PNM) with the onnx-light kernel dispatch table.
 *
 * This header is the single entry point for downstream C++ consumers. Call
 * :cpp:func:`RegisterImageKernels` once before running any model that uses
 * the ``ImageDecoder`` operator.
 *
 * @code
 * #include <onnx_light_kernel_images/register_image_kernels.h>
 *
 * int main() {
 *     onnx_light_kernel_images::RegisterImageKernels();
 *     // ... run models with ImageDecoder nodes ...
 * }
 * @endcode
 */

namespace onnx_light_kernel_images {

/**
 * Registers the ``ImageDecoder`` kernel (ai.onnx domain, since opset 20)
 * with the onnx-light kernel dispatch table
 * (:cpp:func:`onnx_light::core::runtime::RegisterKernelFn`).
 *
 * Supported image formats:
 *   - **BMP** — 24-bit uncompressed (BI_RGB, BITMAPINFOHEADER)
 *   - **TIFF** — baseline 8-bit-per-sample chunky, uncompressed or
 *     compressed with PackBits, LZW or Deflate/ZIP (with optional
 *     horizontal predictor)
 *   - **JPEG** — baseline JFIF (SOF0, 8-bit, 1 or 3 components)
 *   - **PNG** — 8-bit non-interlaced grayscale / truecolor
 *   - **PNM** — Netpbm family (P1–P6 with 8-bit samples)
 *
 * Idempotent: calling this function more than once is safe and cheap.
 */
void RegisterImageKernels();

} // namespace onnx_light_kernel_images
