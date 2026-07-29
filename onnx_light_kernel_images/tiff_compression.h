// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file tiff_compression.h
 * @brief Front-end that adds TIFF compression support on top of the
 *        onnx-light ImageDecoder kernel.
 *
 * The onnx-light C++ ImageDecoder reference kernel only decodes *uncompressed*
 * baseline TIFF (8-bit per sample, chunky). To add support for the common
 * compressed TIFF variants without reimplementing the whole TIFF pixel parser,
 * :cpp:func:`RewriteCompressedTiff` decompresses the strips of a compressed
 * TIFF in memory and rewrites the byte-stream into an equivalent *uncompressed*
 * single-strip baseline TIFF. The rewritten stream can then be handed to the
 * onnx-light ImageDecoder, which handles photometric interpretation, channel
 * ordering and the requested ``pixel_format``.
 *
 * Supported compressions:
 *   - PackBits (tag value 32773)
 *   - LZW (tag value 5), including horizontal differencing predictor
 *   - Deflate / ZIP (tag values 8 and 32946), including horizontal
 *     differencing predictor
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace onnx_light_kernel_images {

/**
 * Rewrites a compressed TIFF byte-stream into an equivalent uncompressed
 * single-strip baseline TIFF.
 *
 * @param data Pointer to the encoded byte-stream.
 * @param size Number of bytes available at @p data.
 * @param out  On success, receives the rewritten uncompressed TIFF.
 *
 * @return ``true`` when @p data is a TIFF that uses a supported compression
 *         (PackBits, LZW or Deflate/ZIP) with a chunky, 8-bit-per-sample,
 *         strip-based layout, and it was successfully decompressed and
 *         rewritten into @p out. Returns ``false`` and leaves @p out untouched
 *         when @p data is not such a TIFF (e.g. not a TIFF at all, already
 *         uncompressed, tiled, planar, non-8-bit, or an unsupported
 *         compression); callers should then use the original bytes.
 */
bool RewriteCompressedTiff(const uint8_t *data, size_t size, std::vector<uint8_t> &out);

} // namespace onnx_light_kernel_images
