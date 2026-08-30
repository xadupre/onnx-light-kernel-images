// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// TIFF, WebP and JPEG2000 support ported from onnx-light before commit
// 7627400d9, which removed these codecs from its ImageDecoder.

#include "onnx_light_kernel_images/kernels/image/legacy_image_codecs.h"
#include "onnx_light_kernel_images/tiff_compression.h"

#if __has_include("onnx_core/runtime/memory/temporary_buffer.h")
#include "onnx_core/runtime/memory/temporary_buffer.h"
#else
#include "onnx_core/runtime/temporary_buffer.h"
#endif
#include "onnx_extensions/kernels/kernels/image/include_image_kernels.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace onnx_light_kernel_images {
namespace {

using onnx_light::core::runtime::RawBufferAllocator;
using onnx_light::core::runtime::detail::TemporaryTypedBuffer;
using onnx_light::onnx_kernels::kernel::ImageDecoder;

inline uint16_t ReadU16LE(const uint8_t *p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

inline uint32_t ReadU32LE(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

inline uint16_t ReadU16BE(const uint8_t *p) {
  return static_cast<uint16_t>((static_cast<uint16_t>(p[0]) << 8) | static_cast<uint16_t>(p[1]));
}

inline uint32_t ReadU32BE(const uint8_t *p) {
  return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
         (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

bool IsTiff(const uint8_t *data, size_t size) {
  return size >= 4 && (((data[0] == 0x49 && data[1] == 0x49) && ReadU16LE(data + 2) == 42) ||
                       ((data[0] == 0x4D && data[1] == 0x4D) && ReadU16BE(data + 2) == 42));
}

bool IsWebp(const uint8_t *data, size_t size) {
  return size >= 12 && std::memcmp(data, "RIFF", 4) == 0 && std::memcmp(data + 8, "WEBP", 4) == 0;
}

bool IsJpeg2000(const uint8_t *data, size_t size) {
  static constexpr uint8_t kJp2Signature[] = {0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50,
                                              0x20, 0x20, 0x0D, 0x0A, 0x87, 0x0A};
  static constexpr uint8_t kJ2kSignature[] = {0xFF, 0x4F, 0xFF, 0x51};
  return (size >= sizeof(kJp2Signature) &&
          std::memcmp(data, kJp2Signature, sizeof(kJp2Signature)) == 0) ||
         (size >= sizeof(kJ2kSignature) &&
          std::memcmp(data, kJ2kSignature, sizeof(kJ2kSignature)) == 0);
}

bool TryDecodeTiff(const uint8_t *data, size_t size, const std::string &pixel_format,
                   int64_t &out_height, int64_t &out_width, std::vector<uint8_t> &out_pixels) {
  if (size < 8) {
    return false;
  }
  bool little_endian;
  if (data[0] == 0x49 && data[1] == 0x49) {
    little_endian = true;
  } else if (data[0] == 0x4D && data[1] == 0x4D) {
    little_endian = false;
  } else {
    return false;
  }

  const auto rd_u16 = [little_endian](const uint8_t *p) {
    return little_endian ? ReadU16LE(p) : ReadU16BE(p);
  };
  const auto rd_u32 = [little_endian](const uint8_t *p) {
    return little_endian ? ReadU32LE(p) : ReadU32BE(p);
  };

  if (rd_u16(data + 2) != 42u) {
    return false;
  }

  const uint32_t ifd_offset = rd_u32(data + 4);
  if (ifd_offset == 0 || static_cast<uint64_t>(ifd_offset) + 2u > size) {
    return false;
  }
  const uint16_t entry_count = rd_u16(data + ifd_offset);
  const uint64_t entries_end = static_cast<uint64_t>(ifd_offset) + 2u + 12ull * entry_count;
  if (entries_end > size) {
    return false;
  }

  const auto type_size = [](uint16_t type) -> uint32_t {
    switch (type) {
    case 1:
    case 2:
    case 6:
    case 7:
      return 1u;
    case 3:
    case 8:
      return 2u;
    case 4:
    case 9:
    case 11:
      return 4u;
    case 5:
    case 10:
    case 12:
      return 8u;
    default:
      return 0u;
    }
  };

  uint32_t image_width = 0;
  uint32_t image_length = 0;
  uint32_t compression = 1;
  uint32_t photometric = 0xFFFFFFFFu;
  uint32_t samples_per_pixel = 1;
  uint32_t rows_per_strip = 0xFFFFFFFFu;
  uint32_t planar_config = 1;
  std::vector<uint32_t> bits_per_sample;
  std::vector<uint32_t> strip_offsets;
  std::vector<uint32_t> strip_byte_counts;

  const auto read_uint_array = [&](uint16_t type, uint32_t count, const uint8_t *value_field,
                                   std::vector<uint32_t> &out) -> bool {
    const uint32_t element_size = type_size(type);
    if (element_size == 0 || (type != 1 && type != 3 && type != 4)) {
      return false;
    }
    const uint64_t total = static_cast<uint64_t>(element_size) * count;
    const uint8_t *src;
    if (total <= 4u) {
      src = value_field;
    } else {
      const uint32_t offset = rd_u32(value_field);
      if (static_cast<uint64_t>(offset) + total > size) {
        return false;
      }
      src = data + offset;
    }
    out.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
      const uint8_t *p = src + static_cast<size_t>(i) * element_size;
      out[i] = type == 1 ? p[0] : (type == 3 ? rd_u16(p) : rd_u32(p));
    }
    return true;
  };

  for (uint16_t i = 0; i < entry_count; ++i) {
    const uint8_t *entry = data + ifd_offset + 2u + 12u * i;
    const uint16_t tag = rd_u16(entry);
    const uint16_t type = rd_u16(entry + 2);
    const uint32_t count = rd_u32(entry + 4);
    const uint8_t *value_field = entry + 8;
    const auto read_scalar_uint = [&](uint32_t &dst) -> bool {
      std::vector<uint32_t> values;
      if (!read_uint_array(type, count, value_field, values) || values.size() != 1u) {
        return false;
      }
      dst = values[0];
      return true;
    };

    switch (tag) {
    case 256:
      if (!read_scalar_uint(image_width))
        return false;
      break;
    case 257:
      if (!read_scalar_uint(image_length))
        return false;
      break;
    case 258:
      if (!read_uint_array(type, count, value_field, bits_per_sample))
        return false;
      break;
    case 259:
      if (!read_scalar_uint(compression))
        return false;
      break;
    case 262:
      if (!read_scalar_uint(photometric))
        return false;
      break;
    case 273:
      if (!read_uint_array(type, count, value_field, strip_offsets))
        return false;
      break;
    case 277:
      if (!read_scalar_uint(samples_per_pixel))
        return false;
      break;
    case 278:
      if (!read_scalar_uint(rows_per_strip))
        return false;
      break;
    case 279:
      if (!read_uint_array(type, count, value_field, strip_byte_counts))
        return false;
      break;
    case 284:
      if (!read_scalar_uint(planar_config))
        return false;
      break;
    default:
      break;
    }
  }

  if (compression != 1 || planar_config != 1 || image_width == 0 || image_length == 0 ||
      samples_per_pixel == 0 || bits_per_sample.size() != samples_per_pixel ||
      strip_offsets.empty() || strip_offsets.size() != strip_byte_counts.size()) {
    return false;
  }
  for (uint32_t bits : bits_per_sample) {
    if (bits != 8u) {
      return false;
    }
  }
  if (rows_per_strip == 0xFFFFFFFFu) {
    rows_per_strip = image_length;
  }
  if (rows_per_strip == 0) {
    return false;
  }

  if (photometric == 0xFFFFFFFFu) {
    photometric = samples_per_pixel == 1 ? 1u : 2u;
  }
  const bool is_rgb = photometric == 2u && samples_per_pixel == 3u;
  const bool is_gray = photometric == 1u && samples_per_pixel == 1u;
  if (!is_rgb && !is_gray) {
    return false;
  }

  const int64_t channels = ImageDecoder::ChannelCount(pixel_format);
  const uint64_t row_bytes = static_cast<uint64_t>(image_width) * samples_per_pixel;
  uint32_t covered_rows = 0;
  for (size_t strip = 0; strip < strip_offsets.size() && covered_rows < image_length; ++strip) {
    const uint32_t strip_offset = strip_offsets[strip];
    const uint32_t strip_size = strip_byte_counts[strip];
    if (static_cast<uint64_t>(strip_offset) + strip_size > size) {
      return false;
    }
    const uint32_t rows = std::min<uint32_t>(rows_per_strip, image_length - covered_rows);
    if (static_cast<uint64_t>(rows) * row_bytes > strip_size) {
      return false;
    }
    covered_rows += rows;
  }
  if (covered_rows != image_length) {
    return false;
  }

  const uint64_t pixel_count64 = static_cast<uint64_t>(image_length) * image_width;
  if (pixel_count64 > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(channels)) {
    return false;
  }
  const uint64_t output_size64 = pixel_count64 * static_cast<uint64_t>(channels);
  if (pixel_count64 > std::numeric_limits<size_t>::max() / static_cast<size_t>(channels) ||
      output_size64 > out_pixels.max_size()) {
    return false;
  }
  const size_t pixel_count = static_cast<size_t>(pixel_count64);
  out_pixels.assign(pixel_count * static_cast<size_t>(channels), 0);

  uint32_t next_row = 0;
  for (size_t strip = 0; strip < strip_offsets.size(); ++strip) {
    const uint32_t strip_offset = strip_offsets[strip];
    const uint32_t strip_size = strip_byte_counts[strip];
    if (static_cast<uint64_t>(strip_offset) + strip_size > size) {
      return false;
    }
    const uint32_t rows = std::min<uint32_t>(rows_per_strip, image_length - next_row);
    if (static_cast<uint64_t>(rows) * row_bytes > strip_size) {
      return false;
    }
    const uint8_t *src = data + strip_offset;
    for (uint32_t row = 0; row < rows; ++row) {
      const uint8_t *src_row = src + static_cast<size_t>(row) * row_bytes;
      uint8_t *dst_row =
          out_pixels.data() + static_cast<size_t>(next_row + row) * image_width * channels;
      for (uint32_t column = 0; column < image_width; ++column) {
        uint8_t *dst = dst_row + static_cast<size_t>(column) * channels;
        if (is_rgb) {
          const uint8_t red = src_row[column * 3];
          const uint8_t green = src_row[column * 3 + 1];
          const uint8_t blue = src_row[column * 3 + 2];
          if (pixel_format == "RGB") {
            dst[0] = red;
            dst[1] = green;
            dst[2] = blue;
          } else if (pixel_format == "BGR") {
            dst[0] = blue;
            dst[1] = green;
            dst[2] = red;
          } else {
            dst[0] = static_cast<uint8_t>((299 * red + 587 * green + 114 * blue + 500) / 1000);
          }
        } else {
          const uint8_t value = src_row[column];
          dst[0] = value;
          if (channels == 3) {
            dst[1] = value;
            dst[2] = value;
          }
        }
      }
    }
    next_row += rows;
    if (next_row >= image_length) {
      break;
    }
  }

  if (next_row != image_length) {
    return false;
  }
  out_height = image_length;
  out_width = image_width;
  return true;
}

bool TryDecodeWebp(const uint8_t *data, size_t size, const std::string &pixel_format,
                   int64_t &out_height, int64_t &out_width, std::vector<uint8_t> &out_pixels,
                   RawBufferAllocator *allocator) {
  using GetInfoFn = int (*)(const uint8_t *, size_t, int *, int *);
  using DecodeRgbIntoFn = uint8_t *(*)(const uint8_t *, size_t, uint8_t *, size_t, int);

  struct WebPApi {
#if defined(_WIN32)
    HMODULE handle = nullptr;
#else
    void *handle = nullptr;
#endif
    GetInfoFn get_info = nullptr;
    DecodeRgbIntoFn decode_rgb_into = nullptr;
  };

  const auto load_webp_api = []() -> WebPApi {
    WebPApi api{};
#if defined(_WIN32)
    api.handle = LoadLibraryA("libwebp.dll");
    if (!api.handle) {
      api.handle = LoadLibraryA("webp.dll");
    }
    if (!api.handle) {
      return api;
    }
    api.get_info = reinterpret_cast<GetInfoFn>(GetProcAddress(api.handle, "WebPGetInfo"));
    api.decode_rgb_into =
        reinterpret_cast<DecodeRgbIntoFn>(GetProcAddress(api.handle, "WebPDecodeRGBInto"));
#elif defined(__APPLE__)
    api.handle = dlopen("libwebp.dylib", RTLD_LAZY | RTLD_LOCAL);
    if (!api.handle) {
      return api;
    }
    api.get_info = reinterpret_cast<GetInfoFn>(dlsym(api.handle, "WebPGetInfo"));
    api.decode_rgb_into = reinterpret_cast<DecodeRgbIntoFn>(dlsym(api.handle, "WebPDecodeRGBInto"));
#else
    api.handle = dlopen("libwebp.so.7", RTLD_LAZY | RTLD_LOCAL);
    if (!api.handle) {
      api.handle = dlopen("libwebp.so", RTLD_LAZY | RTLD_LOCAL);
    }
    if (!api.handle) {
      return api;
    }
    api.get_info = reinterpret_cast<GetInfoFn>(dlsym(api.handle, "WebPGetInfo"));
    api.decode_rgb_into = reinterpret_cast<DecodeRgbIntoFn>(dlsym(api.handle, "WebPDecodeRGBInto"));
#endif
    if (!api.get_info || !api.decode_rgb_into) {
#if defined(_WIN32)
      FreeLibrary(api.handle);
#else
      dlclose(api.handle);
#endif
      return WebPApi{};
    }
    return api;
  };

  static const WebPApi webp_api = load_webp_api();
  if (!webp_api.get_info || !webp_api.decode_rgb_into) {
    return false;
  }

  int width = 0;
  int height = 0;
  if (webp_api.get_info(data, size, &width, &height) == 0 || width <= 0 || height <= 0) {
    return false;
  }

  const int64_t channels = ImageDecoder::ChannelCount(pixel_format);
  const uint64_t pixel_count64 = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
  if (pixel_count64 > std::numeric_limits<size_t>::max() / 3u ||
      pixel_count64 > std::numeric_limits<size_t>::max() / static_cast<size_t>(channels)) {
    return false;
  }
  const size_t pixel_count = static_cast<size_t>(pixel_count64);
  const size_t rgb_byte_count = pixel_count * 3u;
  if (width > std::numeric_limits<int>::max() / 3) {
    return false;
  }
  TemporaryTypedBuffer<uint8_t> rgb(rgb_byte_count, allocator, "WebP rgb");
  const uint8_t *decoded =
      webp_api.decode_rgb_into(data, size, rgb.data(), rgb_byte_count, width * 3);
  if (decoded != rgb.data()) {
    return false;
  }

  out_pixels.resize(pixel_count * static_cast<size_t>(channels));
  for (size_t i = 0; i < pixel_count; ++i) {
    const uint8_t red = rgb.data()[i * 3u];
    const uint8_t green = rgb.data()[i * 3u + 1u];
    const uint8_t blue = rgb.data()[i * 3u + 2u];
    if (pixel_format == "RGB") {
      out_pixels[i * 3u] = red;
      out_pixels[i * 3u + 1u] = green;
      out_pixels[i * 3u + 2u] = blue;
    } else if (pixel_format == "BGR") {
      out_pixels[i * 3u] = blue;
      out_pixels[i * 3u + 1u] = green;
      out_pixels[i * 3u + 2u] = red;
    } else {
      out_pixels[i] = static_cast<uint8_t>((299 * red + 587 * green + 114 * blue + 500) / 1000);
    }
  }

  out_height = height;
  out_width = width;
  return true;
}

struct OpjImageComp {
  uint32_t dx;
  uint32_t dy;
  uint32_t w;
  uint32_t h;
  uint32_t x0;
  uint32_t y0;
  uint32_t prec;
  uint32_t bpp;
  uint32_t sgnd;
  uint32_t resno_decoded;
  uint32_t factor;
  int32_t *data;
  uint16_t alpha;
};

struct OpjImage {
  uint32_t x0;
  uint32_t y0;
  uint32_t x1;
  uint32_t y1;
  uint32_t numcomps;
  int32_t color_space;
  OpjImageComp *comps;
  uint8_t *icc_profile_buf;
  uint32_t icc_profile_len;
};

struct OpjMemStream {
  const uint8_t *data;
  size_t size;
  size_t pos;
};

extern "C" {
using OpjStreamReadFn = size_t (*)(void *, size_t, void *);
using OpjStreamSkipFn = int64_t (*)(int64_t, void *);
using OpjStreamSeekFn = int32_t (*)(int64_t, void *);
}

size_t OpjMemRead(void *buffer, size_t byte_count, void *user_data) {
  auto *stream = static_cast<OpjMemStream *>(user_data);
  if (stream->pos >= stream->size) {
    return static_cast<size_t>(-1);
  }
  byte_count = std::min(byte_count, stream->size - stream->pos);
  std::memcpy(buffer, stream->data + stream->pos, byte_count);
  stream->pos += byte_count;
  return byte_count;
}

int64_t OpjMemSkip(int64_t byte_count, void *user_data) {
  auto *stream = static_cast<OpjMemStream *>(user_data);
  if (byte_count < 0) {
    return -1;
  }
  const size_t skipped = std::min(static_cast<size_t>(byte_count), stream->size - stream->pos);
  stream->pos += skipped;
  return static_cast<int64_t>(skipped);
}

int32_t OpjMemSeek(int64_t offset, void *user_data) {
  auto *stream = static_cast<OpjMemStream *>(user_data);
  if (offset < 0 || static_cast<size_t>(offset) > stream->size) {
    return 0;
  }
  stream->pos = static_cast<size_t>(offset);
  return 1;
}

bool TryDecodeJpeg2000(const uint8_t *data, size_t size, const std::string &pixel_format,
                       int64_t &out_height, int64_t &out_width, std::vector<uint8_t> &out_pixels,
                       RawBufferAllocator *allocator) {
  constexpr int kCodecJ2k = 0;
  constexpr int kCodecJp2 = 2;
  static constexpr uint8_t kJp2Signature[] = {0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50,
                                              0x20, 0x20, 0x0D, 0x0A, 0x87, 0x0A};

  const int codec_format =
      size >= sizeof(kJp2Signature) && std::memcmp(data, kJp2Signature, sizeof(kJp2Signature)) == 0
          ? kCodecJp2
          : kCodecJ2k;

  using CreateDecompressFn = void *(*)(int);
  using SetDefaultParamsFn = void (*)(void *);
  using SetupDecoderFn = int32_t (*)(void *, void *);
  using StreamDefaultCreateFn = void *(*)(int32_t);
  using StreamSetReadFn = void (*)(void *, OpjStreamReadFn);
  using StreamSetSkipFn = void (*)(void *, OpjStreamSkipFn);
  using StreamSetSeekFn = void (*)(void *, OpjStreamSeekFn);
  using StreamSetUserDataFn = void (*)(void *, void *, void *);
  using StreamSetUserDataLenFn = void (*)(void *, uint64_t);
  using ReadHeaderFn = int32_t (*)(void *, void *, OpjImage **);
  using DecodeFn = int32_t (*)(void *, void *, OpjImage *);
  using EndDecompressFn = int32_t (*)(void *, void *);
  using DestroyCodecFn = void (*)(void *);
  using StreamDestroyFn = void (*)(void *);
  using ImageDestroyFn = void (*)(OpjImage *);

  struct OpenJpegApi {
#if defined(_WIN32)
    HMODULE handle = nullptr;
#else
    void *handle = nullptr;
#endif
    CreateDecompressFn create_decompress = nullptr;
    SetDefaultParamsFn set_default_params = nullptr;
    SetupDecoderFn setup_decoder = nullptr;
    StreamDefaultCreateFn stream_default_create = nullptr;
    StreamSetReadFn stream_set_read = nullptr;
    StreamSetSkipFn stream_set_skip = nullptr;
    StreamSetSeekFn stream_set_seek = nullptr;
    StreamSetUserDataFn stream_set_user_data = nullptr;
    StreamSetUserDataLenFn stream_set_user_data_length = nullptr;
    ReadHeaderFn read_header = nullptr;
    DecodeFn decode = nullptr;
    EndDecompressFn end_decompress = nullptr;
    DestroyCodecFn destroy_codec = nullptr;
    StreamDestroyFn stream_destroy = nullptr;
    ImageDestroyFn image_destroy = nullptr;

    bool complete() const {
      return create_decompress && set_default_params && setup_decoder && stream_default_create &&
             stream_set_read && stream_set_skip && stream_set_seek && stream_set_user_data &&
             stream_set_user_data_length && read_header && decode && end_decompress &&
             destroy_codec && stream_destroy && image_destroy;
    }
  };

  const auto load_openjpeg_api = []() -> OpenJpegApi {
    OpenJpegApi api{};
#if defined(_WIN32)
    api.handle = LoadLibraryA("libopenjp2.dll");
    if (!api.handle) {
      api.handle = LoadLibraryA("openjp2.dll");
    }
    if (!api.handle) {
      return api;
    }
    const auto symbol = [&api](const char *name) -> void * {
      return reinterpret_cast<void *>(GetProcAddress(api.handle, name));
    };
#elif defined(__APPLE__)
    api.handle = dlopen("libopenjp2.dylib", RTLD_LAZY | RTLD_LOCAL);
    if (!api.handle) {
      api.handle = dlopen("libopenjp2.7.dylib", RTLD_LAZY | RTLD_LOCAL);
    }
    if (!api.handle) {
      return api;
    }
    const auto symbol = [&api](const char *name) -> void * { return dlsym(api.handle, name); };
#else
    api.handle = dlopen("libopenjp2.so.7", RTLD_LAZY | RTLD_LOCAL);
    if (!api.handle) {
      api.handle = dlopen("libopenjp2.so", RTLD_LAZY | RTLD_LOCAL);
    }
    if (!api.handle) {
      return api;
    }
    const auto symbol = [&api](const char *name) -> void * { return dlsym(api.handle, name); };
#endif
    api.create_decompress = reinterpret_cast<CreateDecompressFn>(symbol("opj_create_decompress"));
    api.set_default_params =
        reinterpret_cast<SetDefaultParamsFn>(symbol("opj_set_default_decoder_parameters"));
    api.setup_decoder = reinterpret_cast<SetupDecoderFn>(symbol("opj_setup_decoder"));
    api.stream_default_create =
        reinterpret_cast<StreamDefaultCreateFn>(symbol("opj_stream_default_create"));
    api.stream_set_read = reinterpret_cast<StreamSetReadFn>(symbol("opj_stream_set_read_function"));
    api.stream_set_skip = reinterpret_cast<StreamSetSkipFn>(symbol("opj_stream_set_skip_function"));
    api.stream_set_seek = reinterpret_cast<StreamSetSeekFn>(symbol("opj_stream_set_seek_function"));
    api.stream_set_user_data =
        reinterpret_cast<StreamSetUserDataFn>(symbol("opj_stream_set_user_data"));
    api.stream_set_user_data_length =
        reinterpret_cast<StreamSetUserDataLenFn>(symbol("opj_stream_set_user_data_length"));
    api.read_header = reinterpret_cast<ReadHeaderFn>(symbol("opj_read_header"));
    api.decode = reinterpret_cast<DecodeFn>(symbol("opj_decode"));
    api.end_decompress = reinterpret_cast<EndDecompressFn>(symbol("opj_end_decompress"));
    api.destroy_codec = reinterpret_cast<DestroyCodecFn>(symbol("opj_destroy_codec"));
    api.stream_destroy = reinterpret_cast<StreamDestroyFn>(symbol("opj_stream_destroy"));
    api.image_destroy = reinterpret_cast<ImageDestroyFn>(symbol("opj_image_destroy"));
    if (!api.complete()) {
#if defined(_WIN32)
      FreeLibrary(api.handle);
#else
      dlclose(api.handle);
#endif
      return OpenJpegApi{};
    }
    return api;
  };

  static const OpenJpegApi openjpeg_api = load_openjpeg_api();
  if (!openjpeg_api.complete()) {
    return false;
  }

  void *codec = openjpeg_api.create_decompress(codec_format);
  if (!codec) {
    return false;
  }
  void *stream = openjpeg_api.stream_default_create(1);
  if (!stream) {
    openjpeg_api.destroy_codec(codec);
    return false;
  }

  bool ok = false;
  OpjImage *image = nullptr;
  OpjMemStream memory_stream{data, size, 0};
  TemporaryTypedBuffer<uint8_t> parameters(16384, allocator, "JPEG2000 parameters");
  openjpeg_api.set_default_params(parameters.data());
  openjpeg_api.stream_set_read(stream, &OpjMemRead);
  openjpeg_api.stream_set_skip(stream, &OpjMemSkip);
  openjpeg_api.stream_set_seek(stream, &OpjMemSeek);
  openjpeg_api.stream_set_user_data(stream, &memory_stream, nullptr);
  openjpeg_api.stream_set_user_data_length(stream, static_cast<uint64_t>(size));

  if (openjpeg_api.setup_decoder(codec, parameters.data()) &&
      openjpeg_api.read_header(stream, codec, &image) && image != nullptr &&
      openjpeg_api.decode(codec, stream, image) && openjpeg_api.end_decompress(codec, stream)) {
    const int64_t channels = ImageDecoder::ChannelCount(pixel_format);
    const uint32_t component_count = image->numcomps;
    const bool want_color = channels == 3;
    if (component_count >= 1 && (!want_color || component_count >= 3)) {
      const OpjImageComp *components = image->comps;
      if (components == nullptr) {
        goto cleanup;
      }
      const uint32_t width = components[0].w;
      const uint32_t height = components[0].h;
      const uint32_t used_components = component_count >= 3 ? 3u : 1u;
      bool geometry_ok = width > 0 && height > 0;
      for (uint32_t component = 0; component < used_components && geometry_ok; ++component) {
        geometry_ok = components[component].w == width && components[component].h == height &&
                      components[component].data != nullptr && components[component].prec > 0 &&
                      components[component].prec <= 31;
      }
      const uint64_t pixel_count64 = static_cast<uint64_t>(width) * height;
      if (geometry_ok &&
          pixel_count64 <= std::numeric_limits<size_t>::max() / static_cast<size_t>(channels)) {
        const size_t pixel_count = static_cast<size_t>(pixel_count64);
        const auto sample = [components](uint32_t component, size_t index) -> uint8_t {
          const OpjImageComp &source = components[component];
          int32_t value = source.data[index];
          if (source.sgnd) {
            value += 1 << (source.prec - 1);
          }
          const int shift = static_cast<int>(source.prec) - 8;
          if (shift > 0) {
            value >>= shift;
          } else if (shift < 0) {
            value <<= -shift;
          }
          return static_cast<uint8_t>(std::clamp(value, 0, 255));
        };

        out_pixels.resize(pixel_count * static_cast<size_t>(channels));
        for (size_t i = 0; i < pixel_count; ++i) {
          if (pixel_format == "RGB") {
            out_pixels[i * 3u] = sample(0, i);
            out_pixels[i * 3u + 1u] = sample(1, i);
            out_pixels[i * 3u + 2u] = sample(2, i);
          } else if (pixel_format == "BGR") {
            out_pixels[i * 3u] = sample(2, i);
            out_pixels[i * 3u + 1u] = sample(1, i);
            out_pixels[i * 3u + 2u] = sample(0, i);
          } else if (component_count >= 3) {
            const uint8_t red = sample(0, i);
            const uint8_t green = sample(1, i);
            const uint8_t blue = sample(2, i);
            out_pixels[i] =
                static_cast<uint8_t>((299 * red + 587 * green + 114 * blue + 500) / 1000);
          } else {
            out_pixels[i] = sample(0, i);
          }
        }
        out_height = height;
        out_width = width;
        ok = true;
      }
    }
  }

cleanup:
  if (image) {
    openjpeg_api.image_destroy(image);
  }
  openjpeg_api.stream_destroy(stream);
  openjpeg_api.destroy_codec(codec);
  return ok;
}

} // namespace

bool IsLegacyCodecImage(const uint8_t *data, size_t size) {
  return IsTiff(data, size) || IsWebp(data, size) || IsJpeg2000(data, size);
}

bool TryDecodeLegacyCodecImage(const uint8_t *data, size_t size, const std::string &pixel_format,
                               int64_t &height, int64_t &width, std::vector<uint8_t> &pixels,
                               RawBufferAllocator *allocator) {
  if (IsTiff(data, size)) {
    std::vector<uint8_t> rewritten;
    if (RewriteCompressedTiff(data, size, rewritten)) {
      return TryDecodeTiff(rewritten.data(), rewritten.size(), pixel_format, height, width, pixels);
    }
    return TryDecodeTiff(data, size, pixel_format, height, width, pixels);
  }
  if (IsWebp(data, size)) {
    return TryDecodeWebp(data, size, pixel_format, height, width, pixels, allocator);
  }
  if (IsJpeg2000(data, size)) {
    return TryDecodeJpeg2000(data, size, pixel_format, height, width, pixels, allocator);
  }
  return false;
}

} // namespace onnx_light_kernel_images
