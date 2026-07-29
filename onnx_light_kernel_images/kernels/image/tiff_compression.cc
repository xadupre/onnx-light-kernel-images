// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_light_kernel_images/tiff_compression.h"

#include <array>
#include <cstring>

namespace onnx_light_kernel_images {

namespace {

// ---------------------------------------------------------------------------
// TIFF tag / type / compression constants.
// ---------------------------------------------------------------------------
constexpr uint16_t kTagImageWidth = 256;
constexpr uint16_t kTagImageLength = 257;
constexpr uint16_t kTagBitsPerSample = 258;
constexpr uint16_t kTagCompression = 259;
constexpr uint16_t kTagStripOffsets = 273;
constexpr uint16_t kTagSamplesPerPixel = 277;
constexpr uint16_t kTagRowsPerStrip = 278;
constexpr uint16_t kTagStripByteCounts = 279;
constexpr uint16_t kTagPlanarConfiguration = 284;
constexpr uint16_t kTagPredictor = 317;
constexpr uint16_t kTagTileWidth = 322;
constexpr uint16_t kTagTileOffsets = 324;

constexpr uint16_t kTypeShort = 3;
constexpr uint16_t kTypeLong = 4;

constexpr uint16_t kCompressionNone = 1;
constexpr uint16_t kCompressionLZW = 5;
constexpr uint16_t kCompressionDeflate = 8;
constexpr uint16_t kCompressionPackBits = 32773;
constexpr uint16_t kCompressionDeflateOld = 32946;

// Byte size of a TIFF field element type.
size_t TypeSize(uint16_t type) {
  switch (type) {
  case 1: // BYTE
  case 2: // ASCII
  case 6: // SBYTE
  case 7: // UNDEFINED
    return 1;
  case 3: // SHORT
  case 8: // SSHORT
    return 2;
  case 4:  // LONG
  case 9:  // SLONG
  case 11: // FLOAT
    return 4;
  case 5:  // RATIONAL
  case 10: // SRATIONAL
  case 12: // DOUBLE
    return 8;
  default:
    return 0;
  }
}

// ---------------------------------------------------------------------------
// Endian-aware readers/writers over a byte buffer.
// ---------------------------------------------------------------------------
struct ByteView {
  const uint8_t *data = nullptr;
  size_t size = 0;
  bool little = true;

  bool InBounds(size_t off, size_t n) const { return off <= size && n <= size - off; }

  uint16_t U16(size_t off) const {
    if (little)
      return static_cast<uint16_t>(data[off] | (data[off + 1] << 8));
    return static_cast<uint16_t>((data[off] << 8) | data[off + 1]);
  }

  uint32_t U32(size_t off) const {
    if (little)
      return static_cast<uint32_t>(data[off]) | (static_cast<uint32_t>(data[off + 1]) << 8) |
             (static_cast<uint32_t>(data[off + 2]) << 16) |
             (static_cast<uint32_t>(data[off + 3]) << 24);
    return (static_cast<uint32_t>(data[off]) << 24) | (static_cast<uint32_t>(data[off + 1]) << 16) |
           (static_cast<uint32_t>(data[off + 2]) << 8) | static_cast<uint32_t>(data[off + 3]);
  }
};

void PutU16(std::vector<uint8_t> &buf, size_t off, uint16_t v, bool little) {
  if (little) {
    buf[off] = static_cast<uint8_t>(v & 0xFF);
    buf[off + 1] = static_cast<uint8_t>((v >> 8) & 0xFF);
  } else {
    buf[off] = static_cast<uint8_t>((v >> 8) & 0xFF);
    buf[off + 1] = static_cast<uint8_t>(v & 0xFF);
  }
}

void PutU32(std::vector<uint8_t> &buf, size_t off, uint32_t v, bool little) {
  if (little) {
    buf[off] = static_cast<uint8_t>(v & 0xFF);
    buf[off + 1] = static_cast<uint8_t>((v >> 8) & 0xFF);
    buf[off + 2] = static_cast<uint8_t>((v >> 16) & 0xFF);
    buf[off + 3] = static_cast<uint8_t>((v >> 24) & 0xFF);
  } else {
    buf[off] = static_cast<uint8_t>((v >> 24) & 0xFF);
    buf[off + 1] = static_cast<uint8_t>((v >> 16) & 0xFF);
    buf[off + 2] = static_cast<uint8_t>((v >> 8) & 0xFF);
    buf[off + 3] = static_cast<uint8_t>(v & 0xFF);
  }
}

// ---------------------------------------------------------------------------
// Parsed IFD entry.
// ---------------------------------------------------------------------------
struct Entry {
  uint16_t tag = 0;
  uint16_t type = 0;
  uint32_t count = 0;
  // Absolute offset of the 12-byte entry's 4-byte value field within the
  // original stream (used to read inline values or the out-of-line pointer).
  size_t value_field_off = 0;
};

// Reads a numeric field (SHORT or LONG, single or multiple values) into `out`.
// Returns false when the field cannot be read (out of bounds, unexpected type).
bool ReadIntArray(const ByteView &v, const Entry &e, std::vector<uint64_t> &out) {
  size_t esize = TypeSize(e.type);
  if (esize == 0)
    return false;
  size_t total = esize * static_cast<size_t>(e.count);
  size_t base;
  if (total <= 4) {
    base = e.value_field_off;
  } else {
    if (!v.InBounds(e.value_field_off, 4))
      return false;
    base = v.U32(e.value_field_off);
  }
  if (!v.InBounds(base, total))
    return false;
  out.clear();
  out.reserve(e.count);
  for (uint32_t i = 0; i < e.count; ++i) {
    size_t off = base + i * esize;
    switch (e.type) {
    case kTypeShort:
      out.push_back(v.U16(off));
      break;
    case kTypeLong:
      out.push_back(v.U32(off));
      break;
    case 1: // BYTE
      out.push_back(v.data[off]);
      break;
    default:
      return false;
    }
  }
  return true;
}

// Returns the first numeric value of an entry, or `def` when unreadable.
uint64_t ReadScalar(const ByteView &v, const Entry &e, uint64_t def) {
  std::vector<uint64_t> tmp;
  if (!ReadIntArray(v, e, tmp) || tmp.empty())
    return def;
  return tmp[0];
}

// ---------------------------------------------------------------------------
// PackBits (RLE) decoder. Decodes the whole strip sequentially.
// ---------------------------------------------------------------------------
bool DecodePackBits(const uint8_t *src, size_t n, std::vector<uint8_t> &out) {
  size_t i = 0;
  while (i < n) {
    int8_t hdr = static_cast<int8_t>(src[i++]);
    if (hdr >= 0) {
      size_t count = static_cast<size_t>(hdr) + 1;
      if (i + count > n)
        return false;
      out.insert(out.end(), src + i, src + i + count);
      i += count;
    } else if (hdr != -128) {
      size_t count = static_cast<size_t>(-static_cast<int>(hdr)) + 1;
      if (i >= n)
        return false;
      out.insert(out.end(), count, src[i]);
      ++i;
    }
    // hdr == -128 is a no-op.
  }
  return true;
}

// ---------------------------------------------------------------------------
// TIFF LZW decoder (MSB-first bit packing, 9-12 bit codes, early change).
// ---------------------------------------------------------------------------
class LzwBitReader {
public:
  LzwBitReader(const uint8_t *src, size_t n) : src_(src), n_(n) {}

  // Reads `width` bits MSB-first. Sets ok=false past end of input.
  uint32_t Read(int width, bool &ok) {
    uint32_t value = 0;
    for (int i = 0; i < width; ++i) {
      if (byte_ >= n_) {
        ok = false;
        return 0;
      }
      uint32_t bit = (src_[byte_] >> (7 - bit_)) & 1u;
      value = (value << 1) | bit;
      if (++bit_ == 8) {
        bit_ = 0;
        ++byte_;
      }
    }
    ok = true;
    return value;
  }

private:
  const uint8_t *src_;
  size_t n_;
  size_t byte_ = 0;
  int bit_ = 0;
};

bool DecodeLzw(const uint8_t *src, size_t n, std::vector<uint8_t> &out) {
  constexpr int kClear = 256;
  constexpr int kEoi = 257;
  std::vector<std::vector<uint8_t>> table;
  auto reset_table = [&table]() {
    table.clear();
    table.resize(258);
    for (int i = 0; i < 256; ++i)
      table[i] = {static_cast<uint8_t>(i)};
  };
  reset_table();

  LzwBitReader reader(src, n);
  int code_width = 9;
  int prev = -1;
  bool ok = true;

  while (true) {
    uint32_t code = reader.Read(code_width, ok);
    if (!ok)
      break; // ran out of input without EOI: accept what we have.
    if (static_cast<int>(code) == kClear) {
      reset_table();
      code_width = 9;
      prev = -1;
      continue;
    }
    if (static_cast<int>(code) == kEoi)
      break;

    std::vector<uint8_t> entry;
    if (code < table.size()) {
      entry = table[code];
    } else if (static_cast<size_t>(code) == table.size() && prev >= 0) {
      entry = table[prev];
      entry.push_back(table[prev][0]);
    } else {
      return false; // invalid code
    }

    out.insert(out.end(), entry.begin(), entry.end());

    if (prev >= 0) {
      std::vector<uint8_t> new_entry = table[prev];
      new_entry.push_back(entry[0]);
      table.push_back(std::move(new_entry));
    }
    prev = static_cast<int>(code);

    // "Early change": widen one code before the table is full.
    if (table.size() + 1 == (1u << code_width) && code_width < 12)
      ++code_width;
  }
  return true;
}

// ---------------------------------------------------------------------------
// DEFLATE (RFC 1951) inflate, with optional zlib (RFC 1950) header.
// ---------------------------------------------------------------------------
class DeflateBitReader {
public:
  DeflateBitReader(const uint8_t *src, size_t n) : src_(src), n_(n) {}

  int Bit() {
    if (byte_ >= n_) {
      eof_ = true;
      return 0;
    }
    int b = (src_[byte_] >> bit_) & 1;
    if (++bit_ == 8) {
      bit_ = 0;
      ++byte_;
    }
    return b;
  }

  uint32_t Bits(int count) {
    uint32_t v = 0;
    for (int i = 0; i < count; ++i)
      v |= static_cast<uint32_t>(Bit()) << i;
    return v;
  }

  void AlignToByte() {
    if (bit_ != 0) {
      bit_ = 0;
      ++byte_;
    }
  }

  bool ReadBytes(uint8_t *dst, size_t count) {
    if (byte_ + count > n_)
      return false;
    std::memcpy(dst, src_ + byte_, count);
    byte_ += count;
    return true;
  }

  bool eof() const { return eof_; }

private:
  const uint8_t *src_;
  size_t n_;
  size_t byte_ = 0;
  int bit_ = 0;
  bool eof_ = false;
};

// Canonical Huffman decoder built from per-symbol code lengths.
struct HuffTree {
  // counts[len] = number of codes of length len; symbols listed by increasing
  // length then symbol value.
  std::array<int, 16> count{};
  std::vector<int> symbol;

  bool Build(const std::vector<int> &lengths) {
    count.fill(0);
    for (int len : lengths) {
      if (len < 0 || len > 15)
        return false;
      ++count[len];
    }
    count[0] = 0;
    symbol.assign(lengths.size(), 0);
    std::array<int, 16> offsets{};
    int sum = 0;
    for (int len = 1; len < 16; ++len) {
      offsets[len] = sum;
      sum += count[len];
    }
    for (size_t sym = 0; sym < lengths.size(); ++sym) {
      if (lengths[sym] != 0)
        symbol[offsets[lengths[sym]]++] = static_cast<int>(sym);
    }
    return true;
  }

  int Decode(DeflateBitReader &br) const {
    int code = 0;
    int first = 0;
    int index = 0;
    for (int len = 1; len <= 15; ++len) {
      code |= br.Bit();
      int cnt = count[len];
      if (code - first < cnt)
        return symbol[index + (code - first)];
      index += cnt;
      first += cnt;
      first <<= 1;
      code <<= 1;
    }
    return -1;
  }
};

bool InflateBlock(DeflateBitReader &br, const HuffTree &lit, const HuffTree &dist,
                  std::vector<uint8_t> &out) {
  static const int kLenBase[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                   31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
  static const int kLenExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                    2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
  static const int kDistBase[30] = {1,    2,    3,    4,    5,    7,    9,    13,    17,    25,
                                    33,   49,   65,   97,   129,  193,  257,  385,   513,   769,
                                    1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
  static const int kDistExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
                                     6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
  while (true) {
    int sym = lit.Decode(br);
    if (br.eof() || sym < 0)
      return false;
    if (sym == 256)
      return true; // end of block
    if (sym < 256) {
      out.push_back(static_cast<uint8_t>(sym));
      continue;
    }
    sym -= 257;
    if (sym >= 29)
      return false;
    int length = kLenBase[sym] + static_cast<int>(br.Bits(kLenExtra[sym]));
    int dsym = dist.Decode(br);
    if (dsym < 0 || dsym >= 30)
      return false;
    int distance = kDistBase[dsym] + static_cast<int>(br.Bits(kDistExtra[dsym]));
    if (distance <= 0 || static_cast<size_t>(distance) > out.size())
      return false;
    size_t start = out.size() - distance;
    for (int i = 0; i < length; ++i)
      out.push_back(out[start + i]);
  }
}

bool BuildFixedTrees(HuffTree &lit, HuffTree &dist) {
  std::vector<int> lit_lengths(288);
  for (int i = 0; i < 144; ++i)
    lit_lengths[i] = 8;
  for (int i = 144; i < 256; ++i)
    lit_lengths[i] = 9;
  for (int i = 256; i < 280; ++i)
    lit_lengths[i] = 7;
  for (int i = 280; i < 288; ++i)
    lit_lengths[i] = 8;
  std::vector<int> dist_lengths(30, 5);
  return lit.Build(lit_lengths) && dist.Build(dist_lengths);
}

bool BuildDynamicTrees(DeflateBitReader &br, HuffTree &lit, HuffTree &dist) {
  static const int kOrder[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
  int hlit = static_cast<int>(br.Bits(5)) + 257;
  int hdist = static_cast<int>(br.Bits(5)) + 1;
  int hclen = static_cast<int>(br.Bits(4)) + 4;
  if (hlit > 286 || hdist > 30)
    return false;

  std::vector<int> cl_lengths(19, 0);
  for (int i = 0; i < hclen; ++i)
    cl_lengths[kOrder[i]] = static_cast<int>(br.Bits(3));
  HuffTree cl_tree;
  if (!cl_tree.Build(cl_lengths))
    return false;

  std::vector<int> lengths;
  lengths.reserve(hlit + hdist);
  while (static_cast<int>(lengths.size()) < hlit + hdist) {
    int sym = cl_tree.Decode(br);
    if (br.eof() || sym < 0)
      return false;
    if (sym < 16) {
      lengths.push_back(sym);
    } else if (sym == 16) {
      if (lengths.empty())
        return false;
      int repeat = static_cast<int>(br.Bits(2)) + 3;
      int prev = lengths.back();
      for (int i = 0; i < repeat; ++i)
        lengths.push_back(prev);
    } else if (sym == 17) {
      int repeat = static_cast<int>(br.Bits(3)) + 3;
      for (int i = 0; i < repeat; ++i)
        lengths.push_back(0);
    } else { // sym == 18
      int repeat = static_cast<int>(br.Bits(7)) + 11;
      for (int i = 0; i < repeat; ++i)
        lengths.push_back(0);
    }
  }
  if (static_cast<int>(lengths.size()) != hlit + hdist)
    return false;

  std::vector<int> lit_lengths(lengths.begin(), lengths.begin() + hlit);
  std::vector<int> dist_lengths(lengths.begin() + hlit, lengths.end());
  return lit.Build(lit_lengths) && dist.Build(dist_lengths);
}

bool Inflate(const uint8_t *src, size_t n, std::vector<uint8_t> &out) {
  DeflateBitReader br(src, n);
  bool final_block = false;
  while (!final_block) {
    final_block = br.Bit() != 0;
    if (br.eof())
      return false;
    int type = static_cast<int>(br.Bits(2));
    if (type == 0) { // stored
      br.AlignToByte();
      uint8_t hdr[4];
      if (!br.ReadBytes(hdr, 4))
        return false;
      uint16_t len = static_cast<uint16_t>(hdr[0] | (hdr[1] << 8));
      size_t base = out.size();
      out.resize(base + len);
      if (len > 0 && !br.ReadBytes(out.data() + base, len))
        return false;
    } else if (type == 1) { // fixed Huffman
      HuffTree lit, dist;
      if (!BuildFixedTrees(lit, dist) || !InflateBlock(br, lit, dist, out))
        return false;
    } else if (type == 2) { // dynamic Huffman
      HuffTree lit, dist;
      if (!BuildDynamicTrees(br, lit, dist) || !InflateBlock(br, lit, dist, out))
        return false;
    } else {
      return false; // reserved
    }
  }
  return true;
}

bool DecodeDeflate(const uint8_t *src, size_t n, std::vector<uint8_t> &out) {
  // TIFF Deflate uses a zlib (RFC 1950) wrapper: 2-byte header, then the raw
  // DEFLATE stream, then a 4-byte Adler-32 checksum. Detect and skip the
  // header when present; otherwise inflate the raw stream directly.
  if (n >= 2) {
    uint16_t cmf_flg = static_cast<uint16_t>((src[0] << 8) | src[1]);
    if ((src[0] & 0x0F) == 8 && (cmf_flg % 31) == 0) {
      return Inflate(src + 2, n - 2, out);
    }
  }
  return Inflate(src, n, out);
}

// ---------------------------------------------------------------------------
// Horizontal-differencing predictor (Predictor == 2) reversal for 8-bit data.
// ---------------------------------------------------------------------------
void UndoHorizontalPredictor(std::vector<uint8_t> &data, uint32_t width, uint32_t rows,
                             uint32_t samples_per_pixel) {
  size_t row_bytes = static_cast<size_t>(width) * samples_per_pixel;
  if (row_bytes == 0)
    return;
  for (uint32_t r = 0; r < rows; ++r) {
    size_t base = static_cast<size_t>(r) * row_bytes;
    if (base + row_bytes > data.size())
      break;
    for (size_t b = samples_per_pixel; b < row_bytes; ++b)
      data[base + b] = static_cast<uint8_t>(data[base + b] + data[base + b - samples_per_pixel]);
  }
}

} // namespace

bool RewriteCompressedTiff(const uint8_t *data, size_t size, std::vector<uint8_t> &out) {
  if (data == nullptr || size < 8)
    return false;

  ByteView v;
  v.data = data;
  v.size = size;
  if (data[0] == 0x49 && data[1] == 0x49)
    v.little = true;
  else if (data[0] == 0x4D && data[1] == 0x4D)
    v.little = false;
  else
    return false;

  if (v.U16(2) != 42) // classic TIFF magic (BigTIFF is 43, unsupported)
    return false;

  uint32_t ifd_offset = v.U32(4);
  if (!v.InBounds(ifd_offset, 2))
    return false;
  uint16_t entry_count = v.U16(ifd_offset);
  size_t entries_start = static_cast<size_t>(ifd_offset) + 2;
  if (!v.InBounds(entries_start, static_cast<size_t>(entry_count) * 12 + 4))
    return false;

  std::vector<Entry> entries;
  entries.reserve(entry_count);
  for (uint16_t i = 0; i < entry_count; ++i) {
    size_t base = entries_start + static_cast<size_t>(i) * 12;
    Entry e;
    e.tag = v.U16(base);
    e.type = v.U16(base + 2);
    e.count = v.U32(base + 4);
    e.value_field_off = base + 8;
    entries.push_back(e);
  }

  auto find = [&entries](uint16_t tag) -> const Entry * {
    for (const Entry &e : entries)
      if (e.tag == tag)
        return &e;
    return nullptr;
  };

  const Entry *compression_entry = find(kTagCompression);
  uint64_t compression =
      compression_entry ? ReadScalar(v, *compression_entry, kCompressionNone) : kCompressionNone;

  bool supported = compression == kCompressionPackBits || compression == kCompressionLZW ||
                   compression == kCompressionDeflate || compression == kCompressionDeflateOld;
  if (!supported)
    return false; // uncompressed or unsupported: let the SDK handle it as-is.

  // Only chunky, strip-based, 8-bit-per-sample images are rewritten; anything
  // else is left to the SDK (which falls back to the empty matrix).
  if (find(kTagTileWidth) != nullptr || find(kTagTileOffsets) != nullptr)
    return false;
  const Entry *planar_entry = find(kTagPlanarConfiguration);
  if (planar_entry && ReadScalar(v, *planar_entry, 1) != 1)
    return false;

  const Entry *width_entry = find(kTagImageWidth);
  const Entry *length_entry = find(kTagImageLength);
  const Entry *strip_offsets_entry = find(kTagStripOffsets);
  const Entry *strip_bytecounts_entry = find(kTagStripByteCounts);
  if (!width_entry || !length_entry || !strip_offsets_entry || !strip_bytecounts_entry)
    return false;

  uint32_t width = static_cast<uint32_t>(ReadScalar(v, *width_entry, 0));
  uint32_t height = static_cast<uint32_t>(ReadScalar(v, *length_entry, 0));
  if (width == 0 || height == 0)
    return false;

  const Entry *spp_entry = find(kTagSamplesPerPixel);
  uint32_t samples_per_pixel = spp_entry ? static_cast<uint32_t>(ReadScalar(v, *spp_entry, 1)) : 1;
  if (samples_per_pixel == 0)
    return false;

  const Entry *bps_entry = find(kTagBitsPerSample);
  uint32_t bits_per_sample = bps_entry ? static_cast<uint32_t>(ReadScalar(v, *bps_entry, 1)) : 1;
  if (bits_per_sample != 8)
    return false; // SDK only decodes 8-bit-per-sample TIFF.

  const Entry *predictor_entry = find(kTagPredictor);
  uint64_t predictor = predictor_entry ? ReadScalar(v, *predictor_entry, 1) : 1;
  if (predictor != 1 && predictor != 2)
    return false;

  std::vector<uint64_t> strip_offsets;
  std::vector<uint64_t> strip_bytecounts;
  if (!ReadIntArray(v, *strip_offsets_entry, strip_offsets) ||
      !ReadIntArray(v, *strip_bytecounts_entry, strip_bytecounts))
    return false;
  if (strip_offsets.empty() || strip_offsets.size() != strip_bytecounts.size())
    return false;

  // Expected number of decoded bytes for the whole image. Computed with 64-bit
  // arithmetic so a crafted width/height cannot overflow, and used both to
  // bound the reservation below and to validate the decompressed size.
  uint64_t expected64 = static_cast<uint64_t>(width) * height * samples_per_pixel;

  // Decompress every strip and concatenate into one row-major buffer. The
  // reservation is capped so a crafted image with huge dimensions but tiny
  // strip data cannot trigger an oversized allocation; correctness does not
  // depend on it.
  std::vector<uint8_t> raw;
  uint64_t total_strip_bytes = 0;
  for (uint64_t bc : strip_bytecounts)
    total_strip_bytes += bc;
  uint64_t reserve_hint = expected64 < total_strip_bytes ? expected64 : total_strip_bytes;
  constexpr uint64_t kReserveCap = 64ull * 1024 * 1024;
  raw.reserve(static_cast<size_t>(reserve_hint < kReserveCap ? reserve_hint : kReserveCap));
  for (size_t i = 0; i < strip_offsets.size(); ++i) {
    size_t off = static_cast<size_t>(strip_offsets[i]);
    size_t len = static_cast<size_t>(strip_bytecounts[i]);
    if (!v.InBounds(off, len))
      return false;
    const uint8_t *src = data + off;
    bool ok = false;
    switch (compression) {
    case kCompressionPackBits:
      ok = DecodePackBits(src, len, raw);
      break;
    case kCompressionLZW:
      ok = DecodeLzw(src, len, raw);
      break;
    case kCompressionDeflate:
    case kCompressionDeflateOld:
      ok = DecodeDeflate(src, len, raw);
      break;
    default:
      ok = false;
      break;
    }
    if (!ok)
      return false;
  }

  size_t expected = static_cast<size_t>(expected64);
  if (raw.size() < expected)
    return false;
  raw.resize(expected); // drop any padding beyond the image.

  if (predictor == 2)
    UndoHorizontalPredictor(raw, width, height, samples_per_pixel);

  // -------------------------------------------------------------------------
  // Rebuild an uncompressed, single-strip baseline TIFF. All original IFD
  // entries are preserved (with their out-of-line values relocated) except:
  //   - Compression      -> 1 (none)
  //   - Predictor        -> 1 (data has been un-predicted)
  //   - RowsPerStrip     -> height (single strip)
  //   - StripOffsets     -> single LONG pointing at the appended pixel data
  //   - StripByteCounts  -> single LONG equal to raw.size()
  // -------------------------------------------------------------------------
  const bool little = v.little;
  size_t ifd_size = 2 + static_cast<size_t>(entry_count) * 12 + 4;
  size_t ifd_end = 8 + ifd_size;

  std::vector<uint8_t> ifd(ifd_size, 0);
  PutU16(ifd, 0, entry_count, little);
  // next-IFD pointer (last 4 bytes) stays 0.

  std::vector<uint8_t> trailing; // out-of-line values, based at ifd_end.

  for (uint16_t i = 0; i < entry_count; ++i) {
    const Entry &e = entries[i];
    size_t rec = 2 + static_cast<size_t>(i) * 12;
    PutU16(ifd, rec, e.tag, little);

    auto write_scalar_long = [&](uint32_t value) {
      PutU16(ifd, rec + 2, kTypeLong, little);
      PutU32(ifd, rec + 4, 1, little);
      PutU32(ifd, rec + 8, value, little);
    };

    if (e.tag == kTagCompression) {
      PutU16(ifd, rec + 2, kTypeShort, little);
      PutU32(ifd, rec + 4, 1, little);
      // SHORT value is stored left-justified in the value field.
      PutU16(ifd, rec + 8, kCompressionNone, little);
      PutU16(ifd, rec + 10, 0, little);
    } else if (e.tag == kTagPredictor) {
      PutU16(ifd, rec + 2, kTypeShort, little);
      PutU32(ifd, rec + 4, 1, little);
      PutU16(ifd, rec + 8, 1, little);
      PutU16(ifd, rec + 10, 0, little);
    } else if (e.tag == kTagRowsPerStrip) {
      write_scalar_long(height);
    } else if (e.tag == kTagStripOffsets) {
      // Value (offset) is patched in after the trailing region is known.
      write_scalar_long(0);
    } else if (e.tag == kTagStripByteCounts) {
      write_scalar_long(static_cast<uint32_t>(raw.size()));
    } else {
      PutU16(ifd, rec + 2, e.type, little);
      PutU32(ifd, rec + 4, e.count, little);
      size_t total = TypeSize(e.type) * static_cast<size_t>(e.count);
      if (total <= 4) {
        // Copy the inline value bytes verbatim.
        std::memcpy(ifd.data() + rec + 8, data + e.value_field_off, 4);
      } else {
        size_t src_off = v.U32(e.value_field_off);
        if (!v.InBounds(src_off, total))
          return false;
        // Relocate the out-of-line value, keeping word alignment.
        if (trailing.size() % 2 != 0)
          trailing.push_back(0);
        uint32_t new_off = static_cast<uint32_t>(ifd_end + trailing.size());
        trailing.insert(trailing.end(), data + src_off, data + src_off + total);
        PutU32(ifd, rec + 8, new_off, little);
      }
    }
  }

  // Append pixel data after the relocated out-of-line values.
  if (trailing.size() % 2 != 0)
    trailing.push_back(0);
  uint32_t strip_data_off = static_cast<uint32_t>(ifd_end + trailing.size());
  trailing.insert(trailing.end(), raw.begin(), raw.end());

  // Patch the StripOffsets value now that the pixel-data offset is known.
  for (uint16_t i = 0; i < entry_count; ++i) {
    if (entries[i].tag == kTagStripOffsets) {
      size_t rec = 2 + static_cast<size_t>(i) * 12;
      PutU32(ifd, rec + 8, strip_data_off, little);
      break;
    }
  }

  out.clear();
  out.reserve(ifd_end + trailing.size());
  // Header: byte order, magic 42, IFD offset (8).
  out.push_back(little ? 0x49 : 0x4D);
  out.push_back(little ? 0x49 : 0x4D);
  out.resize(8, 0);
  PutU16(out, 2, 42, little);
  PutU32(out, 4, 8, little);
  out.insert(out.end(), ifd.begin(), ifd.end());
  out.insert(out.end(), trailing.begin(), trailing.end());
  return true;
}

} // namespace onnx_light_kernel_images
