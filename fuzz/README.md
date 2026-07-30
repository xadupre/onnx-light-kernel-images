# onnx-light-kernel-images Fuzz Harnesses (C++)

This directory contains [libFuzzer](https://llvm.org/docs/LibFuzzer.html)-based
C++ fuzz targets for the image kernels shipped by
`onnx-light-kernel-images`. They feed random / malformed bytes to the public
decode surface so sanitizers can catch memory-safety bugs in the image parsers
and in this repository's own TIFF-compression front-end.

## Harnesses

| File                        | Entry point fuzzed                                                        | Input path                     |
|-----------------------------|---------------------------------------------------------------------------|--------------------------------|
| `fuzz_image_decoder.cc`     | `ImageDecoder` for every pixel format (RGB / BGR / Grayscale)             | Raw bytes → image sniffer      |
| `fuzz_tiff_compression.cc`  | `RewriteCompressedTiff` + `ImageDecoder` on the rewritten TIFF            | Raw bytes → TIFF decompressor  |
| `make_seed_corpus.cc`       | *(seed generator, not a fuzzer)*                                          | Writes valid sample images     |

The `ImageDecoder` sniffs the container format (BMP, TIFF, JPEG, JPEG2000,
PNG, WebP, PNM) from the byte-stream, so `fuzz_image_decoder` exercises every
format's header parser and pixel reader through a single entry point.

`fuzz_tiff_compression` targets the custom in-memory PackBits / LZW /
Deflate decompressor and TIFF rewriter that this repository layers on top of
the onnx-light decoder, then hands the (attacker-influenced) rewritten TIFF to
the decoder, mirroring the runtime `TiffAwareImageDecoder` path.

## Building

The harnesses are compiled when `ONNX_LIGHT_KERNEL_IMAGES_BUILD_FUZZERS=ON` is
passed to CMake. Clang is required because libFuzzer (`-fsanitize=fuzzer,...`)
ships with Clang.

```bash
CC=clang CXX=clang++ cmake -S . -B build-fuzz -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DONNX_LIGHT_KERNEL_IMAGES_BUILD_FUZZERS=ON \
    -DONNX_LIGHT_KERNEL_IMAGES_BUILD_PYTHON=OFF \
    -DONNX_LIGHT_KERNEL_IMAGES_INSTALL=OFF
cmake --build build-fuzz -j
```

The default sanitizer set is `address` (so the link line is
`-fsanitize=fuzzer,address`). Pass
`-DONNX_LIGHT_KERNEL_IMAGES_FUZZER_SANITIZERS=...` to override it; the `fuzzer`
sanitizer is always added automatically. The repository's own kernel library
(`lib_onnx_light_kernel_images`) is instrumented with libFuzzer edge coverage
too, so the fuzzer explores the custom decode paths. The onnx-light SDK itself
is consumed as a prebuilt archive and therefore cannot be instrumented here.

## Running locally

Each harness is a standard libFuzzer executable. First generate the seed
corpora, then point each harness at its directory:

```bash
./build-fuzz/make_seed_corpus \
    /tmp/fuzz_seeds/image_decoder \
    /tmp/fuzz_seeds/tiff_compression

./build-fuzz/fuzz_image_decoder    /tmp/fuzz_seeds/image_decoder    -runs=4000
./build-fuzz/fuzz_tiff_compression /tmp/fuzz_seeds/tiff_compression -runs=4000
```

## Continuous fuzzing in CI

The `.github/workflows/cq_fuzz.yml` workflow builds the harnesses with
Clang + libFuzzer and runs a short smoke campaign (`-runs=4000` per harness)
on a daily schedule (06:00 UTC, skipped if there is no recent commit), on
manual `workflow_dispatch`, and on pushes / pull requests that touch `fuzz/**`,
`onnx_light_kernel_images/kernels/**`, `CMakeLists.txt`, or the workflow
itself. It is meant to catch regressions in the harnesses and obvious shallow
bugs; long-running coverage-guided campaigns should be driven by OSS-Fuzz.

## Design notes

### Why `catch (...) { return 0; }`?

Fuzz targets must never crash on *expected* errors — only on *unexpected*
ones (memory corruption, hangs, sanitizer reports). Decode failures on random
or malformed inputs are expected; catching them lets libFuzzer keep searching
for inputs that cause real bugs.

### Why `LLVMFuzzerTestOneInput`?

`LLVMFuzzerTestOneInput` is the [standard libFuzzer entry point](https://llvm.org/docs/LibFuzzer.html#fuzz-target).
Each harness defines it with `extern "C"` so libFuzzer's runtime can call it
directly without name mangling.

## Adding a new harness

1. Create `fuzz/fuzz_<name>.cc` following the pattern of an existing harness
   (single `extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t
   size)` entry point that catches every exception).
2. CMake picks the new file up automatically because the fuzzer target list is
   globbed from `fuzz/fuzz_*.cc`.
3. If the fuzzer benefits from seed inputs, add them to
   `fuzz/make_seed_corpus.cc` and, when running in CI, wire up a matching
   output directory in `.github/workflows/cq_fuzz.yml`.
