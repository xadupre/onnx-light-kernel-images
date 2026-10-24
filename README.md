# onnx-light-kernel-images

<img src="docs/_static/logo.svg" alt="onnx-light-kernel-images logo" width="116" height="80" align="right" />

[![ci-core](https://github.com/xadupre/onnx-light-kernel-images/actions/workflows/ci_core.yml/badge.svg)](https://github.com/xadupre/onnx-light-kernel-images/actions/workflows/ci_core.yml)
[![docs](https://github.com/xadupre/onnx-light-kernel-images/actions/workflows/docs.yml/badge.svg)](https://github.com/xadupre/onnx-light-kernel-images/actions/workflows/docs.yml)
[![asan-ubsan](https://github.com/xadupre/onnx-light-kernel-images/actions/workflows/cq_asan_ubsan.yml/badge.svg)](https://github.com/xadupre/onnx-light-kernel-images/actions/workflows/cq_asan_ubsan.yml)
[![fuzz](https://github.com/xadupre/onnx-light-kernel-images/actions/workflows/cq_fuzz.yml/badge.svg)](https://github.com/xadupre/onnx-light-kernel-images/actions/workflows/cq_fuzz.yml)
[![hardening](https://github.com/xadupre/onnx-light-kernel-images/actions/workflows/cq_hardening.yml/badge.svg)](https://github.com/xadupre/onnx-light-kernel-images/actions/workflows/cq_hardening.yml)
[![codecov](https://codecov.io/gh/xadupre/onnx-light-kernel-images/branch/main/graph/badge.svg)](https://codecov.io/gh/xadupre/onnx-light-kernel-images)
[![Style](https://github.com/xadupre/onnx-light-kernel-images/actions/workflows/style.yml/badge.svg)](https://github.com/xadupre/onnx-light-kernel-images/actions/workflows/style.yml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/xadupre/onnx-light-kernel-images/badge)](https://securityscorecards.dev/viewer/?uri=github.com/xadupre/onnx-light-kernel-images)

ImageDecoder kernel extension for
[onnx-light](https://github.com/xadupre/onnx-light).

Registers the ONNX `ImageDecoder` operator (ai.onnx, since opset 20) with the
onnx-light kernel dispatch table. The kernel decodes encoded image bytestreams
into `(H, W, C)` `tensor(uint8)` arrays in channel-last layout.

## Supported formats

| Format | Variants |
|--------|----------|
| **BMP** | 24-bit uncompressed (BI_RGB, BITMAPINFOHEADER) |
| **TIFF** | Baseline 8-bit-per-sample chunky, uncompressed or compressed with PackBits, LZW or Deflate/ZIP (with optional horizontal predictor) |
| **JPEG** | Baseline JFIF (SOF0, 8-bit, 1 or 3 components) |
| **JPEG2000** | JP2 file format and raw J2K codestream (via `libopenjp2`) |
| **PNG** | 8-bit non-interlaced grayscale / truecolor |
| **WebP** | Decoded via `libwebp` |
| **PNM** | Netpbm family (P1–P6 with 8-bit samples) |

JPEG2000 and WebP are decoded through the OpenJPEG (`libopenjp2`) and
`libwebp` shared libraries, which are loaded dynamically at runtime. When the
corresponding library is not available the decoder returns an empty matrix, as
described by the ONNX `ImageDecoder` schema.

TIFF pixel conversion and WebP/JPEG2000 post-decode conversion use onnx-light's
`ParallelFor`, reusing the active CPU executor (or its shared global thread pool
outside an executor scope). Images below the default 32,768-pixel threshold and
single-thread executors run inline. Codec decompression remains serial; BMP,
JPEG, PNG and PNM decoding is delegated to onnx-light.
TIFF conversion processes contiguous row segments within each pixel range, retaining
pixel-based scheduling for wide images while computing strip addresses once per segment.

## Build from source

Prebuilt C++ archives and Python wheels are available on the
[GitHub Releases page](https://github.com/xadupre/onnx-light-kernel-images/releases).
Pushing a new tag builds and attaches binaries for Linux and Windows
(x86-64 and ARM64); wheels target Python 3.12–3.13. Release notes are generated
from the changes since the previous release. See [CHANGELOGS.md](CHANGELOGS.md)
for the project changelog.

### Prerequisites

- C++20 compiler
- CMake ≥ 3.15
- Python ≥ 3.12
- [nanobind](https://github.com/wjakob/nanobind) ≥ 1.3.2

### Python wheel (recommended)

```bash
pip install --find-links https://github.com/xadupre/onnx-light/releases/expanded_assets/0.2.1 .
```

The required `onnx-light==0.2.1` wheels are hosted on GitHub Releases, not PyPI.
Only the onnx-light 0.2.1 runtime API is supported; older API compatibility paths
are not provided.
For other pip-based install commands below, set `PIP_FIND_LINKS` to the same URL.
Python builds link to that runtime's shared libraries so kernel registration
updates the same dispatch table used by `ReferenceEvaluator`.

### Pixi environment

```bash
pixi install
pixi run install
pixi run test-python
```

### setup.py with C++ tests

Build the extension and run the C++ unit tests with `ctest`:

```bash
python setup.py build_ext --inplace --cpp-tests
```

To build and test against the exact C++ runtime loaded by a locally built,
importable sibling checkout of
[onnx-light](https://github.com/xadupre/onnx-light), use:

```bash
PYTHONPATH=../onnx-light \
python setup.py build_ext --inplace --cpp-tests --onnx-light-source
```

`--onnx-light-source` rejects mixed installations: the imported Python package,
headers, extension, and shared C++ libraries must all come from the same
onnx-light checkout.
Header discovery supports both the checkout's separate proto/helper include
directories and the header layout shipped inside an installed Python package.

### Pure CMake (C++ only)

```bash
cmake -S . -B build -DONNX_LIGHT_KERNEL_IMAGES_BUILD_TESTS=ON \
      -DONNX_LIGHT_KERNEL_IMAGES_BUILD_PYTHON=OFF
cmake --build build
ctest --test-dir build
```

The build automatically downloads the onnx-light 0.2.1 C++ release archive.
To use a custom install, set `-DONNX_LIGHT_ROOT=/path/to/onnx-light-cpp`.

### Compiler hardening

Pass `-DONNX_HARDENING=ON` to build every C++ target with the
[OpenSSF Compiler Options Hardening Guide for C and C++](https://best.openssf.org/Compiler-Hardening-Guides/Compiler-Options-Hardening-Guide-for-C-and-C++.html)
flags (for example `-D_FORTIFY_SOURCE`, `-fstack-protector-strong`,
`-fstack-clash-protection`, `-fcf-protection`, RELRO/`now` linking on
GCC/Clang and `/GS`, `/guard:cf`, `/Qspectre`, `/DYNAMICBASE` on MSVC). Each
flag is probed and silently skipped when the active toolchain does not support
it. With a pip build the option is forwarded via
`-C cmake.define.ONNX_HARDENING=ON`.

> **macOS:** onnx-light publishes prebuilt C++ archives only for Linux and
> Windows. On macOS, build and install the onnx-light C++ SDK from source
> (`-DONNX_LIGHT_BUILD_PYTHON=OFF`) and point the build at it with
> `-DONNX_LIGHT_ROOT=/path/to/onnx-light-install`.

## C++ usage

```cpp
#include <onnx_light_kernel_images/register_image_kernels.h>

int main() {
    // Register the ImageDecoder kernel once before running models.
    onnx_light_kernel_images::RegisterImageKernels();
    // ... use onnx-light RuntimeSession with ImageDecoder nodes ...
}
```

Link against `onnx_light_kernel_images::lib_onnx_light_kernel_images`:

```cmake
find_package(onnx_light_kernel_images REQUIRED)
target_link_libraries(my_app PRIVATE
    onnx_light_kernel_images::lib_onnx_light_kernel_images)
```

To register only for one runtime session, pass the `RuntimeContext` used by
that session to `onnx_light_kernel_images::RegisterImageKernels(context)` before
its first run. This does not modify the global dispatch table or other contexts;
subgraphs inherit the local registration.

## Python usage

```python
from onnx_light_kernel_images.onnx_py._imgpykernels import register_image_kernels

# Register the kernel once before running models.
register_image_kernels()
```

For session-only registration, pass the evaluator instead:

```python
from onnx_light.onnx.reference import ReferenceEvaluator

session = ReferenceEvaluator(model)
register_image_kernels(session)
image = session.run(None, {"encoded": encoded})[0]
```

`register_image_kernels(context)` also accepts a low-level `RuntimeContext`.
Register before the first run: already-resolved kernels are not replaced.
Repeated calls replace the local `ImageDecoder` registration used for future
kernel resolutions. Kernel instances already resolved by existing nodes remain
unchanged.
Calling without an argument (or with `None`) registers globally (idempotently),
updating the global dispatch table.

Python builds detect the binding ABI from onnx-light's `get_cpp_build_info()`.
If that runtime does not report `python_stable_abi`, configuration fails rather
than guessing. Set `-DONNX_LIGHT_KERNEL_IMAGES_PYTHON_STABLE_ABI=ON` or `OFF`
to match the target runtime explicitly; this is required when cross-compiling.

Once the kernels are registered, build an ``ImageDecoder`` ONNX model and run it
through onnx-light's ``ReferenceEvaluator``, feeding the encoded bytestream as a
``uint8`` tensor and reading back the decoded ``(H, W, C)`` ``uint8`` image.

See the [examples gallery](docs/auto_examples/index.html) for a script that
saves an image in every supported format with Pillow and reloads each one
through such a model.

## Testing

### C++ tests

```bash
cmake -S . -B build -DONNX_LIGHT_KERNEL_IMAGES_BUILD_TESTS=ON \
      -DONNX_LIGHT_KERNEL_IMAGES_BUILD_PYTHON=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

### Python tests

```bash
pip install -e .
pytest unittests/python/
```

### Fuzzing

libFuzzer harnesses under [`fuzz/`](fuzz/) exercise the image decoders and the
custom TIFF-compression front-end with random / malformed inputs. They require
Clang:

```bash
CC=clang CXX=clang++ cmake -S . -B build-fuzz -G Ninja \
      -DONNX_LIGHT_KERNEL_IMAGES_BUILD_FUZZERS=ON \
      -DONNX_LIGHT_KERNEL_IMAGES_BUILD_PYTHON=OFF \
      -DONNX_LIGHT_KERNEL_IMAGES_INSTALL=OFF
cmake --build build-fuzz
./build-fuzz/fuzz_image_decoder -runs=4000
./build-fuzz/fuzz_tiff_compression -runs=4000
```

See [`fuzz/README.md`](fuzz/README.md) for details.

## License

Apache-2.0. See [LICENSE](LICENSE).
