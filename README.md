# onnx-light-kernel-images

[![ci-core](https://github.com/xadupre/onnx-light-kernel-images/actions/workflows/ci_core.yml/badge.svg)](https://github.com/xadupre/onnx-light-kernel-images/actions/workflows/ci_core.yml)
[![codecov](https://codecov.io/gh/xadupre/onnx-light-kernel-images/branch/main/graph/badge.svg)](https://codecov.io/gh/xadupre/onnx-light-kernel-images)
[![Style](https://github.com/xadupre/onnx-light-kernel-images/actions/workflows/style.yml/badge.svg)](https://github.com/xadupre/onnx-light-kernel-images/actions/workflows/style.yml)

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

## Build from source

### Prerequisites

- C++20 compiler
- CMake ≥ 3.15
- Python ≥ 3.10
- [nanobind](https://github.com/wjakob/nanobind) ≥ 1.3.2

### Python wheel (recommended)

```bash
pip install .
```

### Pixi environment

```bash
pixi install
pixi run install
pixi run test-python
```

### setup.py with C++ tests

```bash
python setup.py build_ext --inplace --cpp-tests
```

### Pure CMake (C++ only)

```bash
cmake -S . -B build -DONNX_LIGHT_KERNEL_IMAGES_BUILD_TESTS=ON \
      -DONNX_LIGHT_KERNEL_IMAGES_BUILD_PYTHON=OFF
cmake --build build
ctest --test-dir build
```

The build automatically downloads the onnx-light 0.1.9 C++ release archive.
To use a custom install, set `-DONNX_LIGHT_ROOT=/path/to/onnx-light-cpp`.

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

## Python usage

```python
from onnx_light_kernel_images.onnx_py._imgpykernels import (
    decode_image,
    register_image_kernels,
)

# Register the kernel once before running models.
register_image_kernels()

# Or decode an encoded image bytestream directly. Returns the (H, W, C) shape
# and the raw row-major uint8 pixel bytes. Supported pixel formats are "RGB"
# (default), "BGR" and "Grayscale".
with open("image.bmp", "rb") as f:
    shape, pixels = decode_image(f.read(), "RGB")
```

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

## License

Apache-2.0. See [LICENSE](LICENSE).
