# Changelog

All notable changes to this project are documented in this file.

## [0.2.1] – Unreleased

### Dependencies

- Update the package version and the required onnx-light runtime to 0.2.1.
- Use the onnx-light 0.2.1 C++ release archives and Python wheels.

## [0.2.0]

### New Features

- ONNX `ImageDecoder` kernel for onnx-light, supporting BMP, TIFF, JPEG,
  JPEG2000, PNG, WebP, and PNM images.
- C++ and Python APIs for registering the image kernels.

### Documentation & CI

- Documentation with a supported-kernels reference and image-format examples.
- Tag-triggered GitHub Releases with generated release notes, C++ archives, and
  Python wheels for Linux and Windows on x86-64 and ARM64.
- Build against the onnx-light 0.2.0 C++ runtime
  ([#67](https://github.com/xadupre/onnx-light-kernel-images/pull/67)).
- Pin the Python runtime dependency to `onnx-light==0.2.0`.
