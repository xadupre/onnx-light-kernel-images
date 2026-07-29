Available Kernels
=================

The table below lists the kernels registered by this repository with the
onnx-light kernel dispatch table. It is generated automatically at
documentation build time by scanning the C++ registration sources, so it always
reflects the kernels the repository actually provides.

.. onnx-light-kernels::

Each kernel is registered for the given operator, ONNX domain and device. The
``ImageDecoder`` kernel decodes encoded image bytestreams into ``(H, W, C)``
``tensor(uint8)`` arrays and supports the BMP, TIFF, JPEG, PNG and PNM formats.
