API Reference
=============

C++ API
-------

.. doxygenfunction:: onnx_light_kernel_images::RegisterImageKernels
   :project: onnx_light_kernel_images

Python API
----------

.. py:module:: onnx_light_kernel_images.onnx_py._imgpykernels

.. py:function:: register_image_kernels()

   Registers the ``ImageDecoder`` kernel (ai.onnx domain) with the onnx-light
   kernel dispatch table.

   Supported formats: BMP, TIFF, JPEG, JPEG2000, PNG, WebP, PNM. JPEG2000 and
   WebP are decoded through the ``libopenjp2`` and ``libwebp`` shared
   libraries, loaded dynamically at runtime.

   Idempotent: calling more than once is safe and cheap.

.. py:function:: has_image_kernels() -> bool

   Returns ``True`` when the image kernel extension is available.

.. py:function:: decode_image(data: bytes, pixel_format: str = "RGB")

   Decodes an encoded image bytestream with the ``ImageDecoder`` kernel and
   returns the pixels as a channel-last ``(H, W, C)`` ``uint8`` NumPy array.

   ``data`` holds the encoded image file contents (BMP, TIFF, JPEG, JPEG2000,
   PNG, WebP or PNM). ``pixel_format`` selects the channel layout and is one of
   ``"RGB"`` (default), ``"BGR"`` or ``"Grayscale"``.

   Compressed TIFF inputs (PackBits, LZW, Deflate/ZIP) are rewritten to an
   uncompressed baseline TIFF before decoding. Inputs that cannot be decoded
   return an empty ``(0, 0, C)`` array, as described by the ONNX
   ``ImageDecoder`` schema.
