"""Python bindings for the image decoder kernels."""

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    import onnx_light

    _runtime_dll_directory = os.add_dll_directory(
        str(Path(onnx_light.__file__).resolve().parent / "onnx_py")
    )
