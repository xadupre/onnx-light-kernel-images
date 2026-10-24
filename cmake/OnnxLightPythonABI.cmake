# Native RuntimeContext arguments must share the runtime's nanobind type registry.
if(ONNX_LIGHT_KERNEL_IMAGES_PYTHON_STABLE_ABI STREQUAL "")
  if(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR
        "Set ONNX_LIGHT_KERNEL_IMAGES_PYTHON_STABLE_ABI to ON or OFF when cross-compiling.")
  endif()
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "from onnx_light import get_cpp_build_info; print(get_cpp_build_info()['python_stable_abi'])"
    RESULT_VARIABLE _onnx_light_abi_status
    OUTPUT_VARIABLE _onnx_light_stable_abi
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _onnx_light_abi_status EQUAL 0)
    message(FATAL_ERROR
        "Could not query the onnx-light Python binding ABI. "
        "Set ONNX_LIGHT_KERNEL_IMAGES_PYTHON_STABLE_ABI to ON or OFF.")
  endif()
else()
  set(_onnx_light_stable_abi "${ONNX_LIGHT_KERNEL_IMAGES_PYTHON_STABLE_ABI}")
endif()
string(STRIP "${_onnx_light_stable_abi}" _onnx_light_stable_abi)
string(TOUPPER "${_onnx_light_stable_abi}" _onnx_light_stable_abi)
if(NOT _onnx_light_stable_abi STREQUAL "ON" AND
   NOT _onnx_light_stable_abi STREQUAL "OFF")
  message(FATAL_ERROR
      "Invalid onnx-light Python binding ABI '${_onnx_light_stable_abi}'. "
      "Set ONNX_LIGHT_KERNEL_IMAGES_PYTHON_STABLE_ABI to ON or OFF.")
endif()
