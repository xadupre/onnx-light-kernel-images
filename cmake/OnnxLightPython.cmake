# Python builds must share the installed runtime's kernel registry, rather than
# linking a second copy from the standalone static C++ SDK.
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
    "from onnx_light import get_cpp_build_info; i = get_cpp_build_info(); print(';'.join(i[k].replace('\\\\', '/') for k in ('include_dir', 'core_library', 'proto_library', 'library_dir')))"
  RESULT_VARIABLE _onnx_light_info_status
  OUTPUT_VARIABLE _onnx_light_info
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _onnx_light_info_status EQUAL 0)
  message(FATAL_ERROR "Install onnx-light==0.2.3 before building the Python extension.")
endif()
list(GET _onnx_light_info 0 _onnx_light_include)
list(GET _onnx_light_info 1 ONNX_LIGHT_KERNEL_IMAGES_ONNX_LIGHT_LIBRARY)
list(GET _onnx_light_info 2 ONNX_LIGHT_KERNEL_IMAGES_ONNX_LIGHT_PROTO_LIBRARY)
list(GET _onnx_light_info 3 _onnx_light_runtime_dir)
get_filename_component(ONNX_LIGHT_KERNEL_IMAGES_ONNX_LIGHT_SOURCE_DIR
    "${_onnx_light_include}" DIRECTORY)

# Windows wheels ship DLLs without import libraries. Recreate those libraries
# from the DLL exports with the MSVC tools already used for the extension build.
if(WIN32)
  get_filename_component(_onnx_light_msvc_bin "${CMAKE_LINKER}" DIRECTORY)
  find_program(_onnx_light_dumpbin dumpbin HINTS "${_onnx_light_msvc_bin}")
  if(NOT _onnx_light_dumpbin)
    message(FATAL_ERROR "dumpbin is required to link the onnx-light Python runtime.")
  endif()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|arm64|aarch64")
    set(_onnx_light_machine ARM64)
  else()
    set(_onnx_light_machine X64)
  endif()
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/onnx_light_imports")
  foreach(_component IN ITEMS core proto kernels backend_test threads)
    set(_dll "${_onnx_light_runtime_dir}/lib_onnx_${_component}.dll")
    if(NOT EXISTS "${_dll}")
      message(FATAL_ERROR "Missing onnx-light runtime DLL: ${_dll}")
    endif()
    execute_process(
      COMMAND "${_onnx_light_dumpbin}" /exports "${_dll}"
      RESULT_VARIABLE _dump_status
      OUTPUT_VARIABLE _exports
    )
    if(NOT _dump_status EQUAL 0)
      message(FATAL_ERROR "Could not read exports from ${_dll}.")
    endif()
    string(REGEX MATCHALL
        "\n[ ]+[0-9]+[ ]+[0-9A-Fa-f]+[ ]+[0-9A-Fa-f]+[ ]+[^ \r\n]+"
        _export_lines "${_exports}")
    set(_def_contents "LIBRARY lib_onnx_${_component}.dll\nEXPORTS\n")
    foreach(_line IN LISTS _export_lines)
      string(REGEX REPLACE
          "^[\n ]+[0-9]+[ ]+[0-9A-Fa-f]+[ ]+[0-9A-Fa-f]+[ ]+" ""
          _symbol "${_line}")
      string(APPEND _def_contents "${_symbol}\n")
    endforeach()
    if(NOT _export_lines)
      message(FATAL_ERROR "No exports found in ${_dll}.")
    endif()
    set(_base "${CMAKE_CURRENT_BINARY_DIR}/onnx_light_imports/lib_onnx_${_component}")
    file(WRITE "${_base}.def" "${_def_contents}")
    execute_process(
      COMMAND "${CMAKE_AR}" /nologo "/machine:${_onnx_light_machine}"
          "/def:${_base}.def" "/out:${_base}.lib"
      RESULT_VARIABLE _implib_status
    )
    if(NOT _implib_status EQUAL 0)
      message(FATAL_ERROR "Could not create an import library for ${_dll}.")
    endif()
    if(_component STREQUAL "core")
      set(ONNX_LIGHT_KERNEL_IMAGES_ONNX_LIGHT_IMPLIB "${_base}.lib")
    else()
      string(TOUPPER "${_component}" _component_upper)
      set(ONNX_LIGHT_KERNEL_IMAGES_ONNX_LIGHT_${_component_upper}_IMPLIB "${_base}.lib")
    endif()
  endforeach()
endif()
