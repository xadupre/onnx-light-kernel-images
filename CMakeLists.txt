cmake_minimum_required(VERSION 3.15)

# Default to a Release build when no build type is specified.
get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _is_multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

project(onnx_light_kernel_images VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(ONNX_LIGHT_KERNEL_IMAGES_BUILD_TESTS "Build C++ unit tests." OFF)
option(ONNX_LIGHT_KERNEL_IMAGES_BUILD_PYTHON "Build Python extension." ON)
option(ONNX_LIGHT_KERNEL_IMAGES_INSTALL "Install the C++ library and headers." ON)
option(ONNX_LIGHT_KERNEL_IMAGES_ENABLE_COVERAGE
       "Instrument the C++ targets for code coverage (GCC/Clang only)." OFF)
option(ONNX_LIGHT_KERNEL_IMAGES_BUILD_FUZZERS
       "Build libFuzzer-instrumented C++ fuzz harnesses (fuzz/fuzz_*.cc). Requires Clang and adds -fsanitize=fuzzer,address."
       OFF)
set(ONNX_LIGHT_KERNEL_IMAGES_FUZZER_SANITIZERS "address"
    CACHE STRING
    "Comma-separated sanitizer list for fuzz harnesses (in addition to 'fuzzer').")

if(ONNX_LIGHT_KERNEL_IMAGES_ENABLE_COVERAGE AND
   NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(FATAL_ERROR
      "ONNX_LIGHT_KERNEL_IMAGES_ENABLE_COVERAGE requires a GCC or Clang "
      "compiler; got '${CMAKE_CXX_COMPILER_ID}'.")
endif()

if(MSVC)
  if(NOT CMAKE_GENERATOR MATCHES "Ninja")
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/MP>)
  endif()
endif()

include(GNUInstallDirs)
include(FetchContent)

# ---------------------------------------------------------------------------
# Locate onnx-light C++ SDK (0.1.9 release)
#
# Strategy:
#   1. If ONNX_LIGHT_ROOT is set (or the package is already installed), use it.
#   2. Otherwise, download the prebuilt C++ archive from the GitHub release.
# ---------------------------------------------------------------------------
if(NOT DEFINED ONNX_LIGHT_ROOT)
  # Auto-detect platform archive name.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
      set(_onnx_light_archive "onnx-light-cpp-linux-aarch64.tar.gz")
    else()
      set(_onnx_light_archive "onnx-light-cpp-linux-x86_64.tar.gz")
    endif()
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64")
      set(_onnx_light_archive "onnx-light-cpp-windows-ARM64.zip")
    else()
      set(_onnx_light_archive "onnx-light-cpp-windows-AMD64.zip")
    endif()
  else()
    message(FATAL_ERROR
        "No prebuilt onnx-light C++ archive for ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}. "
        "Set ONNX_LIGHT_ROOT to the path of a locally built onnx-light install tree.")
  endif()

  set(_onnx_light_url
      "https://github.com/xadupre/onnx-light/releases/download/0.1.9/${_onnx_light_archive}")

  FetchContent_Declare(
    onnx_light_cpp
    URL "${_onnx_light_url}"
    DOWNLOAD_EXTRACT_TIMESTAMP ON
  )
  FetchContent_MakeAvailable(onnx_light_cpp)

  # -------------------------------------------------------------------------
  # Work around a packaging quirk in the onnx-light 0.1.9 prebuilt archives:
  # include/onnx_light/onnx_proto/ ships byte-identical copies of the headers
  # in include/onnx_light/ (onnx.h, fields.h, stream*.h, ...), each guarded
  # only by `#pragma once`. MSVC resolves a quoted `#include "onnx.h"` against
  # the directories of every header on the current include stack, so a single
  # translation unit can pull in *both* physical copies and fail with C2011
  # "class type redefinition" (GCC/Clang resolve to a single copy and are
  # unaffected). Replace each duplicated onnx_proto/ header with a thin
  # forwarding include so every include path converges on one physical file.
  # Only the archive we downloaded ourselves is patched; a user-supplied
  # ONNX_LIGHT_ROOT is never modified. The SHA256 guard keeps this idempotent
  # and skips headers that legitimately differ from their parent copy.
  # -------------------------------------------------------------------------
  set(_onnx_light_proto_dir
      "${onnx_light_cpp_SOURCE_DIR}/include/onnx_light/onnx_proto")
  if(IS_DIRECTORY "${_onnx_light_proto_dir}")
    file(GLOB _onnx_light_proto_headers
         "${_onnx_light_proto_dir}/*.h" "${_onnx_light_proto_dir}/*.hpp")
    foreach(_onnx_light_hdr IN LISTS _onnx_light_proto_headers)
      get_filename_component(_onnx_light_hdr_name "${_onnx_light_hdr}" NAME)
      set(_onnx_light_parent_hdr
          "${onnx_light_cpp_SOURCE_DIR}/include/onnx_light/${_onnx_light_hdr_name}")
      if(EXISTS "${_onnx_light_parent_hdr}")
        file(SHA256 "${_onnx_light_hdr}" _onnx_light_hash_proto)
        file(SHA256 "${_onnx_light_parent_hdr}" _onnx_light_hash_parent)
        if(_onnx_light_hash_proto STREQUAL _onnx_light_hash_parent)
          file(WRITE "${_onnx_light_hdr}"
"#pragma once\n// Forwarding stub written by onnx-light-kernel-images to collapse the\n// duplicated onnx_proto/ header mirror onto a single physical file so MSVC\n// does not hit `#pragma once` class redefinitions (see CMakeLists.txt).\n#include \"../${_onnx_light_hdr_name}\"\n")
        endif()
      endif()
    endforeach()
  endif()

  set(ONNX_LIGHT_ROOT "${onnx_light_cpp_SOURCE_DIR}")
endif()

# Find the installed onnx-light CMake package.
list(APPEND CMAKE_PREFIX_PATH "${ONNX_LIGHT_ROOT}")
find_package(onnx_light REQUIRED)

# Find optional dependencies that onnx-light's imported targets need.
find_package(OpenMP QUIET)
find_package(OpenSSL QUIET)

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
set(KERNEL_IMAGES_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_kernel_images/kernels/image/register_image_kernels.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_kernel_images/kernels/image/tiff_compression.cc"
)

set(PY_KERNELS_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_kernel_images/onnx_py/_imgpy_kernels.cc"
)

# ---------------------------------------------------------------------------
# C++ library: lib_onnx_light_kernel_images
# ---------------------------------------------------------------------------
if(ONNX_LIGHT_KERNEL_IMAGES_BUILD_PYTHON AND UNIX AND NOT APPLE)
  set(_img_lib_type SHARED)
else()
  set(_img_lib_type STATIC)
endif()

add_library(lib_onnx_light_kernel_images ${_img_lib_type} ${KERNEL_IMAGES_SOURCES})
target_include_directories(lib_onnx_light_kernel_images PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
set_property(TARGET lib_onnx_light_kernel_images PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(lib_onnx_light_kernel_images PUBLIC cxx_std_20)
target_link_libraries(lib_onnx_light_kernel_images PUBLIC
    onnx_light::lib_onnx_kernels
    onnx_light::lib_onnx_core
    onnx_light::lib_onnx_proto
)

# Coverage instrumentation is applied PUBLIC so it propagates to the C++ test
# executable that links this library, letting gcov emit .gcno/.gcda data for
# both the kernel sources and the tests exercising them.
if(ONNX_LIGHT_KERNEL_IMAGES_ENABLE_COVERAGE)
  target_compile_options(lib_onnx_light_kernel_images PUBLIC
      --coverage -O0 -g)
  target_link_options(lib_onnx_light_kernel_images PUBLIC --coverage)
endif()

# When building the fuzz harnesses, instrument this repository's own kernel
# sources (in particular the TIFF-compression rewriter) with libFuzzer edge
# coverage and the requested sanitizers so the fuzzer actually explores the
# custom decode path. ``fuzzer-no-link`` adds coverage without pulling in the
# libFuzzer main, which lives in the harness executables instead. The onnx-light
# SDK itself is consumed as a prebuilt archive and therefore cannot be
# instrumented here.
if(ONNX_LIGHT_KERNEL_IMAGES_BUILD_FUZZERS)
  target_compile_options(lib_onnx_light_kernel_images PUBLIC
      "-fsanitize=fuzzer-no-link,${ONNX_LIGHT_KERNEL_IMAGES_FUZZER_SANITIZERS}")
endif()

# ---------------------------------------------------------------------------
# Python extension: _imgpykernels
# ---------------------------------------------------------------------------
if(ONNX_LIGHT_KERNEL_IMAGES_BUILD_PYTHON)
  if(CMAKE_CROSSCOMPILING)
    find_package(Python3 3.10 REQUIRED COMPONENTS Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter)
    if(NOT TARGET Python::Module)
      add_library(Python::Module ALIAS Python3::Module)
    endif()
    if(TARGET Python3::SABIModule AND NOT TARGET Python::SABIModule)
      add_library(Python::SABIModule ALIAS Python3::SABIModule)
    endif()
  else()
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule)
  endif()

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NB_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(_imgpykernels STABLE_ABI ${PY_KERNELS_SOURCES})
  target_include_directories(_imgpykernels PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(_imgpykernels PRIVATE lib_onnx_light_kernel_images)

  # Set RPATH so the extension finds liblib_onnx_light_kernel_images.so
  # in the same directory at runtime.
  set_target_properties(_imgpykernels PROPERTIES
      INSTALL_RPATH "$ORIGIN"
      BUILD_RPATH "$ORIGIN"
  )

  install(TARGETS _imgpykernels
      LIBRARY DESTINATION onnx_light_kernel_images/onnx_py
      RUNTIME DESTINATION onnx_light_kernel_images/onnx_py)

  # Copy the onnx-light shared libraries next to the Python extension so
  # the dynamic linker can find them at import time.
  if(_img_lib_type STREQUAL "SHARED")
    install(TARGETS lib_onnx_light_kernel_images
        LIBRARY DESTINATION onnx_light_kernel_images/onnx_py
        RUNTIME DESTINATION onnx_light_kernel_images/onnx_py)
  endif()
endif()

# ---------------------------------------------------------------------------
# C++ install
# ---------------------------------------------------------------------------
if(ONNX_LIGHT_KERNEL_IMAGES_INSTALL)
  install(TARGETS lib_onnx_light_kernel_images
      EXPORT onnx_light_kernel_imagesTargets
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  install(FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_kernel_images/register_image_kernels.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light_kernel_images/tiff_compression.h"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light_kernel_images")

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
      "${CMAKE_CURRENT_BINARY_DIR}/onnx_light_kernel_imagesConfigVersion.cmake"
      COMPATIBILITY SameMajorVersion)

  configure_package_config_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnx_light_kernel_imagesConfig.cmake.in"
      "${CMAKE_CURRENT_BINARY_DIR}/onnx_light_kernel_imagesConfig.cmake"
      INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnx_light_kernel_images")

  install(EXPORT onnx_light_kernel_imagesTargets
      NAMESPACE onnx_light_kernel_images::
      DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnx_light_kernel_images")

  install(FILES
      "${CMAKE_CURRENT_BINARY_DIR}/onnx_light_kernel_imagesConfig.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/onnx_light_kernel_imagesConfigVersion.cmake"
      DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnx_light_kernel_images")
endif()

# ---------------------------------------------------------------------------
# C++ unit tests (GTest)
# ---------------------------------------------------------------------------
if(ONNX_LIGHT_KERNEL_IMAGES_BUILD_TESTS)
  enable_testing()

  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP ON
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(test_image_decoder
      "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/test_image_decoder.cc")
  target_link_libraries(test_image_decoder PRIVATE
      lib_onnx_light_kernel_images
      onnx_light::lib_onnx_kernels
      onnx_light::lib_onnx_core
      onnx_light::lib_onnx_proto
      GTest::gtest_main)
  target_include_directories(test_image_decoder PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}")

  include(GoogleTest)
  gtest_discover_tests(test_image_decoder)
endif()

# ---------------------------------------------------------------------------
# libFuzzer harnesses (fuzz/fuzz_*.cc)
#
# Each fuzz/fuzz_<name>.cc file exposes the standard libFuzzer entry point
# ``LLVMFuzzerTestOneInput``. The companion fuzz/make_seed_corpus.cc is built
# as a plain executable (no fuzzer instrumentation) and emits seed inputs for
# OSS-Fuzz / local libFuzzer runs.
#
# Requires Clang because libFuzzer (-fsanitize=fuzzer,...) ships with Clang.
# ``-fsanitize=fuzzer,address`` is the standard OSS-Fuzz instrumentation; set
# ``ONNX_LIGHT_KERNEL_IMAGES_FUZZER_SANITIZERS`` to override the sanitizer list
# (the ``fuzzer`` sanitizer is always added automatically).
# ---------------------------------------------------------------------------
if(ONNX_LIGHT_KERNEL_IMAGES_BUILD_FUZZERS)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
        "ONNX_LIGHT_KERNEL_IMAGES_BUILD_FUZZERS=ON requires Clang "
        "(CMAKE_CXX_COMPILER_ID is '${CMAKE_CXX_COMPILER_ID}'). "
        "Re-run cmake with CC=clang CXX=clang++.")
  endif()

  file(GLOB _img_fuzz_sources CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/fuzz/fuzz_*.cc")

  foreach(_img_fuzz_source IN LISTS _img_fuzz_sources)
    get_filename_component(_img_fuzz_target "${_img_fuzz_source}" NAME_WE)
    add_executable(${_img_fuzz_target} "${_img_fuzz_source}")
    target_include_directories(${_img_fuzz_target} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${_img_fuzz_target} PRIVATE
        lib_onnx_light_kernel_images
        onnx_light::lib_onnx_kernels
        onnx_light::lib_onnx_core
        onnx_light::lib_onnx_proto)
    target_compile_options(${_img_fuzz_target} PRIVATE
        "-fsanitize=fuzzer-no-link,${ONNX_LIGHT_KERNEL_IMAGES_FUZZER_SANITIZERS}")
    target_link_options(${_img_fuzz_target} PRIVATE
        "-fsanitize=fuzzer,${ONNX_LIGHT_KERNEL_IMAGES_FUZZER_SANITIZERS}")
    set_property(TARGET ${_img_fuzz_target} PROPERTY FOLDER "Fuzzers")
  endforeach()

  # Seed-corpus generator: a plain (non-fuzzer-instrumented) executable that
  # writes valid sample images into the requested directories so OSS-Fuzz /
  # libFuzzer can use them as starting inputs.
  add_executable(make_seed_corpus
      "${CMAKE_CURRENT_SOURCE_DIR}/fuzz/make_seed_corpus.cc")
  target_compile_features(make_seed_corpus PRIVATE cxx_std_20)
  set_property(TARGET make_seed_corpus PROPERTY FOLDER "Fuzzers")
endif()
