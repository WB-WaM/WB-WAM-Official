# ~~~
# Copyright 2021 Olivier Le Doeuff
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# This module defines the following variables:
#
# - TensorRT_FOUND: A boolean specifying whether or not TensorRT was found.
# - TensorRT_VERSION: The exact version of TensorRT found
# - TensorRT_VERSION_MAJOR: The major version of TensorRT.
# - TensorRT_VERSION_MINOR: The minor version of TensorRT.
# - TensorRT_VERSION_PATCH: The patch version of TensorRT.
# - TensorRT_VERSION_TWEAK: The tweak version of TensorRT.
# - TensorRT_INCLUDE_DIRS: The path to TensorRT ``include`` folder containing the header files    required to compile a project linking against TensorRT.
# - TensorRT_LIBRARY_DIRS: The path to TensorRT library directory that contains libraries.
#
# This module create following targets:
# - trt::nvinfer
# - trt::nvinfer_plugin
# - trt::nvonnxparser
# - trt::nvparsers
# This script was inspired from https://github.com/NicolasIRAGNE/CMakeScripts
# This script was inspired from https://github.com/NVIDIA/tensorrt-laboratory/blob/master/cmake/FindTensorRT.cmake
#
# Hints
# ^^^^^
# A user may set ``TensorRT_ROOT`` to an installation root to tell this module where to look.
# ~~~

if(NOT TensorRT_FIND_COMPONENTS)
  set(TensorRT_FIND_COMPONENTS nvinfer nvinfer_plugin nvonnxparser nvparsers)
endif()
set(TensorRT_LIBRARIES)

# Build a list of likely TensorRT installation roots so `just build` can work
# without requiring callers to manually export TensorRT_ROOT first.
set(_TensorRT_SEARCH_ROOTS)

if(DEFINED TensorRT_ROOT AND TensorRT_ROOT)
  list(APPEND _TensorRT_SEARCH_ROOTS "${TensorRT_ROOT}")
endif()

if(DEFINED ENV{TensorRT_ROOT} AND NOT "$ENV{TensorRT_ROOT}" STREQUAL "")
  list(APPEND _TensorRT_SEARCH_ROOTS "$ENV{TensorRT_ROOT}")
endif()

file(GLOB _TensorRT_AUTO_CANDIDATES
  LIST_DIRECTORIES true
  "${CMAKE_SOURCE_DIR}/../TensorRT*"
  "${CMAKE_SOURCE_DIR}/../../TensorRT*"
  "$ENV{HOME}/TensorRT*"
  "/opt/TensorRT*"
  "/usr/local/TensorRT*"
)

foreach(_candidate IN LISTS _TensorRT_AUTO_CANDIDATES)
  if(IS_DIRECTORY "${_candidate}" AND EXISTS "${_candidate}/include/NvInfer.h")
    list(APPEND _TensorRT_SEARCH_ROOTS "${_candidate}")
  endif()
endforeach()

list(REMOVE_DUPLICATES _TensorRT_SEARCH_ROOTS)

# find the include directory of TensorRT
find_path(
  TensorRT_INCLUDE_DIR
  NAMES NvInfer.h
  PATHS ${_TensorRT_SEARCH_ROOTS}
  PATH_SUFFIXES include
)

string(FIND ${TensorRT_INCLUDE_DIR} "NOTFOUND" _include_dir_notfound)
if(NOT _include_dir_notfound EQUAL -1)
  if(TensorRT_FIND_REQUIRED)
    message(FATAL_ERROR "Fail to find TensorRT, please set TensorRT_ROOT. Include path not found.")
  endif()
  return()
endif()
set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})

if((NOT DEFINED TensorRT_ROOT OR NOT TensorRT_ROOT) AND TensorRT_INCLUDE_DIR)
  get_filename_component(TensorRT_ROOT "${TensorRT_INCLUDE_DIR}" DIRECTORY)
  message(STATUS "Auto-detected TensorRT root: ${TensorRT_ROOT}")
endif()

# Extract version of tensorrt
if(EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MAJOR REGEX "^#define NV_TENSORRT_MAJOR [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MINOR REGEX "^#define NV_TENSORRT_MINOR [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_PATCH REGEX "^#define NV_TENSORRT_PATCH [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_TWEAK REGEX "^#define NV_TENSORRT_BUILD [0-9]+.*$")

  string(REGEX REPLACE "^#define NV_TENSORRT_MAJOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MAJOR "${TensorRT_MAJOR}")
  string(REGEX REPLACE "^#define NV_TENSORRT_MINOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MINOR "${TensorRT_MINOR}")
  string(REGEX REPLACE "^#define NV_TENSORRT_PATCH ([0-9]+).*$" "\\1" TensorRT_VERSION_PATCH "${TensorRT_PATCH}")
  string(REGEX REPLACE "^#define NV_TENSORRT_BUILD ([0-9]+).*$" "\\1" TensorRT_VERSION_TWEAK "${TensorRT_TWEAK}")
  set(TensorRT_VERSION "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}.${TensorRT_VERSION_TWEAK}")
endif()

function(_find_trt_component component)

  # Find library for component (ie nvinfer, nvparsers, etc...)
  find_library(
    TensorRT_${component}_LIBRARY
    NAMES ${component}
    PATHS ${_TensorRT_SEARCH_ROOTS} ${TENSORRT_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64
  )

  string(FIND ${TensorRT_${component}_LIBRARY} "NOTFOUND" _library_not_found)

  if(NOT TensorRT_LIBRARY_DIR)
    get_filename_component(_path ${TensorRT_${component}_LIBRARY} DIRECTORY)
    set(TensorRT_LIBRARY_DIR
        "${_path}"
        CACHE INTERNAL "TensorRT_LIBRARY_DIR"
    )
  endif()

  if(NOT TensorRT_LIBRARY_DIRS)
    get_filename_component(_path ${TensorRT_${component}_LIBRARY} DIRECTORY)
    set(TensorRT_LIBRARY_DIRS
        "${_path}"
        CACHE INTERNAL "TensorRT_LIBRARY_DIRS"
    )
  endif()

  # Library found, and doesn't already exists
  if(_library_not_found EQUAL -1 AND NOT TARGET trt::${component})
    set(TensorRT_${component}_FOUND
        TRUE
        CACHE INTERNAL "Found ${component}"
    )

    # Create a target
    add_library(trt::${component} IMPORTED INTERFACE)
    target_include_directories(trt::${component} SYSTEM INTERFACE "${TensorRT_INCLUDE_DIRS}")
    target_link_libraries(trt::${component} INTERFACE "${TensorRT_${component}_LIBRARY}")
    set(TensorRT_LIBRARIES ${TensorRT_LIBRARIES} ${TensorRT_${component}_LIBRARY})
  endif()

endfunction()

# Find each components
foreach(component IN LISTS TensorRT_FIND_COMPONENTS)
  _find_trt_component(${component})
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT HANDLE_COMPONENTS VERSION_VAR TensorRT_VERSION REQUIRED_VARS TensorRT_INCLUDE_DIR)

