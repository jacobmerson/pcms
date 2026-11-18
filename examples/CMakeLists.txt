# PCMS examples.
#
# This directory is a CMake project of its own and is also part of PCMS's
# build:
#
#   in tree      PCMS_ENABLE_EXAMPLES=ON (the default) adds it to the PCMS
#                build; the examples are registered as ctests and run with
#                `ctest -L example`.
#   standalone   cmake -S examples -B build -Dpcms_DIR=<prefix>/lib/cmake/pcms
#                builds them against an installed PCMS with find_package(pcms),
#                the way a user's project does; this is the install test of CI.
#                Python examples are tested when `import pcms` works; add the
#                location of the module (and of PyOmega_h) to
#                PCMS_EXAMPLES_PYTHONPATH when it is not on PYTHONPATH.
#
# An example whose dependencies are missing is skipped at configure time, or
# stops the configuration when the environment variable PCMS_REQUIRE_EXAMPLES
# is 1.

cmake_minimum_required(VERSION 3.23)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(pcms_examples CXX)
  find_package(pcms REQUIRED)
  enable_testing()
  set(PCMS_EXAMPLES_STANDALONE ON)
else()
  set(PCMS_EXAMPLES_STANDALONE OFF)
endif()

set(PCMS_EXAMPLES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PcmsExamples)

add_subdirectory(fields)

# The examples may only use the public API: a test that scans their sources.
# It lives in the PCMS source tree, so it is part of the in-tree build only.
if(NOT PCMS_EXAMPLES_STANDALONE
   AND EXISTS "${PROJECT_SOURCE_DIR}/tools/check-examples-public-api.py"
   AND PCMS_EXAMPLES_PYTHON_EXECUTABLE)
  add_test(
    NAME example.public-api-gate
    COMMAND
      ${PCMS_EXAMPLES_PYTHON_EXECUTABLE}
      "${PROJECT_SOURCE_DIR}/tools/check-examples-public-api.py"
      --include-dir "${PROJECT_SOURCE_DIR}/include" "${PCMS_EXAMPLES_ROOT}")
  set_tests_properties(example.public-api-gate PROPERTIES LABELS example)
endif()
