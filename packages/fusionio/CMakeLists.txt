cmake_minimum_required(VERSION 3.23)
project(fusionio2 VERSION 2.0.0 LANGUAGES CXX)

# Fusion-IO 2.0 is built on PCMS and uses only its installed (public) headers.
# It can be built against an installed PCMS or added to a PCMS build with
# add_subdirectory.
if(NOT TARGET pcms::pcms)
  find_package(pcms CONFIG REQUIRED)
endif()
# Components, as in PCMS: each is an option; one whose prerequisite is missing
# is an error naming it, and every dependency is found once, here.
#
#   WRAPPED  Fusion-IO 1.x reads the files and evaluates the fields until PCMS
#            evaluates them natively. It is the CMake package fusionio; this
#            package installs as fusionio2 so that both can live in one prefix.
#            Needs Fusion-IO 1.x.
#   NATIVE   the M3D-C1 evaluator on PCMS's Omega_h geometry. Needs WRAPPED
#            (1.x's M3D-C1 file loader reads the mesh and coefficients) and a
#            PCMS with PCMS_ENABLE_OMEGA_H.
#   NIMROD   NIMROD dumps through nimeval. Needs NIMROD's nimeval library.
option(FUSIONIO_ENABLE_WRAPPED "Read and evaluate files with Fusion-IO 1.x" ON)
if(FUSIONIO_ENABLE_WRAPPED)
  find_package(fusionio CONFIG QUIET)
  if(NOT fusionio_FOUND)
    message(FATAL_ERROR "FUSIONIO_ENABLE_WRAPPED needs Fusion-IO 1.x (the "
                        "CMake package fusionio; set fusionio_DIR)")
  endif()
  message(STATUS "Found Fusion-IO 1.x: ${fusionio_DIR}")
endif()

option(FUSIONIO_ENABLE_NIMROD "Evaluate NIMROD dumps with NIMROD's nimeval" OFF)
if(FUSIONIO_ENABLE_NIMROD)
  # NIMROD's C-callable evaluator (nimcore/eval_field.F90), the library nimpy
  # wraps. It carries its own RPATH to its dependencies.
  find_library(NIMROD_NIMEVAL_LIBRARY nimeval HINTS ${NIMROD_DIR}
                                                     ENV NIMROD_DIR
               PATH_SUFFIXES lib)
  if(NOT NIMROD_NIMEVAL_LIBRARY)
    message(FATAL_ERROR "FUSIONIO_ENABLE_NIMROD needs NIMROD's nimeval "
                        "library (set NIMROD_DIR)")
  endif()
  message(STATUS "Found NIMROD nimeval: ${NIMROD_NIMEVAL_LIBRARY}")
endif()

# The PCMS package finds Omega_h if and only if PCMS has its Omega_h backend.
# Without NATIVE, Open(..., Evaluation::Native) and fio_trace -native throw.
if(FUSIONIO_ENABLE_WRAPPED AND TARGET Omega_h::omega_h)
  set(FUSIONIO_NATIVE_DEFAULT ON)
else()
  set(FUSIONIO_NATIVE_DEFAULT OFF)
endif()
option(FUSIONIO_ENABLE_NATIVE "Build the native M3D-C1 evaluator"
       ${FUSIONIO_NATIVE_DEFAULT})
if(FUSIONIO_ENABLE_NATIVE)
  if(NOT FUSIONIO_ENABLE_WRAPPED)
    message(FATAL_ERROR "FUSIONIO_ENABLE_NATIVE needs FUSIONIO_ENABLE_WRAPPED "
                        "(Fusion-IO 1.x reads the M3D-C1 files)")
  endif()
  if(NOT TARGET Omega_h::omega_h)
    message(FATAL_ERROR "FUSIONIO_ENABLE_NATIVE needs a PCMS built with "
                        "PCMS_ENABLE_OMEGA_H")
  endif()
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(fusionio src/quantity.cpp src/utilities.cpp)
add_library(fusionio::fusionio ALIAS fusionio)
target_compile_features(fusionio PUBLIC cxx_std_20)
target_include_directories(
  fusionio
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(fusionio PUBLIC pcms::pcms)
set(FUSIONIO_HEADERS include/fusionio/fusionio.hpp
                     include/fusionio/utilities.hpp)
if(FUSIONIO_ENABLE_WRAPPED)
  target_sources(fusionio PRIVATE src/fusion_io_space.cpp)
  target_link_libraries(fusionio PRIVATE fusionio::fusionio_fusionio)
else()
  target_sources(fusionio PRIVATE src/wrapped_unavailable.cpp)
endif()
if(FUSIONIO_ENABLE_NATIVE)
  target_sources(fusionio PRIVATE src/m3dc1_native.cpp)
  target_compile_definitions(fusionio PRIVATE FUSIONIO_ENABLE_NATIVE)
  target_link_libraries(fusionio PRIVATE Omega_h::omega_h)
endif()
if(FUSIONIO_ENABLE_NIMROD)
  target_sources(fusionio PRIVATE src/nimrod_space.cpp)
  target_link_libraries(fusionio PRIVATE ${NIMROD_NIMEVAL_LIBRARY})
else()
  target_sources(fusionio PRIVATE src/nimrod_unavailable.cpp)
endif()

install(TARGETS fusionio EXPORT fusionio2-targets)
install(FILES ${FUSIONIO_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fusionio)
install(EXPORT fusionio2-targets NAMESPACE fusionio::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fusionio2)
# the components built, for find_package(fusionio2 COMPONENTS ...)
configure_package_config_file(
  cmake/fusionio2-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/fusionio2-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fusionio2)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/fusionio2-config-version.cmake
  COMPATIBILITY SameMinorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fusionio2-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/fusionio2-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fusionio2)

option(FUSIONIO_BUILD_TOOLS "Build the Fusion-IO tools (fio_trace)" ON)
if(FUSIONIO_BUILD_TOOLS AND FUSIONIO_ENABLE_WRAPPED)
  find_package(MPI REQUIRED COMPONENTS CXX)
  add_executable(fio_trace tools/trace/main.cpp)
  target_link_libraries(fio_trace PRIVATE fusionio MPI::MPI_CXX)
  install(TARGETS fio_trace)
endif()

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(test)
endif()
