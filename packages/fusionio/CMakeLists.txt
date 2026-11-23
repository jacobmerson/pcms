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
#   GNET     GNET distributions. Needs a PCMS with
#            PCMS_ENABLE_POLYNOMIAL_RECONSTRUCTION and HDF5.
#   BOUT     BOUT++ grid files. Needs netCDF (netcdf-c); the BOUT++ builder
#            (fusionio/bout.hpp) is always built.
include(CheckCXXSourceCompiles)
# Whether the PCMS found was built with the configuration macro `name`.
function(fusionio_pcms_has name result)
  set(CMAKE_REQUIRED_LIBRARIES pcms::utility)
  check_cxx_source_compiles("#include <pcms/configuration.hpp>
#ifndef ${name}
#error ${name} is not defined
#endif
int main() {}" ${result})
endfunction()

# SampleMarkers works with any function; its tests (and GNET) use polynomial
# reconstruction.
fusionio_pcms_has(PCMS_ENABLE_POLYNOMIAL_RECONSTRUCTION
                  FUSIONIO_PCMS_HAS_POLYNOMIAL_RECONSTRUCTION)

option(FUSIONIO_ENABLE_GNET "Read GNET distributions" OFF)
if(FUSIONIO_ENABLE_GNET)
  if(NOT FUSIONIO_PCMS_HAS_POLYNOMIAL_RECONSTRUCTION)
    message(FATAL_ERROR "FUSIONIO_ENABLE_GNET needs a PCMS built with "
                        "PCMS_ENABLE_POLYNOMIAL_RECONSTRUCTION")
  endif()
  # before Fusion-IO 1.x: its config defines HDF5_LIBRARIES (pkg-config's
  # "hdf5", without the library directory), which FindHDF5 would then reuse
  set(HDF5_USE_STATIC_LIBRARIES OFF)
  find_package(HDF5 REQUIRED COMPONENTS C)
  message(STATUS "Found HDF5: ${HDF5_VERSION}")
endif()

option(FUSIONIO_ENABLE_BOUT "Read BOUT++ grid files" OFF)
if(FUSIONIO_ENABLE_BOUT)
  find_path(NETCDF_INCLUDE_DIR netcdf.h HINTS ${NETCDF_DIR} ENV NETCDF_DIR
            PATH_SUFFIXES include)
  find_library(NETCDF_LIBRARY netcdf HINTS ${NETCDF_DIR} ENV NETCDF_DIR
               PATH_SUFFIXES lib lib64)
  if(NOT NETCDF_INCLUDE_DIR OR NOT NETCDF_LIBRARY)
    message(FATAL_ERROR "FUSIONIO_ENABLE_BOUT needs netCDF (netcdf-c; set "
                        "NETCDF_DIR)")
  endif()
  message(STATUS "Found netCDF: ${NETCDF_LIBRARY}")
  add_library(fusionio_netcdf INTERFACE IMPORTED)
  target_include_directories(fusionio_netcdf INTERFACE ${NETCDF_INCLUDE_DIR})
endif()

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

add_library(fusionio src/quantity.cpp src/utilities.cpp src/sampling.cpp
                     src/bout.cpp)
add_library(fusionio::fusionio ALIAS fusionio)
target_compile_features(fusionio PUBLIC cxx_std_20)
target_include_directories(
  fusionio
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(fusionio PUBLIC pcms::pcms)
set(FUSIONIO_HEADERS include/fusionio/bout.hpp include/fusionio/fusionio.hpp
                     include/fusionio/sampling.hpp
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
if(FUSIONIO_ENABLE_BOUT)
  target_sources(fusionio PRIVATE src/bout_reader.cpp)
  # its headers after PCMS's: a netCDF prefix may hold another PCMS's
  target_link_libraries(fusionio PRIVATE $<BUILD_INTERFACE:fusionio_netcdf>
                                         ${NETCDF_LIBRARY})
else()
  target_sources(fusionio PRIVATE src/bout_unavailable.cpp)
endif()
if(FUSIONIO_ENABLE_GNET)
  target_sources(fusionio PRIVATE src/gnet_reader.cpp src/gnet_space.cpp
                                  src/gnet_markers.cpp)
  target_link_libraries(fusionio PRIVATE HDF5::HDF5)
  list(APPEND FUSIONIO_HEADERS include/fusionio/gnet.hpp)
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

option(FUSIONIO_BUILD_TOOLS "Build the Fusion-IO tools (fio_trace, gnet2gtc)" ON)
if(FUSIONIO_BUILD_TOOLS AND FUSIONIO_ENABLE_WRAPPED)
  find_package(MPI REQUIRED COMPONENTS CXX)
  add_executable(fio_trace tools/trace/main.cpp)
  target_link_libraries(fio_trace PRIVATE fusionio MPI::MPI_CXX)
  install(TARGETS fio_trace)
endif()

if(FUSIONIO_BUILD_TOOLS AND FUSIONIO_ENABLE_GNET)
  add_executable(gnet2gtc tools/gnet/gnet2gtc.cpp)
  target_link_libraries(gnet2gtc PRIVATE fusionio)
  install(TARGETS gnet2gtc)
endif()

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(test)
endif()
