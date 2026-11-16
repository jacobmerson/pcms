cmake_minimum_required(VERSION 3.23)
project(fusionio VERSION 2.0.0 LANGUAGES CXX)

# Fusion-IO 2.0 is built on PCMS and uses only its installed (public) headers.
# It can be built against an installed PCMS or added to a PCMS build with
# add_subdirectory.
if(NOT TARGET pcms::core)
  find_package(pcms CONFIG REQUIRED)
endif()
# Fusion-IO 1.x evaluates the fields until PCMS evaluates them natively. Its
# CMake package is also named fusionio; the installed configuration loads it
# from the file found here (FUSIONIO_1X_CONFIG) rather than by name.
find_package(fusionio CONFIG REQUIRED)
set(FUSIONIO_1X_CONFIG "${fusionio_CONFIG}")

option(FUSIONIO_ENABLE_NIMROD "Evaluate NIMROD dumps with NIMROD's nimeval" OFF)
if(FUSIONIO_ENABLE_NIMROD)
  # NIMROD's C-callable evaluator (nimcore/eval_field.F90), the library nimpy
  # wraps. It carries its own RPATH to its dependencies.
  find_library(NIMROD_NIMEVAL_LIBRARY nimeval HINTS ${NIMROD_DIR}
                                                     ENV NIMROD_DIR
               PATH_SUFFIXES lib REQUIRED)
  message(STATUS "Found NIMROD nimeval: ${NIMROD_NIMEVAL_LIBRARY}")
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(fusionio src/fusion_io_space.cpp)
add_library(fusionio::fusionio ALIAS fusionio)
target_compile_features(fusionio PUBLIC cxx_std_20)
target_include_directories(
  fusionio
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(fusionio PUBLIC pcms::core
                      PRIVATE fusionio::fusionio_fusionio)
set(FUSIONIO_HEADERS include/fusionio/fusionio.hpp)
if(FUSIONIO_ENABLE_NIMROD)
  target_sources(fusionio PRIVATE src/nimrod_space.cpp)
  target_link_libraries(fusionio PRIVATE ${NIMROD_NIMEVAL_LIBRARY})
else()
  target_sources(fusionio PRIVATE src/nimrod_unavailable.cpp)
endif()

install(TARGETS fusionio EXPORT fusionio-targets)
install(FILES ${FUSIONIO_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fusionio)
install(EXPORT fusionio-targets NAMESPACE fusionio::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fusionio)
configure_package_config_file(
  cmake/fusionio-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/fusionio-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fusionio)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/fusionio-config-version.cmake
  COMPATIBILITY SameMinorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fusionio-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/fusionio-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fusionio)

option(FUSIONIO_BUILD_TOOLS "Build the Fusion-IO tools (fio_trace)" ON)
if(FUSIONIO_BUILD_TOOLS)
  find_package(MPI REQUIRED COMPONENTS CXX)
  add_executable(fio_trace tools/trace/main.cpp)
  target_link_libraries(fio_trace PRIVATE fusionio MPI::MPI_CXX)
  install(TARGETS fio_trace)
endif()

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(test)
endif()
