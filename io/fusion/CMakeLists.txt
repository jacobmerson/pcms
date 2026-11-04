cmake_minimum_required(VERSION 3.23)
project(fusion_io VERSION 2.0.0 LANGUAGES CXX)

# Fusion-IO 2.0 is built on PCMS and uses only its installed (public) headers.
# It can be built against an installed PCMS or added to a PCMS build with
# add_subdirectory.
if(NOT TARGET pcms::core)
  find_package(pcms CONFIG REQUIRED)
endif()
# Fusion-IO 1.x evaluates the fields until PCMS evaluates them natively.
find_package(fusionio CONFIG REQUIRED)

option(FUSION_IO_ENABLE_NIMROD "Evaluate NIMROD dumps with NIMROD's nimeval" OFF)
if(FUSION_IO_ENABLE_NIMROD)
  # NIMROD's C-callable evaluator (nimcore/eval_field.F90), the library nimpy
  # wraps. It carries its own RPATH to its dependencies.
  find_library(NIMROD_NIMEVAL_LIBRARY nimeval HINTS ${NIMROD_DIR}
                                                     ENV NIMROD_DIR
               PATH_SUFFIXES lib REQUIRED)
  message(STATUS "Found NIMROD nimeval: ${NIMROD_NIMEVAL_LIBRARY}")
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(fusion_io src/fusion_io_space.cpp)
add_library(fusion_io::fusion_io ALIAS fusion_io)
target_compile_features(fusion_io PUBLIC cxx_std_20)
target_include_directories(
  fusion_io
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(fusion_io PUBLIC pcms::core
                      PRIVATE fusionio::fusionio_fusionio)
set(FUSION_IO_HEADERS)
if(FUSION_IO_ENABLE_NIMROD)
  target_sources(fusion_io PRIVATE src/nimrod_space.cpp)
  target_link_libraries(fusion_io PRIVATE ${NIMROD_NIMEVAL_LIBRARY})
endif()

install(TARGETS fusion_io EXPORT fusion_io-targets)
if(FUSION_IO_HEADERS)
  install(FILES ${FUSION_IO_HEADERS}
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fusion_io)
endif()
install(EXPORT fusion_io-targets NAMESPACE fusion_io::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fusion_io)
configure_package_config_file(
  cmake/fusion_io-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/fusion_io-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fusion_io)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/fusion_io-config-version.cmake
  COMPATIBILITY SameMinorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fusion_io-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/fusion_io-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fusion_io)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(test)
endif()
