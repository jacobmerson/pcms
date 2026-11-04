find_package(Catch2 3 REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

# Fusion-IO's example data (fusion-io/examples/data: m3dc1/, geqdsk/).
set(FUSION_IO_TEST_DATA_DIR "" CACHE PATH "Fusion-IO example data directory")

if(FUSION_IO_TEST_DATA_DIR)
  add_executable(fusion_io_tests test_main.cpp test_fusion_io.cpp)
  target_compile_definitions(
    fusion_io_tests
    PRIVATE PCMS_TEST_FUSIONIO_DIR="${FUSION_IO_TEST_DATA_DIR}")
  target_include_directories(fusion_io_tests
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
  target_link_libraries(
    fusion_io_tests PRIVATE fusion_io pcms::transfer fusionio::fusionio_fusionio
                            Catch2::Catch2 MPI::MPI_CXX)
  add_test(NAME fusion_io_tests COMMAND fusion_io_tests)
else()
  message(STATUS "FUSION_IO_TEST_DATA_DIR not set; Fusion-IO tests disabled")
endif()
