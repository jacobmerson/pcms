find_package(Catch2 3 REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

# Fusion-IO's example data (fusion-io/examples/data: m3dc1/, geqdsk/).
set(FUSION_IO_TEST_DATA_DIR "" CACHE PATH "Fusion-IO example data directory")
# The NIMROD case is not distributed with any test data; point this at a local
# copy of the cylrwtm case (nimrod.in and dump_gs.00000.h5).
set(FUSION_IO_TEST_NIMROD_DIR "" CACHE PATH "NIMROD case for the NIMROD tests")

set(FUSION_IO_TEST_SOURCES)
if(FUSION_IO_TEST_DATA_DIR)
  list(APPEND FUSION_IO_TEST_SOURCES test_fusion_io.cpp test_source.cpp)
  set_source_files_properties(
    test_fusion_io.cpp test_source.cpp
    PROPERTIES COMPILE_DEFINITIONS
               "FUSION_IO_TEST_DATA_DIR=\"${FUSION_IO_TEST_DATA_DIR}\"")
else()
  message(STATUS "FUSION_IO_TEST_DATA_DIR not set; Fusion-IO tests disabled")
endif()
if(FUSION_IO_ENABLE_NIMROD AND FUSION_IO_TEST_NIMROD_DIR)
  list(APPEND FUSION_IO_TEST_SOURCES test_nimrod.cpp)
  set_source_files_properties(
    test_nimrod.cpp
    PROPERTIES COMPILE_DEFINITIONS
               "FUSION_IO_TEST_NIMROD_DIR=\"${FUSION_IO_TEST_NIMROD_DIR}\"")
endif()

if(FUSION_IO_TEST_SOURCES)
  add_executable(fusion_io_tests test_main.cpp ${FUSION_IO_TEST_SOURCES})
  target_include_directories(fusion_io_tests
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
  target_link_libraries(
    fusion_io_tests PRIVATE fusion_io pcms::transfer fusionio::fusionio_fusionio
                            Catch2::Catch2 MPI::MPI_CXX)
  if(FUSION_IO_ENABLE_NIMROD)
    # the NIMROD tests compare with nimeval directly
    target_link_libraries(fusion_io_tests PRIVATE ${NIMROD_NIMEVAL_LIBRARY})
  endif()
  add_test(NAME fusion_io_tests COMMAND fusion_io_tests)
endif()

if(TARGET fio_trace AND FUSION_IO_TEST_DATA_DIR)
  # A few transits of two field lines in the M3D-C1 equilibrium.
  add_test(
    NAME fio_trace_smoke
    COMMAND fio_trace -m3dc1 ${FUSION_IO_TEST_DATA_DIR}/m3dc1/C1.h5 -dR 0.05
            -p 2 -t 2 -s 20 -pout 0 -qout 0
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
