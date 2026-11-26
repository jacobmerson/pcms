find_package(Catch2 3 REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

# Fusion-IO's example data (fusion-io/examples/data: m3dc1/, geqdsk/).
set(FUSIONIO_TEST_DATA_DIR "" CACHE PATH "Fusion-IO example data directory")
# The NIMROD case is not distributed with any test data; point this at a local
# copy of the cylrwtm case (nimrod.in and dump_gs.00000.h5).
set(FUSIONIO_TEST_NIMROD_DIR "" CACHE PATH "NIMROD case for the NIMROD tests")

set(FUSIONIO_TEST_SOURCES)
# The tests of the wrapped sources need Fusion-IO 1.x.
if(FUSIONIO_TEST_DATA_DIR AND FUSIONIO_ENABLE_WRAPPED)
  list(APPEND FUSIONIO_TEST_SOURCES test_fusion_io.cpp test_source.cpp
       test_utilities.cpp bench_transfer.cpp)
  if(FUSIONIO_ENABLE_NATIVE)
    list(APPEND FUSIONIO_TEST_SOURCES test_m3dc1_native.cpp)
  endif()
  set_source_files_properties(
    test_fusion_io.cpp test_source.cpp test_m3dc1_native.cpp
    test_utilities.cpp bench_transfer.cpp
    PROPERTIES COMPILE_DEFINITIONS
               "FUSIONIO_TEST_DATA_DIR=\"${FUSIONIO_TEST_DATA_DIR}\"")
elseif(FUSIONIO_ENABLE_WRAPPED)
  message(STATUS "FUSIONIO_TEST_DATA_DIR not set; Fusion-IO tests disabled")
endif()
if(FUSIONIO_ENABLE_NIMROD AND FUSIONIO_TEST_NIMROD_DIR)
  list(APPEND FUSIONIO_TEST_SOURCES test_nimrod.cpp)
  set_source_files_properties(
    test_nimrod.cpp
    PROPERTIES COMPILE_DEFINITIONS
               "FUSIONIO_TEST_NIMROD_DIR=\"${FUSIONIO_TEST_NIMROD_DIR}\"")
endif()

# The BOUT++ builder needs no data; the reader's tests read the CBM18 grid of
# the PCMS test data (bout/cbm18/cbm18_grid.nc).
set(FUSIONIO_TEST_BOUT_GRID "" CACHE FILEPATH "CBM18 grid for the BOUT++ tests")
list(APPEND FUSIONIO_TEST_SOURCES test_bout.cpp)
if(FUSIONIO_ENABLE_BOUT AND FUSIONIO_TEST_BOUT_GRID)
  list(APPEND FUSIONIO_TEST_SOURCES test_bout_cbm18.cpp)
  set_source_files_properties(
    test_bout_cbm18.cpp
    PROPERTIES COMPILE_DEFINITIONS
               "FUSIONIO_TEST_BOUT_GRID=\"${FUSIONIO_TEST_BOUT_GRID}\"")
endif()

if(FUSIONIO_PCMS_HAS_POLYNOMIAL_RECONSTRUCTION)
  # SampleMarkers of an analytic function reconstructed on a point cloud
  list(APPEND FUSIONIO_TEST_SOURCES test_sampling.cpp)
endif()

if(FUSIONIO_ENABLE_GNET)
  # The synthetic GNET file is written by the tests and by this tool (the input
  # of reference runs of GNET tools).
  add_library(gnet_synthetic STATIC gnet_synthetic.cpp)
  target_compile_features(gnet_synthetic PUBLIC cxx_std_20)
  target_link_libraries(gnet_synthetic PUBLIC HDF5::HDF5)
  target_include_directories(gnet_synthetic PUBLIC
                             ${CMAKE_CURRENT_SOURCE_DIR})
  add_executable(make_synthetic_gnet make_synthetic_gnet.cpp)
  target_link_libraries(make_synthetic_gnet PRIVATE gnet_synthetic)
  list(APPEND FUSIONIO_TEST_SOURCES test_gnet_reader.cpp
       test_gnet_sampling.cpp test_gnet_reference.cpp)
endif()

if(FUSIONIO_TEST_SOURCES)
  add_executable(fusionio_tests test_main.cpp ${FUSIONIO_TEST_SOURCES})
  target_include_directories(fusionio_tests
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
  target_link_libraries(fusionio_tests PRIVATE fusionio pcms::transfer
                                               Catch2::Catch2 MPI::MPI_CXX)
  if(FUSIONIO_ENABLE_WRAPPED)
    target_link_libraries(fusionio_tests PRIVATE fusionio::fusionio_fusionio)
  endif()
  if(FUSIONIO_ENABLE_GNET)
    target_link_libraries(fusionio_tests PRIVATE gnet_synthetic)
  endif()
  if(FUSIONIO_ENABLE_NIMROD)
    # the NIMROD tests compare with nimeval directly
    target_link_libraries(fusionio_tests PRIVATE ${NIMROD_NIMEVAL_LIBRARY})
  endif()
  if(FUSIONIO_ENABLE_NATIVE)
    target_compile_definitions(fusionio_tests PRIVATE FUSIONIO_ENABLE_NATIVE)
    target_link_libraries(fusionio_tests PRIVATE Omega_h::omega_h)
  endif()
  add_test(NAME fusionio_tests COMMAND fusionio_tests)
endif()

if(FUSIONIO_PCMS_HAS_POLYNOMIAL_RECONSTRUCTION)
  # the markers do not depend on the number of threads: the files written with
  # one and with four must be identical
  foreach(threads 1 4)
    add_test(NAME sampling_threads_${threads}
             COMMAND fusionio_tests "[threads]")
    set_tests_properties(
      sampling_threads_${threads}
      PROPERTIES
        ENVIRONMENT
        "OMP_NUM_THREADS=${threads};OMP_PROC_BIND=false;FUSIONIO_TEST_MARKERS_FILE=${CMAKE_CURRENT_BINARY_DIR}/markers_${threads}.txt"
        FIXTURES_SETUP sampling_threads)
  endforeach()
  add_test(NAME sampling_threads_equal
           COMMAND ${CMAKE_COMMAND} -E compare_files
                   ${CMAKE_CURRENT_BINARY_DIR}/markers_1.txt
                   ${CMAKE_CURRENT_BINARY_DIR}/markers_4.txt)
  set_tests_properties(sampling_threads_equal PROPERTIES FIXTURES_REQUIRED
                                                         sampling_threads)
endif()

if(TARGET gnet2gtc)
  # the tool on the synthetic file, in memory and streaming
  add_test(NAME gnet2gtc_input
           COMMAND make_synthetic_gnet
                   ${CMAKE_CURRENT_BINARY_DIR}/gnet2gtc_input.h5)
  set_tests_properties(gnet2gtc_input PROPERTIES FIXTURES_SETUP gnet2gtc_input)
  foreach(mode memory stream)
    set(args --samples 500 --batch 200 --threshold-scale 1)
    if(mode STREQUAL stream)
      list(APPEND args --stream)
    endif()
    add_test(NAME gnet2gtc_${mode}
             COMMAND gnet2gtc --input ${CMAKE_CURRENT_BINARY_DIR}/gnet2gtc_input.h5
                     --output ${CMAKE_CURRENT_BINARY_DIR}/gnet2gtc_${mode}.h5
                     ${args})
    set_tests_properties(
      gnet2gtc_${mode}
      PROPERTIES FIXTURES_REQUIRED gnet2gtc_input
                 PASS_REGULAR_EXPRESSION "Completed Successfully"
                 FIXTURES_SETUP gnet2gtc_output)
  endforeach()
endif()

if(TARGET fio_trace AND FUSIONIO_TEST_DATA_DIR)
  # A few transits of two field lines in the M3D-C1 equilibrium.
  add_test(
    NAME fio_trace_smoke
    COMMAND fio_trace -m3dc1 ${FUSIONIO_TEST_DATA_DIR}/m3dc1/C1.h5 -dR 0.05
            -p 2 -t 2 -s 20 -pout 0 -qout 0
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(TARGET bout-grid AND FUSIONIO_TEST_BOUT_GRID)
  add_test(NAME bout_grid_cbm18 COMMAND bout-grid ${FUSIONIO_TEST_BOUT_GRID} 16
                                        15)
  set_tests_properties(bout_grid_cbm18 PROPERTIES PASS_REGULAR_EXPRESSION
                                                  "pressure \\(Pa\\) = ")
endif()
