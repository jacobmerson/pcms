# Helpers of the examples project (see examples/CMakeLists.txt).

# What this PCMS provides. The targets exist when PCMS was built with the
# dependency: Omega_h::omega_h comes with the package config (or the PCMS
# build) only for an Omega_h build.
if(TARGET Omega_h::omega_h)
  set(PCMS_EXAMPLES_HAVE_OMEGA_H ON)
else()
  set(PCMS_EXAMPLES_HAVE_OMEGA_H OFF)
endif()

# The interpreter of the Python examples and of the public-API gate.
if(Python_EXECUTABLE)
  set(PCMS_EXAMPLES_PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
else()
  find_package(Python3 QUIET COMPONENTS Interpreter)
  set(PCMS_EXAMPLES_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
endif()

set(PCMS_EXAMPLES_PYTHONPATH
    ""
    CACHE STRING "Directories with the pcms (and PyOmega_h) Python modules")

# Whether the Python examples can run: the pcms module is built in tree, or
# importable from the given path. Those that build meshes also need PyOmega_h.
function(pcms_examples_can_import modules result)
  if(NOT PCMS_EXAMPLES_PYTHON_EXECUTABLE)
    set(${result} OFF PARENT_SCOPE)
    return()
  endif()
  # the given path in front of the environment's
  list(JOIN PCMS_EXAMPLES_PYTHONPATH ":" path)
  execute_process(
    COMMAND
      ${CMAKE_COMMAND} -E env "PYTHONPATH=${path}:$ENV{PYTHONPATH}"
      ${PCMS_EXAMPLES_PYTHON_EXECUTABLE} -c "import ${modules}"
    RESULT_VARIABLE rc
    OUTPUT_QUIET ERROR_QUIET)
  if(rc EQUAL 0)
    set(${result} ON PARENT_SCOPE)
  else()
    set(${result} OFF PARENT_SCOPE)
  endif()
endfunction()

pcms_examples_can_import("PyOmega_h" PCMS_EXAMPLES_HAVE_PYOMEGA_H)
if(PCMS_EXAMPLES_STANDALONE)
  pcms_examples_can_import("pcms" PCMS_EXAMPLES_HAVE_PYTHON)
elseif(TARGET pcms AND PCMS_EXAMPLES_PYTHON_EXECUTABLE)
  set(PCMS_EXAMPLES_HAVE_PYTHON ON)
else()
  set(PCMS_EXAMPLES_HAVE_PYTHON OFF)
endif()
message(STATUS "Examples: Omega_h ${PCMS_EXAMPLES_HAVE_OMEGA_H}, "
               "Python ${PCMS_EXAMPLES_HAVE_PYTHON}, "
               "PyOmega_h ${PCMS_EXAMPLES_HAVE_PYOMEGA_H}")

# pcms_examples_skip(<what> <reason>): leave an example out, or stop if the
# environment requires every example.
function(pcms_examples_skip what reason)
  if("$ENV{PCMS_REQUIRE_EXAMPLES}" STREQUAL "1")
    message(FATAL_ERROR "example ${what}: ${reason} (PCMS_REQUIRE_EXAMPLES=1)")
  endif()
  message(STATUS "Skipping example ${what}: ${reason}")
endfunction()

# pcms_add_example(<name> <source> [CORE_ONLY] [ARGS <args>...])
#
# A C++ example: an executable that links the public PCMS API, run as the ctest
# example.<track>.<directory>.cpp. An example that uses Omega_h or MPI itself
# gets them on its link line when PCMS has them (PCMS links them privately);
# CORE_ONLY is for an example of the core API, which needs only Kokkos and
# PCMS.
function(pcms_add_example name source)
  cmake_parse_arguments(EX "CORE_ONLY" "" "ARGS" ${ARGN})
  get_filename_component(dir "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
  get_filename_component(track_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
  get_filename_component(track "${track_dir}" NAME)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE pcms::pcms)
  if(NOT EX_CORE_ONLY)
    if(PCMS_EXAMPLES_HAVE_OMEGA_H)
      target_link_libraries(${name} PRIVATE Omega_h::omega_h)
    endif()
    if(TARGET MPI::MPI_CXX)
      target_link_libraries(${name} PRIVATE MPI::MPI_CXX)
    endif()
  endif()
  target_include_directories(${name} PRIVATE "${PCMS_EXAMPLES_ROOT}/common")
  set(test example.${track}.${dir}.cpp)
  add_test(NAME ${test} COMMAND ${name} ${EX_ARGS})
  set_tests_properties(${test} PROPERTIES LABELS example TIMEOUT 60)
  if(DEFINED PCMS_TEST_OMP_NUM_THREADS)
    set_property(
      TEST ${test}
      APPEND
      PROPERTY ENVIRONMENT "OMP_NUM_THREADS=${PCMS_TEST_OMP_NUM_THREADS}")
  endif()
endfunction()

# pcms_add_python_example(<script> [NEEDS_PYOMEGA_H] [ARGS <args>...])
#
# The Python twin of the example in the current directory, run as the ctest
# example.<track>.<directory>.py. In tree it runs against the module that the
# build produces.
function(pcms_add_python_example script)
  cmake_parse_arguments(EX "NEEDS_PYOMEGA_H" "" "ARGS" ${ARGN})
  get_filename_component(dir "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
  get_filename_component(track_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
  get_filename_component(track "${track_dir}" NAME)
  if(NOT PCMS_EXAMPLES_HAVE_PYTHON)
    pcms_examples_skip("${track}/${dir} (Python)"
                       "Python or the pcms module is not available")
    return()
  endif()
  if(EX_NEEDS_PYOMEGA_H AND NOT PCMS_EXAMPLES_HAVE_PYOMEGA_H)
    pcms_examples_skip("${track}/${dir} (Python)" "PyOmega_h is not available")
    return()
  endif()
  set(test example.${track}.${dir}.py)
  add_test(NAME ${test}
           COMMAND ${PCMS_EXAMPLES_PYTHON_EXECUTABLE}
                   "${CMAKE_CURRENT_SOURCE_DIR}/${script}" ${EX_ARGS})
  set_tests_properties(${test} PROPERTIES LABELS example TIMEOUT 60)
  set(path ${PCMS_EXAMPLES_PYTHONPATH})
  if(TARGET pcms)
    list(PREPEND path "$<TARGET_FILE_DIR:pcms>")
  endif()
  set(modification "")
  foreach(p IN LISTS path)
    list(APPEND modification "PYTHONPATH=path_list_prepend:${p}")
  endforeach()
  set_tests_properties(${test} PROPERTIES ENVIRONMENT_MODIFICATION
                                          "${modification}")
  if(DEFINED PCMS_TEST_OMP_NUM_THREADS)
    set_property(
      TEST ${test}
      APPEND
      PROPERTY ENVIRONMENT "OMP_NUM_THREADS=${PCMS_TEST_OMP_NUM_THREADS}")
  endif()
endfunction()
