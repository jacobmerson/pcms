find_package(Python REQUIRED)
include(FindPythonModule.cmake)
message(STATUS "Python_EXECUTABLE ${Python_EXECUTABLE}")
find_python_module(mpi4py REQUIRED) 

if(mpi4py_FOUND)
  execute_process(
    COMMAND
      "${Python_EXECUTABLE}" "-c"
      "import mpi4py as m; print(m.__version__); print(m.get_include());"
    RESULT_VARIABLE
      _mpi4py_SEARCH_SUCCESS
    OUTPUT_VARIABLE
      _mpi4py_VALUES
    ERROR_VARIABLE
      _mpi4py_ERROR_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )

  # Convert the process output into a list
  string(REGEX REPLACE ";" "\\\\;" _mpi4py_VALUES ${_mpi4py_VALUES})
  string(REGEX REPLACE "\n" ";" _mpi4py_VALUES ${_mpi4py_VALUES})
  list(GET _mpi4py_VALUES 0 mpi4py_VERSION)
  list(GET _mpi4py_VALUES 1 mpi4py_INCLUDE_DIRS)

  # Make sure all directory separators are '/'
  string(REGEX REPLACE "\\\\" "/" mpi4py_INCLUDE_DIRS ${mpi4py_INCLUDE_DIRS})

  # Get the major and minor version numbers
  string(REGEX REPLACE "\\." ";" _mpi4py_VERSION_LIST ${mpi4py_VERSION})
  list(GET _mpi4py_VERSION_LIST 0 mpi4py_VERSION_MAJOR)
  list(GET _mpi4py_VERSION_LIST 1 mpi4py_VERSION_MINOR)
  list(GET _mpi4py_VERSION_LIST 2 mpi4py_VERSION_PATCH)
  string(REGEX MATCH "[0-9]*" mpi4py_VERSION_PATCH ${mpi4py_VERSION_PATCH})
  math(EXPR mpi4py_VERSION_DECIMAL
      "(${mpi4py_VERSION_MAJOR} * 10000) + (${mpi4py_VERSION_MINOR} * 100) + ${mpi4py_VERSION_PATCH}")
endif()

pybind11_add_module(c_pcms_python c_client_pybind.cpp ../capi/kokkos.cpp ../capi/client.cpp)
pybind11_add_module(pcms_python client_pybind.cpp ../capi/kokkos.cpp)

target_include_directories(c_pcms_python
  SYSTEM PRIVATE
    ${mpi4py_INCLUDE_DIRS}
)

target_include_directories(pcms_python
  SYSTEM PRIVATE
    ${mpi4py_INCLUDE_DIRS}
)

set_target_properties(c_pcms_python pcms_python PROPERTIES CXX_STANDARD 17)
target_link_libraries(pcms_python PRIVATE Kokkos::kokkos pybind11::module MPI::MPI_C pcms::core)
target_link_libraries(c_pcms_python PRIVATE Kokkos::kokkos pybind11::module MPI::MPI_C pcms::core)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/PythonFieldAdapters.py ${CMAKE_CURRENT_SOURCE_DIR}/c_test_proxy_coupling_xgc_client.py
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR})


install(
        TARGETS c_pcms_python pcms_python
        EXPORT pcms_python-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pcms/pythonapi/)
install(
        EXPORT pcms_python-targets
        NAMESPACE pcms::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pcms)




