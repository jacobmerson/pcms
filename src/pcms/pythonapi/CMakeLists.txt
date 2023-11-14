pybind11_add_module(c_pcms_python c_client_pybind.cpp ../capi/kokkos.cpp ../capi/client.cpp)
pybind11_add_module(pcms_python client_pybind.cpp ../capi/kokkos.cpp)

set_target_properties(c_pcms_python pcms_python PROPERTIES CXX_STANDARD 17)
target_link_libraries(pcms_python PRIVATE Kokkos::kokkos pybind11::module MPI::MPI_C pcms::core)
target_link_libraries(c_pcms_python PRIVATE Kokkos::kokkos pybind11::module MPI::MPI_C pcms::core)

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




