# Header-only transient coupled-simulation layer
# (docs/transient_coupled_simulation_plan.md). Dependency-free: it needs only
# pcms::utility for the Real typedef. SUNDIALS-backed strategies are stubbed
# behind PCMS_ENABLE_SUNDIALS inside the headers.
set(PCMS_TRANSIENT_HEADERS
        participant.hpp
        coupling.hpp
        accelerator.hpp
        schwarz.hpp
        error_estimator.hpp
        timestepper.hpp
        coupled_simulation.hpp
        transient.hpp
        testing/heat1d.hpp
        testing/linear_coupling.hpp
)

add_library(pcms_transient INTERFACE)
set_target_properties(pcms_transient PROPERTIES EXPORT_NAME transient)
target_sources(pcms_transient INTERFACE
        FILE_SET transient
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..
        FILES ${PCMS_TRANSIENT_HEADERS})
add_library(pcms::transient ALIAS pcms_transient)
target_compile_features(pcms_transient INTERFACE cxx_std_20)
target_link_libraries(pcms_transient INTERFACE pcms::utility)

# SUNDIALS-backed strategies (SundialsTimestepper wrapping SUNAdaptController).
# The concrete SUNAdaptController_{I,PI,PID,...} constructors live in
# SUNDIALS::arkode (which transitively pulls SUNDIALS::core, where the base
# EstimateStep/UpdateH/Reset dispatch lives). The flag is exposed as an INTERFACE
# compile definition (not via pcms/configuration.h) because this is a standalone
# header-only lib whose consumers may not link pcms::core.
if(PCMS_ENABLE_SUNDIALS)
  target_link_libraries(pcms_transient INTERFACE SUNDIALS::arkode SUNDIALS::core)
  target_compile_definitions(pcms_transient INTERFACE PCMS_ENABLE_SUNDIALS)
endif()
target_include_directories(pcms_transient INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(
        TARGETS pcms_transient
        EXPORT pcms_transient-targets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pcms/transient
        FILE_SET transient DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pcms)

install(
        EXPORT pcms_transient-targets
        NAMESPACE pcms::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pcms)
