set(PCMS_TRANSIENT_PARTICIPANT_ADAPTER_HEADERS
    participant_protocol.hpp
    participant_client.hpp
    remote_participant.hpp)
set(PCMS_TRANSIENT_PARTICIPANT_ADAPTER_SOURCES
    participant_client.cpp
    remote_participant.cpp)

if(PCMS_ENABLE_OMEGA_H)
  list(APPEND PCMS_TRANSIENT_PARTICIPANT_ADAPTER_HEADERS
       coupler_mesh_backend.hpp)
  list(APPEND PCMS_TRANSIENT_PARTICIPANT_ADAPTER_SOURCES
       coupler_mesh_backend.cpp)
endif()

add_library(pcms_transient_participant_adapter
            ${PCMS_TRANSIENT_PARTICIPANT_ADAPTER_SOURCES})
set_target_properties(
  pcms_transient_participant_adapter
  PROPERTIES OUTPUT_NAME pcmstransientparticipantadapter
             EXPORT_NAME transient_participant_adapter)
add_library(
  pcms::transient_participant_adapter
  ALIAS pcms_transient_participant_adapter)

target_sources(
  pcms_transient_participant_adapter
  PUBLIC FILE_SET transient_participant_adapter
         TYPE HEADERS
         BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../..
         FILES ${PCMS_TRANSIENT_PARTICIPANT_ADAPTER_HEADERS})
target_compile_features(
  pcms_transient_participant_adapter PUBLIC cxx_std_20)
target_link_libraries(
  pcms_transient_participant_adapter
  PUBLIC pcms::transient redev::redev MPI::MPI_CXX)

if(PCMS_ENABLE_OMEGA_H)
  target_link_libraries(
    pcms_transient_participant_adapter
    PUBLIC pcms::coupler pcms::field pcms::transfer Omega_h::omega_h)
endif()

target_include_directories(
  pcms_transient_participant_adapter
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(
  TARGETS pcms_transient_participant_adapter
  EXPORT pcms_transient_participant_adapter-targets
  INCLUDES DESTINATION
           ${CMAKE_INSTALL_INCLUDEDIR}/pcms/transient/participant_adapter
  FILE_SET transient_participant_adapter
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pcms)

install(
  EXPORT pcms_transient_participant_adapter-targets
  NAMESPACE pcms::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pcms)
