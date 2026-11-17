# Point-cloud backend (no external dependency). No public headers.

# discretization: part of pcms::discretization
target_sources(pcms_discretization PRIVATE discretization.cpp
                                           discretization.hpp)

# layout and field data: part of pcms::field
target_sources(pcms_field PRIVATE layout.cpp field_data.cpp layout.hpp
                                  field_data.hpp)

# MLS evaluation and the geometry
target_sources(pcms_field PRIVATE mls_evaluator.hpp evaluator.hpp geometry.hpp)
