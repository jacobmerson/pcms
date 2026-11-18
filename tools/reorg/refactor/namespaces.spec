# PCMS reorganization: namespace moves.
#
# `namespace <target>` starts a section. Each path below it is a file whose
# top-level `namespace pcms` code moves to <target>, or a directory (ending
# in /) meaning every C/C++ file below it. `!<path>` leaves a file out of its
# section (write it after the directory that lists it).
# All moves are applied in one pass by move_namespaces.sh.

namespace pcms::backend::xgc
  src/pcms/backends/xgc/
  include/pcms/backends/xgc/
namespace pcms::backend::mfem
  src/pcms/backends/mfem/
  include/pcms/backends/mfem/
namespace pcms::backend::omega_h_meshfields
  src/pcms/backends/omega_h_meshfields/
namespace pcms::backend::point_cloud
  src/pcms/backends/point_cloud/
  include/pcms/backends/point_cloud/
  !include/pcms/backends/point_cloud/layout.hpp
  !include/pcms/backends/point_cloud/mls_evaluator.hpp
  !src/pcms/backends/point_cloud/layout.cpp
namespace pcms::backend::uniform_grid
  src/pcms/backends/uniform_grid/
  include/pcms/backends/uniform_grid/
namespace pcms::backend::omega_h
  src/pcms/backends/omega_h/
  include/pcms/backends/omega_h/

namespace pcms::utility
  src/pcms/utility/array_mask.hpp
  src/pcms/utility/mesh_geometry.hpp
  src/pcms/utility/mesh_geometry.cpp
  src/pcms/utility/mpi_type.hpp
  src/pcms/utility/omega_h_array_utils.hpp

  # interpolator_aliases.hpp keeps its generic type aliases (member_type,
  # ScratchMatView, ...) in pcms:: (still private), since other files define
  # unrelated aliases with the same names.
namespace pcms::field
  src/pcms/field/element_dispatch.hpp
  src/pcms/field/evaluator/interpolator_logger.hpp
  src/pcms/field/evaluator/interpolator_view_utils.hpp
  src/pcms/field/evaluator/mls_interpolation_impl.hpp
  src/pcms/field/gid_permutation.hpp
  src/pcms/field/layout/empty.hpp
  src/pcms/field/evaluator/mls_interpolation.cpp
  src/pcms/field/gid_permutation.cpp
  src/pcms/field/layout/empty.cpp

namespace pcms::transfer
  src/pcms/transfer/bilinear_form_integrator.hpp
  src/pcms/transfer/conservative_projection_solver.hpp
  src/pcms/transfer/linear_form_integrator.hpp
  src/pcms/transfer/mass_matrix_integrator.hpp
  src/pcms/transfer/omega_h_conservative_projection.hpp
  src/pcms/transfer/omega_h_control_variate_projection.hpp
  src/pcms/transfer/omega_h_form_integrator_utils.hpp
  src/pcms/transfer/omega_h_intersection_rhs_integrator.hpp
  src/pcms/transfer/omega_h_mass_integrator.hpp
  src/pcms/transfer/omega_h_mc_rhs_integrator.hpp
  src/pcms/transfer/petsc_utils.hpp
  src/pcms/transfer/conservative_projection_solver.cpp
  src/pcms/transfer/omega_h_conservative_projection.cpp
  src/pcms/transfer/omega_h_control_variate_projection.cpp
  src/pcms/transfer/omega_h_intersection_rhs_integrator.cpp
  src/pcms/transfer/omega_h_mass_integrator.cpp
  src/pcms/transfer/omega_h_mc_rhs_integrator.cpp
  src/pcms/transfer/petsc_utils.cpp

namespace pcms::coupler
  src/pcms/coupler/partition.hpp

# Implementation types that no public signature names (the pcms::detail tier)
namespace pcms::detail
  include/pcms/field/detail/transformed_point_evaluator.hpp
  include/pcms/field/evaluator/mls_interpolation.hpp
  include/pcms/localization/mls_support_helpers.hpp
  include/pcms/localization/uniform_grid_search.hpp
  src/pcms/localization/mls_support_helpers.cpp
  include/pcms/coupler/detail/field_communicator.hpp
  include/pcms/coupler/detail/field_exchange_planner.hpp
  include/pcms/coupler/detail/field_layout_communicator.hpp
  include/pcms/coupler/detail/field_serializer.hpp
  include/pcms/coupler/detail/global_communicator.hpp
  src/pcms/coupler/field_exchange_planner.cpp
  src/pcms/coupler/field_layout_communicator.cpp
  include/pcms/backends/point_cloud/layout.hpp
  include/pcms/backends/point_cloud/mls_evaluator.hpp
  src/pcms/backends/point_cloud/layout.cpp
  include/pcms/field/gradient_source.hpp
  src/pcms/field/gradient_source.cpp
  include/pcms/field/simple_field_data.hpp
