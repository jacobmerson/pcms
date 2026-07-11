#ifndef PCMS_TRANSFER_OMEGA_H_INTERSECTION_RHS_INTEGRATOR_HPP
#define PCMS_TRANSFER_OMEGA_H_INTERSECTION_RHS_INTEGRATOR_HPP

#include "pcms/field/function_space.h"
#include "pcms/field/layout/omega_h_lagrange.h"
#include "pcms/transfer/integration_point_set.hpp"
#include "pcms/transfer/linear_form_integrator.hpp"
#include "pcms/transfer/mesh_intersection.hpp"
#include <Kokkos_Core.hpp>
#include <memory>

namespace pcms
{

// Conservative-projection RHS integrator for Lagrange spaces on Omega_h 2D
// simplex meshes. Source and target orders are handled independently: the
// source enters only through pointwise samples (any supported order), while the
// target order selects the basis and DOF layout (P0: one element-constant DOF;
// P1: three vertex DOFs). The quadrature order is source_order + target_order.
//
// Construction validates both spaces, computes mesh intersections, and runs a
// two-pass device kernel (count then fill) to build the quadrature data and
// allocate the owned RHS vector sized to the target DOF count.
//
// Assemble(sampled_values):
//   Zeros the owned vector, then for each integration point i and local target
//   DOF j:
//     contribution = stored_coeff[i][j] * sampled_values(i, 0)
//   Scatter-adds via VecSetValuesCOO with ADD_VALUES.
// GetVector() returns the assembled vector (non-owning handle).
class OmegaHIntersectionRHSIntegrator : public LinearFormIntegrator
{
public:
  OmegaHIntersectionRHSIntegrator(const FunctionSpace& source_space,
                                  const FunctionSpace& target_space);
  OmegaHIntersectionRHSIntegrator(
    std::shared_ptr<const OmegaHLagrangeLayout> source_layout,
    CoordinateSystem source_coordinate_system,
    std::shared_ptr<const OmegaHLagrangeLayout> target_layout,
    CoordinateSystem target_coordinate_system);
  ~OmegaHIntersectionRHSIntegrator();

  Vec GetVector() const noexcept override;

  const IntegrationPointSet<DeviceMemorySpace>& GetIntegrationPoints()
    const noexcept override;

  void Assemble(
    Rank2View<const Real, DeviceMemorySpace> sampled_values) override;

private:
  // Internal data bundle produced by the two-pass device kernel. node_gids and
  // coeffs are laid out with ndof_per_elem entries per integration point.
  struct Data
  {
    Kokkos::View<Real**, DeviceMemorySpace> coords;       // [num_pts, 2]
    Kokkos::View<PetscInt*, DeviceMemorySpace> node_gids; // [num_pts*ndof]
    Kokkos::View<Real*, DeviceMemorySpace> coeffs;        // [num_pts*ndof]
    int ndof_per_elem = 0;                                // target DOFs/element
    PetscInt num_target_dofs = 0;                         // target DOF count
  };

  static Data BuildData(const FunctionSpace& source_space,
                        const FunctionSpace& target_space);
  static Data BuildData(
    std::shared_ptr<const OmegaHLagrangeLayout> source_layout,
    CoordinateSystem source_coordinate_system,
    std::shared_ptr<const OmegaHLagrangeLayout> target_layout,
    CoordinateSystem target_coordinate_system);

  // Order-templated two-pass builder, selected at runtime on the target order.
  template <int TgtOrder>
  static Data BuildDataImpl(const OmegaHLagrangeLayout& source_layout,
                            const OmegaHLagrangeLayout& target_layout,
                            int quad_order);

  OmegaHIntersectionRHSIntegrator(Data data);

  Vec vec_ = nullptr;
  IntegrationPointSet<DeviceMemorySpace> point_set_;
  Kokkos::View<PetscInt*, DeviceMemorySpace>
    node_gids_;                                   // COO idx, num_pts*ndof
  Kokkos::View<Real*, DeviceMemorySpace> coeffs_; // weights, num_pts*ndof
  int ndof_per_elem_ = 0;                         // target DOFs/element
};

// Builds an OmegaHIntersectionRHSIntegrator for conservative L2 projection.
// Both spaces must use OmegaHLagrangeLayout, scalar, Cartesian, order-0 or
// order-1 (source and target orders may differ).
std::unique_ptr<LinearFormIntegrator> BuildOmegaHConservativeRHSIntegrator(
  const FunctionSpace& source_space, const FunctionSpace& target_space);

} // namespace pcms

#endif // PCMS_TRANSFER_OMEGA_H_INTERSECTION_RHS_INTEGRATOR_HPP
