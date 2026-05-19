#ifndef PCMS_TRANSFER_OMEGA_H_RHS_INTEGRATOR_HPP
#define PCMS_TRANSFER_OMEGA_H_RHS_INTEGRATOR_HPP

#include "pcms/field/function_space.h"
#include "pcms/field/layout/omega_h_lagrange.h"
#include "pcms/transfer/integration_point_set.hpp"
#include "pcms/transfer/linear_form_integrator.hpp"
#include "pcms/transfer/mesh_intersection.hpp"
#include <Kokkos_Core.hpp>
#include <memory>

namespace pcms
{

// Concrete IntegrationPointSet backed by a device-side coordinate buffer.
class OmegaHRHSIntegrationPointSet : public IntegrationPointSet<DeviceMemorySpace>
{
public:
  explicit OmegaHRHSIntegrationPointSet(
    Kokkos::View<Real**, DeviceMemorySpace> coords);

  CoordinateView<DeviceMemorySpace> GetCoordinates() const override;

private:
  Kokkos::View<Real**, DeviceMemorySpace> coords_view_;
};

// Conservative-projection RHS integrator for order-1 Lagrange spaces on
// Omega_h 2D simplex meshes.
//
// Construction validates both spaces, computes mesh intersections, and runs a
// two-pass device kernel (count then fill) to build the quadrature data and
// allocate the owned RHS vector sized to the target DOF count.
//
// Assemble(sampled_values):
//   Zeros the owned vector, then for each integration point i and local node j:
//     contribution = stored_coeff[i][j] * sampled_values(i, 0)
//   Scatter-adds via VecSetValuesCOO with ADD_VALUES.
// GetVector() returns the assembled vector (non-owning handle).
class OmegaHRHSIntegrator : public LinearFormIntegrator
{
public:
  OmegaHRHSIntegrator(const FunctionSpace& source_space,
                      const FunctionSpace& target_space);
  OmegaHRHSIntegrator(std::shared_ptr<const OmegaHLagrangeLayout> source_layout,
                      CoordinateSystem source_coordinate_system,
                      std::shared_ptr<const OmegaHLagrangeLayout> target_layout,
                      CoordinateSystem target_coordinate_system);
  ~OmegaHRHSIntegrator();

  Vec GetVector() const noexcept override;

  const IntegrationPointSet<DeviceMemorySpace>&
  GetIntegrationPoints() const noexcept override;

  void Assemble(
    Rank2View<const Real, DeviceMemorySpace> sampled_values) override;

private:
  // Internal data bundle produced by the two-pass device kernel.
  struct Data {
    Kokkos::View<Real**, DeviceMemorySpace> coords;        // [num_pts, 2]
    Kokkos::View<PetscInt*, DeviceMemorySpace> node_gids; // [num_pts * 3]
    Kokkos::View<Real*, DeviceMemorySpace> coeffs;         // [num_pts * 3]
    PetscInt nverts = 0;                                   // target DOF count
  };

  static Data BuildData(const FunctionSpace& source_space,
                        const FunctionSpace& target_space);
  static Data BuildData(
    std::shared_ptr<const OmegaHLagrangeLayout> source_layout,
    CoordinateSystem source_coordinate_system,
    std::shared_ptr<const OmegaHLagrangeLayout> target_layout,
    CoordinateSystem target_coordinate_system);

  OmegaHRHSIntegrator(Data data);

  Vec vec_ = nullptr;
  OmegaHRHSIntegrationPointSet point_set_;
  Kokkos::View<PetscInt*, DeviceMemorySpace> node_gids_; // COO indices, num_pts * 3
  Kokkos::View<Real*, DeviceMemorySpace> coeffs_;        // weights,    num_pts * 3
};

// Builds an OmegaHRHSIntegrator for conservative L2 projection.
// Both spaces must use OmegaHLagrangeLayout, order-1, scalar, Cartesian.
std::unique_ptr<LinearFormIntegrator> BuildOmegaHConservativeRHSIntegrator(
  const FunctionSpace& source_space,
  const FunctionSpace& target_space);

} // namespace pcms

#endif // PCMS_TRANSFER_OMEGA_H_RHS_INTEGRATOR_HPP
