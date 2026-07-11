#ifndef PCMS_TRANSFER_OMEGA_H_MC_RHS_INTEGRATOR_HPP
#define PCMS_TRANSFER_OMEGA_H_MC_RHS_INTEGRATOR_HPP

#include "pcms/field/function_space.h"
#include "pcms/field/layout/omega_h_lagrange.h"
#include "pcms/transfer/integration_point_set.hpp"
#include "pcms/transfer/linear_form_integrator.hpp"
#include <Kokkos_Core.hpp>
#include <cstdint>
#include <memory>

namespace pcms
{

// Strategy for generating sample points on each target element.
enum class MonteCarloSampling
{
  UniformRandom, // independent uniform draws per element (XorShift64 pool)
};

// Monte Carlo RHS integrator for conservative L2 projection onto order-1
// Lagrange spaces on Omega_h 2D simplex meshes.
//
// Instead of intersection-based quadrature, each load-vector entry
//   b_j = \int phi_j f dx
// is estimated per target element T from a uniform sample x_1..x_N over T:
//   b_j ~= (|T| / N) * sum_i phi_j(x_i) f(x_i)
// Unit-square samples (pseudo-random) are mapped to each triangle
// with the area-preserving sqrt parametrization. Sample locations are fixed
// at construction and exposed through GetIntegrationPoints(); Assemble()
// consumes source-field values sampled there, exactly like
// OmegaHIntersectionRHSIntegrator, so this drops into
// GalerkinProjectionSolver::Solve.
class OmegaHMonteCarloRHSIntegrator : public LinearFormIntegrator
{
public:
  OmegaHMonteCarloRHSIntegrator(const FunctionSpace& target_space,
                                int samples_per_element,
                                MonteCarloSampling sampling,
                                uint64_t seed = 8675309);
  OmegaHMonteCarloRHSIntegrator(
    std::shared_ptr<const OmegaHLagrangeLayout> target_layout,
    CoordinateSystem target_coordinate_system, int samples_per_element,
    MonteCarloSampling sampling, uint64_t seed = 8675309);
  ~OmegaHMonteCarloRHSIntegrator();

  OmegaHMonteCarloRHSIntegrator(const OmegaHMonteCarloRHSIntegrator&) = delete;
  OmegaHMonteCarloRHSIntegrator& operator=(
    const OmegaHMonteCarloRHSIntegrator&) = delete;

  Vec GetVector() const noexcept override;

  const IntegrationPointSet<DeviceMemorySpace>& GetIntegrationPoints()
    const noexcept override;

  void Assemble(
    Rank2View<const Real, DeviceMemorySpace> sampled_values) override;

private:
  // Internal data bundle produced by the sample-generation kernel.
  struct Data
  {
    Kokkos::View<Real**, DeviceMemorySpace> coords;       // [num_samples, 2]
    Kokkos::View<PetscInt*, DeviceMemorySpace> node_gids; // [num_samples * 3]
    Kokkos::View<Real*, DeviceMemorySpace> coeffs;        // [num_samples * 3]
    PetscInt nverts = 0;                                  // target DOF count
  };

  static Data BuildData(
    const std::shared_ptr<const OmegaHLagrangeLayout>& target_layout,
    CoordinateSystem target_coordinate_system, int samples_per_element,
    MonteCarloSampling sampling, uint64_t seed);

  explicit OmegaHMonteCarloRHSIntegrator(Data data);

  Vec vec_ = nullptr;
  IntegrationPointSet<DeviceMemorySpace> point_set_;
  Kokkos::View<PetscInt*, DeviceMemorySpace> node_gids_; // COO indices
  Kokkos::View<Real*, DeviceMemorySpace> coeffs_;        // basis * |T| / N
};

} // namespace pcms

#endif // PCMS_TRANSFER_OMEGA_H_MC_RHS_INTEGRATOR_HPP
