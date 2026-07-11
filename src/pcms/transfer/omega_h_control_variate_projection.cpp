#include "pcms/transfer/omega_h_control_variate_projection.hpp"
#include "pcms/transfer/conservative_projection_solver.hpp"
#include "pcms/transfer/omega_h_mass_integrator.hpp"
#include "pcms/utility/arrays.h"
#include <Kokkos_Core.hpp>
#include <Omega_h_array.hpp>

namespace pcms
{

OmegaHControlVariateProjection::OmegaHControlVariateProjection(
  const FunctionSpace& source_space, const FunctionSpace& target_space,
  int samples_per_element, MonteCarloSampling sampling, uint64_t seed)
  : target_layout_(std::dynamic_pointer_cast<const OmegaHLagrangeLayout>(
      target_space.GetLayout())),
    rhs_integrator_(std::make_unique<OmegaHMonteCarloRHSIntegrator>(
      target_layout_, target_space.GetCoordinateSystem(), samples_per_element,
      sampling, seed)),
    interpolator_(source_space, target_space),
    control_variate_(target_space.CreateField<Real>())
{
  const auto sample_coords =
    rhs_integrator_->GetIntegrationPoints().GetCoordinates();
  source_at_samples_ = source_space.CreatePointEvaluator<Real>(
    EvaluationRequest::FromCoordinates(sample_coords));
  control_variate_at_samples_ = target_space.CreatePointEvaluator<Real>(
    EvaluationRequest::FromCoordinates(sample_coords));

  // Mass integrator is only needed to build the solver; PETSc reference-counts
  // the matrix so it remains alive inside the KSP after this scope ends.
  OmegaHMassIntegrator mass_integrator(target_layout_,
                                       target_space.GetCoordinateSystem());
  solver_ = std::make_unique<GalerkinProjectionSolver>(mass_integrator,
                                                       *rhs_integrator_);
}

// Defined here so that GalerkinProjectionSolver (forward-declared in the
// header) is a complete type when unique_ptr's destructor is instantiated.
OmegaHControlVariateProjection::~OmegaHControlVariateProjection() = default;

void OmegaHControlVariateProjection::Apply(const Field<Real>& source,
                                           Field<Real>& target) const
{
  // 1. Control variate: interpolate the source field onto the target space.
  interpolator_.Apply(source, control_variate_);

  // 2. Sample the source field and the control variate at the fixed Monte
  //    Carlo sample points; the stochastic RHS integrates the residual.
  const std::size_t num_samples =
    rhs_integrator_->GetIntegrationPoints().NumPoints();
  Kokkos::View<Real**, DeviceMemorySpace> f_samples("cv_f_samples", num_samples,
                                                    1);
  Kokkos::View<Real**, DeviceMemorySpace> residual("cv_residual", num_samples,
                                                   1);
  source_at_samples_->Evaluate(source, MakeRank2View(f_samples));
  control_variate_at_samples_->Evaluate(control_variate_,
                                        MakeRank2View(residual));
  Kokkos::parallel_for(
    "cv_residual", Kokkos::RangePolicy<DefaultExecutionSpace>(0, num_samples),
    KOKKOS_LAMBDA(int i) {
      residual(i, 0) = f_samples(i, 0) - residual(i, 0);
    });

  // 3. Project the residual: M * delta = r.
  const auto delta = solver_->Solve(MakeConstRank2View(residual));

  // 4. x = g + delta. delta is indexed by global id (PETSc row), the DOF
  //    holder by local vertex id.
  const auto g_nodal = control_variate_.GetDOFHolderData();
  auto g_view = Kokkos::View<const Real*, DeviceMemorySpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
    g_nodal.data_handle(), g_nodal.extent(0));
  const auto device_gids = target_layout_->GetGids();
  const GO* gids_ptr = device_gids.data_handle();
  const int nverts = static_cast<int>(g_view.extent(0));

  Kokkos::View<Real*, DeviceMemorySpace> result("cv_result", nverts);
  Kokkos::parallel_for(
    "cv_add_correction", Kokkos::RangePolicy<DefaultExecutionSpace>(0, nverts),
    KOKKOS_LAMBDA(int i) {
      result(i) = g_view(i) + delta[static_cast<Omega_h::LO>(gids_ptr[i])];
    });
  Kokkos::fence();

  target.SetDOFHolderData(
    Rank2View<const Real, DeviceMemorySpace>(result.data(), nverts, 1));
}

} // namespace pcms
