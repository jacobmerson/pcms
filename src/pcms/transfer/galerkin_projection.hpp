#ifndef PCMS_TRANSFER_GALERKIN_PROJECTION_HPP
#define PCMS_TRANSFER_GALERKIN_PROJECTION_HPP

#include "pcms/backends/omega_h/lagrange_layout.hpp"
#include "pcms/field/coordinate_map.hpp"
#include "pcms/field/field.hpp"
#include "pcms/field/function_space.hpp"
#include "pcms/field/point_evaluator.hpp"
#include "pcms/field/point_evaluator_factory.hpp"
#include "pcms/geometry/quadrature.hpp"
#include "pcms/transfer/interpolator.hpp"
#include "pcms/transfer/projection.hpp"
#include "pcms/transfer/transfer_operator.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <optional>

namespace pcms::transfer
{

class GalerkinProjectionSolver;

// Galerkin L2 projection onto a Lagrange space on an Omega_h mesh through a
// quadrature of the target geometry:
//   M x = E^T W (f - g) + M g
// with E the target space evaluated at the quadrature points, W the weights,
// f the source at the points, and g the control variate (the interpolation of
// the source onto the target space, only with options.control_variate; zero
// otherwise). The right-hand side is accumulated by
// PointEvaluator::EvaluateTranspose; M is the PETSc mass matrix and the KSP is
// set up at construction.
//
// Construction (one-time cost): the quadrature, the evaluators of the source
// and the target at its points, the mass matrix and its factorization.
// Apply() (per-call cost): the source at the points, the transpose, a KSPSolve.
class GalerkinProjection : public TransferOperator<Real>
{
public:
  /// Holds `source` and `target_space`.
  GalerkinProjection(std::shared_ptr<const PointEvaluatorFactory> source,
                     std::shared_ptr<const FunctionSpace> target_space,
                     ProjectionOptions options);

  // Defined in the .cpp where GalerkinProjectionSolver is a complete type.
  ~GalerkinProjection() override;

  void Apply(const Field<Real>& source, Field<Real>& target) const override;

private:
  std::shared_ptr<const PointEvaluatorFactory> source_;
  std::shared_ptr<const FunctionSpace> target_space_;
  std::shared_ptr<const omega_h::OmegaHLagrangeLayout> target_layout_;
  Quadrature quadrature_;
  std::unique_ptr<PointEvaluator<Real>> source_at_points_;
  std::unique_ptr<PointEvaluator<Real>> target_at_points_;
  std::unique_ptr<GalerkinProjectionSolver> solver_;
  // the control variate: the interpolator and its scratch field on the target
  // space, overwritten on every Apply
  std::optional<pcms::detail::Interpolator<Real>> interpolator_;
  mutable std::optional<Field<Real>> control_variate_;

  mutable Kokkos::View<Real**, DeviceMemorySpace> samples_; // [points][comps]
  mutable Kokkos::View<Real**, DeviceMemorySpace> dofs_; // [holders+ghosts][comps]
  mutable Kokkos::View<Real*, DeviceMemorySpace> rows_;  // [rows of M]
  mutable Kokkos::View<Real**, DeviceMemorySpace> target_values_;
};

} // namespace pcms::transfer

#endif // PCMS_TRANSFER_GALERKIN_PROJECTION_HPP
