#ifndef PCMS_TRANSFER_OMEGA_H_CONSERVATIVE_PROJECTION_H
#define PCMS_TRANSFER_OMEGA_H_CONSERVATIVE_PROJECTION_H

#include "pcms/field/function_space.h"
#include "pcms/field/layout/omega_h_lagrange.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/transfer/omega_h_intersection_rhs_integrator.hpp"
#include "pcms/transfer/transfer_operator.hpp"
#include <memory>

namespace pcms
{

// Forward declaration keeps PETSc headers out of this header.
class GalerkinProjectionSolver;

// Conservative Galerkin projection between Omega_h order-1 Lagrange spaces.
//
// Construction (one-time cost):
//   - Computes mesh intersections and quadrature data
//   (OmegaHIntersectionRHSIntegrator).
//   - Localizes integration points in the source mesh (PointEvaluator).
//   - Assembles the target mass matrix and factors it via KSP
//     (GalerkinProjectionSolver).
//
// Apply() (per-call cost):
//   - Evaluates the source field at the fixed integration points.
//   - Assembles a fresh RHS vector.
//   - Calls KSPSolve reusing the cached factorization.
//   No mesh intersection, no matrix assembly, no refactorization per call.
class OmegaHConservativeProjection : public TransferOperator<Real>
{
public:
  OmegaHConservativeProjection(const FunctionSpace& source_space,
                               const FunctionSpace& target_space);

  // Defined in the .cpp where GalerkinProjectionSolver is a complete type.
  ~OmegaHConservativeProjection() override;

  void Apply(const Field<Real>& source, Field<Real>& target) const override;

private:
  std::shared_ptr<const OmegaHLagrangeLayout> source_layout_;
  std::shared_ptr<const OmegaHLagrangeLayout> target_layout_;
  std::unique_ptr<OmegaHIntersectionRHSIntegrator> rhs_integrator_;
  std::unique_ptr<PointEvaluator<Real>> evaluator_;
  std::unique_ptr<GalerkinProjectionSolver> solver_;
};

} // namespace pcms

#endif // PCMS_TRANSFER_OMEGA_H_CONSERVATIVE_PROJECTION_H
