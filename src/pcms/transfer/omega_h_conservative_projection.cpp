#include "pcms/transfer/omega_h_conservative_projection.hpp"
#include "pcms/transfer/conservative_projection_solver.hpp"
#include "pcms/transfer/omega_h_mass_integrator.hpp"
#include "pcms/utility/arrays.h"
#include <Omega_h_array.hpp>
#include <memory>

namespace pcms
{

namespace
{

void CheckApplyCompatible(const Field<Real>& source, const Field<Real>& target,
                          const OmegaHLagrangeLayout& source_layout,
                          const OmegaHLagrangeLayout& target_layout)
{
  if (&source.GetLayout() != &source_layout) {
    throw pcms_error(
      "OmegaHConservativeProjection::Apply: source field layout mismatch");
  }
  if (&target.GetLayout() != &target_layout) {
    throw pcms_error(
      "OmegaHConservativeProjection::Apply: target field layout mismatch");
  }

  const auto& source_md = source.GetData().GetMetadata();
  const auto& target_md = target.GetData().GetMetadata();
  if (source_md.value_type != FieldValueType::Scalar ||
      target_md.value_type != FieldValueType::Scalar) {
    throw pcms_error(
      "OmegaHConservativeProjection::Apply: only scalar fields are supported");
  }
  if (source_md.value_coordinate_system != target_md.value_coordinate_system) {
    throw pcms_error("OmegaHConservativeProjection::Apply: source and target "
                     "value coordinate systems differ");
  }
}

} // namespace

OmegaHConservativeProjection::OmegaHConservativeProjection(
  const FunctionSpace& source_space, const FunctionSpace& target_space)
  : source_layout_(std::dynamic_pointer_cast<const OmegaHLagrangeLayout>(
      source_space.GetLayout())),
    target_layout_(std::dynamic_pointer_cast<const OmegaHLagrangeLayout>(
      target_space.GetLayout()))
{
  rhs_integrator_ = std::make_unique<OmegaHIntersectionRHSIntegrator>(
    source_layout_, source_space.GetCoordinateSystem(), target_layout_,
    target_space.GetCoordinateSystem());

  evaluator_ =
    source_space.CreatePointEvaluator<Real>(EvaluationRequest::FromCoordinates(
      rhs_integrator_->GetIntegrationPoints().GetCoordinates()));

  // Mass integrator is only needed to build the solver; PETSc reference-counts
  // the matrix so it remains alive inside the KSP after this scope ends.
  OmegaHMassIntegrator mass_integrator(target_layout_,
                                       target_space.GetCoordinateSystem());
  solver_ = std::make_unique<GalerkinProjectionSolver>(mass_integrator,
                                                       *rhs_integrator_);
}

// Defined here so that GalerkinProjectionSolver (forward-declared in the
// header) is a complete type when unique_ptr's destructor is instantiated.
OmegaHConservativeProjection::~OmegaHConservativeProjection() = default;

void OmegaHConservativeProjection::Apply(const Field<Real>& source,
                                         Field<Real>& target) const
{
  CheckApplyCompatible(source, target, *source_layout_, *target_layout_);

  const auto target_values = solver_->Solve(*evaluator_, source);
  auto target_values_h = Omega_h::HostRead<Omega_h::Real>(target_values);
  target.SetDOFHolderDataHost(Rank2View<const Real, HostMemorySpace>(
    target_values_h.data(), target_layout_->GetNumOwnedDofHolder(),
    target_layout_->GetNumComponents()));
}

} // namespace pcms
