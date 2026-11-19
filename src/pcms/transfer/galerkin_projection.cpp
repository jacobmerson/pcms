#include "pcms/transfer/galerkin_projection.hpp"
#include "pcms/transfer/conservative_projection_solver.hpp"
#include "pcms/transfer/omega_h_form_integrator_utils.hpp"
#include "pcms/transfer/omega_h_mass_integrator.hpp"
#include "pcms/utility/arrays.hpp"
#include <Kokkos_Core.hpp>
#include <Omega_h_array.hpp>

namespace pcms::transfer
{

namespace
{

const Geometry& SourceGeometry(const PointEvaluatorFactory& source)
{
  const auto geometry = source.GetGeometry();
  if (geometry == nullptr) {
    throw pcms_error("GalerkinProjection: the source has no geometry");
  }
  return *geometry;
}

Quadrature MakeQuadrature(const FunctionSpace& target_space,
                          const ProjectionOptions& options)
{
  if (options.quadrature) {
    if (options.quadrature->points == nullptr ||
        options.quadrature->points->GetGeometry() !=
          target_space.GetGeometry() ||
        options.quadrature->weights.extent(0) !=
          static_cast<size_t>(options.quadrature->points->NumPoints())) {
      throw pcms_error("GalerkinProjection: the quadrature must be bound on "
                       "the target space's geometry and have one weight per "
                       "point");
    }
    return *options.quadrature;
  }
  return target_space.GetGeometry()->CreateQuadrature(options.quadrature_options);
}

} // namespace

GalerkinProjection::GalerkinProjection(
  std::shared_ptr<const PointEvaluatorFactory> source_ptr,
  std::shared_ptr<const FunctionSpace> target_space_ptr,
  ProjectionOptions options)
  : source_(std::move(source_ptr)),
    target_space_(std::move(target_space_ptr)),
    target_layout_(detail::ResolveOmegaHLagrangeSpace(
                     *target_space_, 0, 1, "GalerkinProjection", "target")
                     .layout),
    quadrature_(MakeQuadrature(*target_space_, options))
{
  const PointEvaluatorFactory& source = *source_;
  const FunctionSpace& target_space = *target_space_;
  const Geometry& source_geometry = SourceGeometry(source);

  // The source is bound to the points in the cells of its geometry that the
  // quadrature names (no search), else searched for; the target is bound on
  // its own geometry already.
  const CoordinateView<DeviceMemorySpace> points(
    quadrature_.points->GetCoordinateSystem(),
    MakeConstRank2View(quadrature_.points->LocatedPointValues()));
  source_at_points_ = source.CreatePointEvaluator<Real>(source_geometry.Bind(
    points, quadrature_.other, OutOfBoundsPolicy{}, options.to_source));
  target_at_points_ =
    target_space.CreatePointEvaluator<Real>(quadrature_.points);
  if (!target_at_points_->SupportsTranspose()) {
    throw pcms_error("GalerkinProjection: the target space's evaluator does "
                     "not support the transpose");
  }

  if (options.control_variate) {
    interpolator_.emplace(
      source_, target_space_,
      source_geometry.Bind(
        target_space.GetLayout()->GetDOFHolderCoordinates(),
        OutOfBoundsPolicy{}, options.to_source));
    control_variate_.emplace(target_space.CreateFunction<Real>());
  }

  // Mass integrator is only needed to build the solver; PETSc reference-counts
  // the matrix so it remains alive inside the KSP after this scope ends.
  OmegaHMassIntegrator mass_integrator(target_space, options.mass_matrix_type);
  solver_ = std::make_unique<GalerkinProjectionSolver>(mass_integrator);

  const int num_components = target_layout_->GetNumComponents();
  const auto num_points = static_cast<std::size_t>(quadrature_.points->NumPoints());
  samples_ = Kokkos::View<Real**, DeviceMemorySpace>("projection_samples",
                                                     num_points, num_components);
  dofs_ = Kokkos::View<Real**, DeviceMemorySpace>(
    "projection_dofs", target_layout_->GetNumDofHolderWithGhosts(),
    num_components);
  rows_ = Kokkos::View<Real*, DeviceMemorySpace>(
    "projection_rows", target_layout_->GetNumDofHolderWithGhosts());
  target_values_ = Kokkos::View<Real**, DeviceMemorySpace>(
    "projection_target_values", target_layout_->GetNumDofHolder(),
    num_components);
}

// Defined here so that GalerkinProjectionSolver (forward-declared in the
// header) is a complete type when unique_ptr's destructor is instantiated.
GalerkinProjection::~GalerkinProjection() = default;

void GalerkinProjection::Apply(const Field<Real>& source,
                               Field<Real>& target) const
{
  const int num_components = target_layout_->GetNumComponents();
  if (source.GetLayout().GetNumComponents() != num_components ||
      source.GetData().GetValueType() != target.GetData().GetValueType()) {
    throw pcms_error("GalerkinProjection::Apply: the source and the target "
                     "must have the same value type and number of components");
  }
  if (&target.GetLayout() != target_layout_.get()) {
    throw pcms_error("GalerkinProjection::Apply: target field layout mismatch");
  }
  const bool control_variate = interpolator_.has_value();
  const auto num_points = static_cast<LO>(samples_.extent(0));
  // Shallow copies so the device lambdas capture the views, not `this`.
  auto samples = samples_;
  auto dofs = dofs_;
  auto rows = rows_;
  const auto weights = quadrature_.weights;

  // 1. f at the points, minus the control variate g there; the control
  //    variate is the interpolation of the source onto the target space.
  source_at_points_->Evaluate(
    source, ValueView<Real, DeviceMemorySpace>(source.GetData().GetValueBasis(),
                                               pcms::MakeRank2View(samples)));
  if (control_variate) {
    interpolator_->Apply(source, *control_variate_);
    Kokkos::View<Real**, DeviceMemorySpace> g("projection_control_variate",
                                              samples.extent(0),
                                              samples.extent(1));
    target_at_points_->Evaluate(
      *control_variate_,
      ValueView<Real, DeviceMemorySpace>(
        control_variate_->GetData().GetValueBasis(), pcms::MakeRank2View(g)));
    Kokkos::parallel_for(
      "projection_residual",
      Kokkos::RangePolicy<DefaultExecutionSpace>(0, num_points),
      KOKKOS_LAMBDA(const LO i) {
        for (int c = 0; c < num_components; ++c) {
          samples(i, c) -= g(i, c);
        }
      });
  }

  // 2. b = E^T (W f)
  Kokkos::parallel_for(
    "projection_weight", Kokkos::RangePolicy<DefaultExecutionSpace>(0, num_points),
    KOKKOS_LAMBDA(const LO i) {
      for (int c = 0; c < num_components; ++c) {
        samples(i, c) *= weights(i);
      }
    });
  Kokkos::deep_copy(dofs, 0.0);
  target_at_points_->EvaluateTranspose(MakeConstRank2View(samples_),
                                       pcms::MakeRank2View(dofs));

  // 3. M x = b for each component, in the rows of M (the layout's
  //    global-to-local order), then x (+ g) in the owned DOF holders.
  const int num_dof_holders = target_layout_->GetNumDofHolder();
  const int num_holders = target_layout_->GetNumDofHolderWithGhosts();
  const auto global_to_local = target_layout_->GetGlobalToLocalPermutation();
  const bool has_ghosts = target_layout_->HasGhosts();
  const auto owned_indices = target_layout_->GetOwnedIndices();
  const auto values = pcms::MakeRank2View(target_values_);
  for (int c = 0; c < num_components; ++c) {
    Kokkos::parallel_for(
      "projection_rows", Kokkos::RangePolicy<DefaultExecutionSpace>(0, num_holders),
      KOKKOS_LAMBDA(const LO h) { rows(global_to_local(h)) = dofs(h, c); });
    const auto solution = solver_->Solve(Kokkos::View<const Real*, DeviceMemorySpace>(rows));
    Kokkos::parallel_for(
      "projection_scatter_solution",
      Kokkos::RangePolicy<DefaultExecutionSpace>(0, num_dof_holders),
      KOKKOS_LAMBDA(const int o) {
        const LO i = has_ghosts ? owned_indices(o) : o;
        values(o, c) = solution[global_to_local(i)];
      });
  }
  if (control_variate) {
    // x = g + M^-1 r: g is indexed by owned DOF holder, as the values
    const auto g_nodal = control_variate_->GetDOFHolderData().GetValues();
    Kokkos::parallel_for(
      "projection_add_control_variate",
      Kokkos::RangePolicy<DefaultExecutionSpace>(0, num_dof_holders),
      KOKKOS_LAMBDA(const int o) {
        for (int c = 0; c < num_components; ++c) {
          values(o, c) += g_nodal(o, c);
        }
      });
  }
  Kokkos::fence();
  target.SetDOFHolderDataUnchecked(MakeConstRank2View(target_values_));
  target.Synchronize();
}

} // namespace pcms::transfer
