#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <pcms/field/evaluation_request.h>
#include <pcms/field/function_space/lagrange.h>
#include <pcms/field/layout/omega_h_lagrange.h>
#include <pcms/transfer/conservative_projection_solver.hpp>
#include <pcms/transfer/omega_h_conservative_projection.hpp>
#include <pcms/transfer/omega_h_control_variate_projection.hpp>
#include <pcms/transfer/omega_h_mass_integrator.hpp>
#include <pcms/transfer/omega_h_mc_rhs_integrator.hpp>
#include "field_test_utils.h"
#include <petscksp.h>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace
{

// Build a unit-square 2D simplex mesh.
//   diagonal=0: T0=(0,1,3), T1=(1,2,3)
//   diagonal=1: T0=(0,1,2), T1=(0,2,3)
Omega_h::Mesh BuildUnitSquare(Omega_h::Library& lib, int diagonal)
{
  const Omega_h::Reals coords({0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0});
  Omega_h::LOs ev2v = (diagonal == 0) ? Omega_h::LOs({0, 1, 3, 1, 2, 3})
                                      : Omega_h::LOs({0, 1, 2, 0, 2, 3});
  Omega_h::Mesh mesh(&lib);
  Omega_h::build_from_elems_and_coords(&mesh, OMEGA_H_SIMPLEX, 2, ev2v, coords);
  mesh.add_tag<Omega_h::I8>(
    0, "class_dim", 1,
    Omega_h::Read<Omega_h::I8>(mesh.nverts(), Omega_h::I8(0)));
  mesh.add_tag<Omega_h::ClassId>(
    0, "class_id", 1,
    Omega_h::Read<Omega_h::ClassId>(mesh.nverts(), Omega_h::ClassId(0)));
  mesh.add_tag<Omega_h::I8>(
    1, "class_dim", 1,
    Omega_h::Read<Omega_h::I8>(mesh.nedges(), Omega_h::I8(1)));
  mesh.add_tag<Omega_h::ClassId>(
    1, "class_id", 1,
    Omega_h::Read<Omega_h::ClassId>(mesh.nedges(), Omega_h::ClassId(0)));
  mesh.add_tag<Omega_h::I8>(
    2, "class_dim", 1,
    Omega_h::Read<Omega_h::I8>(mesh.nelems(), Omega_h::I8(2)));
  mesh.add_tag<Omega_h::ClassId>(
    2, "class_id", 1,
    Omega_h::Read<Omega_h::ClassId>(mesh.nelems(), Omega_h::ClassId(0)));
  return mesh;
}

pcms::LagrangeFunctionSpace MakeP1Space(Omega_h::Mesh& mesh)
{
  return pcms::LagrangeFunctionSpace::FromMesh(
    mesh, 1, 1, pcms::CoordinateSystem::Cartesian, "global",
    pcms::LagrangeFunctionSpace::Backend::OmegaH);
}

// Evaluates source_field at the integrator's sample points and assembles.
void EvaluateAndAssemble(pcms::LinearFormIntegrator& integrator,
                         const pcms::FunctionSpace& source_space,
                         const pcms::Field<pcms::Real>& source_field)
{
  const auto& pts = integrator.GetIntegrationPoints();
  const std::size_t npts = pts.GetCoordinates().GetValues().extent(0);
  auto evaluator = source_space.CreatePointEvaluator<pcms::Real>(
    pcms::EvaluationRequest::FromCoordinates(pts.GetCoordinates()));
  Kokkos::View<pcms::Real**, pcms::DeviceMemorySpace> sampled("sampled", npts,
                                                              1);
  evaluator->Evaluate(source_field, pcms::MakeRank2View(sampled));
  integrator.Assemble(pcms::MakeConstRank2View(sampled));
}

} // namespace

// ---------------------------------------------------------------------------
// Monte Carlo RHS integrator
// ---------------------------------------------------------------------------

TEST_CASE("OmegaHMonteCarloRHSIntegrator: sample points lie inside the domain",
          "[mc_rhs_integrator]")
{
  Omega_h::Library lib;
  auto target_mesh = BuildUnitSquare(lib, 0);
  auto target_space = MakeP1Space(target_mesh);

  const int samples_per_element = 16;
  for (const auto sampling : {pcms::MonteCarloSampling::UniformRandom}) {
    pcms::OmegaHMonteCarloRHSIntegrator integrator(
      target_space, samples_per_element, sampling);
    const auto raw_coords =
      integrator.GetIntegrationPoints().GetCoordinates().GetValues();
    auto coords_view = Kokkos::View<const pcms::Real**, Kokkos::LayoutRight,
                                    pcms::DeviceMemorySpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
      raw_coords.data_handle(), raw_coords.extent(0), raw_coords.extent(1));
    auto coords_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, coords_view);

    REQUIRE(
      coords_h.extent(0) ==
      static_cast<std::size_t>(target_mesh.nelems() * samples_per_element));
    for (std::size_t i = 0; i < coords_h.extent(0); ++i) {
      CAPTURE(i, coords_h(i, 0), coords_h(i, 1));
      CHECK(coords_h(i, 0) >= -1e-12);
      CHECK(coords_h(i, 0) <= 1.0 + 1e-12);
      CHECK(coords_h(i, 1) >= -1e-12);
      CHECK(coords_h(i, 1) <= 1.0 + 1e-12);
    }
  }
}

TEST_CASE("OmegaHMonteCarloRHSIntegrator: constant field integrates exactly",
          "[mc_rhs_integrator]")
{
  // For f = c the estimator sums to c * |domain| for any sample placement,
  // because the P1 basis functions partition unity at every sample point.
  Omega_h::Library lib;
  auto source_mesh = BuildUnitSquare(lib, 1);
  auto target_mesh = BuildUnitSquare(lib, 0);
  auto source_space = MakeP1Space(source_mesh);
  auto target_space = MakeP1Space(target_mesh);

  auto source_field = source_space.CreateField<pcms::Real>();
  pcms::test::SetField(
    source_field, OMEGA_H_LAMBDA(pcms::Real, pcms::Real) { return 2.0; });

  for (const auto sampling : {pcms::MonteCarloSampling::UniformRandom}) {
    pcms::OmegaHMonteCarloRHSIntegrator integrator(target_space, 8, sampling);
    EvaluateAndAssemble(integrator, source_space, source_field);

    PetscScalar sum = 0.0;
    VecSum(integrator.GetVector(), &sum);
    INFO("sampling=" << "UniformRandom");
    CHECK(static_cast<pcms::Real>(sum) == Catch::Approx(2.0).margin(1e-12));
  }
}

// ---------------------------------------------------------------------------
// Control variate projection
// ---------------------------------------------------------------------------

TEST_CASE("OmegaHControlVariateProjection: exact for fields in the target "
          "space even with few samples",
          "[mc_rhs_integrator][control_variate]")
{
  // For a globally linear field the control variate (target-space interpolant
  // of the source field) equals the source field, so the sampled residual is
  // identically zero and the projection is exact regardless of sample count.
  Omega_h::Library lib;
  auto source_mesh = BuildUnitSquare(lib, 1);
  auto target_mesh = BuildUnitSquare(lib, 0);
  auto source_space = MakeP1Space(source_mesh);
  auto target_space = MakeP1Space(target_mesh);

  auto source_field = source_space.CreateField<pcms::Real>();
  auto target_field = target_space.CreateField<pcms::Real>();
  pcms::test::SetField(
    source_field,
    OMEGA_H_LAMBDA(pcms::Real x, pcms::Real y) { return 3.0 * x - y + 0.25; });

  pcms::OmegaHControlVariateProjection projection(
    source_space, target_space, /*samples_per_element=*/4,
    pcms::MonteCarloSampling::UniformRandom);
  projection.Apply(source_field, target_field);

  const auto values =
    pcms::FlattenToRank1View(target_field.GetDOFHolderDataHost());
  const auto coords_h = Omega_h::HostRead<Omega_h::Real>(target_mesh.coords());
  REQUIRE(static_cast<Omega_h::LO>(values.size()) == target_mesh.nverts());
  for (Omega_h::LO i = 0; i < target_mesh.nverts(); ++i) {
    const double expected =
      3.0 * coords_h[2 * i + 0] - coords_h[2 * i + 1] + 0.25;
    CAPTURE(i, expected, values[i]);
    CHECK(values[i] == Catch::Approx(expected).margin(1e-9));
  }
}

TEST_CASE("OmegaHControlVariateProjection: reduces error vs plain Monte Carlo",
          "[mc_rhs_integrator][control_variate]")
{
  // For a non-linear field the control variate does not cancel f exactly, but
  // the residual is much smaller than f, so with identical sample counts and
  // seeds the control-variate projection must be closer to the exact
  // (intersection-quadrature) projection than the plain Monte Carlo one.
  Omega_h::Library lib;
  auto source_mesh = BuildUnitSquare(lib, 1);
  auto target_mesh = BuildUnitSquare(lib, 0);
  auto source_space = MakeP1Space(source_mesh);
  auto target_space = MakeP1Space(target_mesh);

  auto source_field = source_space.CreateField<pcms::Real>();
  pcms::test::SetField(
    source_field, OMEGA_H_LAMBDA(pcms::Real x, pcms::Real y) {
      return x * x + x * y + 0.5 * y * y;
    });

  // Reference: exact mesh-intersection quadrature projection.
  auto reference_field = target_space.CreateField<pcms::Real>();
  pcms::OmegaHConservativeProjection reference_projection(source_space,
                                                          target_space);
  reference_projection.Apply(source_field, reference_field);
  const auto reference =
    pcms::FlattenToRank1View(reference_field.GetDOFHolderDataHost());

  const int samples_per_element = 64;
  const uint64_t seed = 20240611;

  // Plain Monte Carlo projection.
  pcms::OmegaHMonteCarloRHSIntegrator mc_integrator(
    target_space, samples_per_element, pcms::MonteCarloSampling::UniformRandom,
    seed);
  auto evaluator = source_space.CreatePointEvaluator<pcms::Real>(
    pcms::EvaluationRequest::FromCoordinates(
      mc_integrator.GetIntegrationPoints().GetCoordinates()));
  pcms::OmegaHMassIntegrator mass_integrator(target_space);
  pcms::GalerkinProjectionSolver solver(mass_integrator, mc_integrator);
  const auto mc_projected = solver.Solve(*evaluator, source_field);
  const auto mc_projected_h = Omega_h::HostRead<Omega_h::Real>(mc_projected);

  // Control-variate projection with the same sampling configuration.
  auto cv_field = target_space.CreateField<pcms::Real>();
  pcms::OmegaHControlVariateProjection cv_projection(
    source_space, target_space, samples_per_element,
    pcms::MonteCarloSampling::UniformRandom, seed);
  cv_projection.Apply(source_field, cv_field);
  const auto cv_values =
    pcms::FlattenToRank1View(cv_field.GetDOFHolderDataHost());

  double mc_error = 0.0;
  double cv_error = 0.0;
  for (Omega_h::LO i = 0; i < target_mesh.nverts(); ++i) {
    mc_error = std::max(mc_error, std::abs(mc_projected_h[i] - reference[i]));
    cv_error = std::max(cv_error, std::abs(cv_values[i] - reference[i]));
  }
  CAPTURE(mc_error, cv_error);
  CHECK(cv_error < mc_error);
}
