#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <pcms/field/evaluation_request.h>
#include <pcms/field/function_space/lagrange.h>
#include <pcms/field/layout/omega_h_lagrange.h>
#include <pcms/transfer/omega_h_intersection_rhs_integrator.hpp>
#include "field_test_utils.h"
#include <petscksp.h>
#include <cmath>
#include <unordered_map>
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

// Independent reference for the assembled conservative load vector
//   b_j = \int phi_j^target f dx
// when f is exactly representable in the target order-1 space (constant or
// affine): the integral then equals (M g)_j, where M is the target consistent
// mass matrix and g_k = f(target node k). Assembled here from per-element mass
// matrices on the target mesh alone — no intersection or source-mesh machinery
// — so it cross-checks the integrator instead of restating its computation.
//
// The result is keyed by global DOF id to match the PETSc row indexing of the
// integrator's assembled vector.
std::unordered_map<pcms::GO, pcms::Real> ExpectedLoadByGid(
  Omega_h::Mesh& target_mesh, const pcms::OmegaHLagrangeLayout& target_layout,
  const std::vector<pcms::Real>& g_by_vert)
{
  const auto coords_h = Omega_h::HostRead<Omega_h::Real>(target_mesh.coords());
  const auto faces2nodes_h = Omega_h::HostRead<Omega_h::LO>(
    target_mesh.ask_down(Omega_h::FACE, Omega_h::VERT).ab2b);
  const auto& gids = target_layout.GetGidsHost();

  std::unordered_map<pcms::GO, pcms::Real> b;
  for (int e = 0; e < target_mesh.nelems(); ++e) {
    const int v[3] = {faces2nodes_h[3 * e + 0], faces2nodes_h[3 * e + 1],
                      faces2nodes_h[3 * e + 2]};
    const double x0 = coords_h[2 * v[0]], y0 = coords_h[2 * v[0] + 1];
    const double x1 = coords_h[2 * v[1]], y1 = coords_h[2 * v[1] + 1];
    const double x2 = coords_h[2 * v[2]], y2 = coords_h[2 * v[2] + 1];
    const double area =
      0.5 * std::abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));

    // Consistent P1 element mass matrix M^e = (area/12) [[2,1,1],[1,2,1],
    // [1,1,2]], so (M^e g)_j = (area/12) (g_j + sum_k g_k).
    const double g_sum = g_by_vert[v[0]] + g_by_vert[v[1]] + g_by_vert[v[2]];
    for (int j = 0; j < 3; ++j) {
      const double bj = (area / 12.0) * (g_by_vert[v[j]] + g_sum);
      b[static_cast<pcms::GO>(gids[v[j]])] += static_cast<pcms::Real>(bj);
    }
  }
  return b;
}

// Assembles the integrator's load vector for source_field and compares it,
// per global DOF id, against the supplied expected values.
void CheckAssembledLoadMatches(
  pcms::FunctionSpace& source_space, pcms::LinearFormIntegrator& integrator,
  const pcms::Field<pcms::Real>& source_field,
  const std::unordered_map<pcms::GO, pcms::Real>& expected)
{
  const auto& pts = integrator.GetIntegrationPoints();
  const std::size_t npts = pts.GetCoordinates().GetValues().extent(0);

  auto evaluator = source_space.CreatePointEvaluator<pcms::Real>(
    pcms::EvaluationRequest::FromCoordinates(pts.GetCoordinates()));
  Kokkos::View<pcms::Real**, pcms::DeviceMemorySpace> sampled("sampled", npts,
                                                              1);
  evaluator->Evaluate(source_field, pcms::MakeRank2View(sampled));

  integrator.Assemble(pcms::MakeConstRank2View(sampled));
  Vec vec = integrator.GetVector();

  const PetscScalar* vec_array = nullptr;
  VecGetArrayRead(vec, &vec_array);
  for (const auto& [gid, ref_val] : expected) {
    const pcms::Real got = static_cast<pcms::Real>(vec_array[gid]);
    CAPTURE(gid, ref_val, got);
    CHECK(got == Catch::Approx(ref_val).epsilon(1e-10));
  }
  VecRestoreArrayRead(vec, &vec_array);
  // vec is owned by integrator; no VecDestroy here.
}

} // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("OmegaHIntersectionRHSIntegrator: integration points lie inside "
          "target domain",
          "[rhs_integrator]")
{
  Omega_h::Library lib;
  auto source_mesh = BuildUnitSquare(lib, 1);
  auto target_mesh = BuildUnitSquare(lib, 0);

  auto source_space = pcms::LagrangeFunctionSpace::FromMesh(
    source_mesh, 1, 1, pcms::CoordinateSystem::Cartesian, "global",
    pcms::LagrangeFunctionSpace::Backend::OmegaH);
  auto target_space = pcms::LagrangeFunctionSpace::FromMesh(
    target_mesh, 1, 1, pcms::CoordinateSystem::Cartesian, "global",
    pcms::LagrangeFunctionSpace::Backend::OmegaH);

  auto integrator =
    pcms::BuildOmegaHConservativeRHSIntegrator(source_space, target_space);

  const auto raw_coords =
    integrator->GetIntegrationPoints().GetCoordinates().GetValues();
  auto raw_coords_view = Kokkos::View<const pcms::Real**, Kokkos::LayoutRight,
                                      pcms::DeviceMemorySpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
    raw_coords.data_handle(), raw_coords.extent(0), raw_coords.extent(1));
  auto raw_coords_host =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, raw_coords_view);
  const std::size_t n = raw_coords_host.extent(0);

  REQUIRE(n > 0);
  for (std::size_t i = 0; i < n; ++i) {
    CAPTURE(i, raw_coords_host(i, 0), raw_coords_host(i, 1));
    CHECK(raw_coords_host(i, 0) >= -1e-12);
    CHECK(raw_coords_host(i, 0) <= 1.0 + 1e-12);
    CHECK(raw_coords_host(i, 1) >= -1e-12);
    CHECK(raw_coords_host(i, 1) <= 1.0 + 1e-12);
  }
}

TEST_CASE("OmegaHIntersectionRHSIntegrator: zero source field gives zero "
          "assembled vector",
          "[rhs_integrator]")
{
  Omega_h::Library lib;
  auto source_mesh = BuildUnitSquare(lib, 1);
  auto target_mesh = BuildUnitSquare(lib, 0);

  auto source_space = pcms::LagrangeFunctionSpace::FromMesh(
    source_mesh, 1, 1, pcms::CoordinateSystem::Cartesian, "global",
    pcms::LagrangeFunctionSpace::Backend::OmegaH);
  auto target_space = pcms::LagrangeFunctionSpace::FromMesh(
    target_mesh, 1, 1, pcms::CoordinateSystem::Cartesian, "global",
    pcms::LagrangeFunctionSpace::Backend::OmegaH);

  auto source_field = source_space.CreateField<pcms::Real>();
  // Default-constructed field has zero data.

  auto integrator =
    pcms::BuildOmegaHConservativeRHSIntegrator(source_space, target_space);
  const auto& pts = integrator->GetIntegrationPoints();
  const std::size_t npts = pts.GetCoordinates().GetValues().extent(0);

  auto evaluator = source_space.CreatePointEvaluator<pcms::Real>(
    pcms::EvaluationRequest::FromCoordinates(pts.GetCoordinates()));

  Kokkos::View<pcms::Real**, pcms::DeviceMemorySpace> sampled("sampled", npts,
                                                              1);
  evaluator->Evaluate(source_field, pcms::MakeRank2View(sampled));

  integrator->Assemble(pcms::MakeConstRank2View(sampled));
  Vec vec = integrator->GetVector();

  PetscReal norm = 0.0;
  VecNorm(vec, NORM_INFINITY, &norm);
  INFO("vector inf-norm=" << norm);
  CHECK(static_cast<pcms::Real>(norm) == Catch::Approx(0.0).margin(1e-14));
  // vec is owned by integrator; no VecDestroy here.
}

TEST_CASE("OmegaHIntersectionRHSIntegrator: constant field matches "
          "consistent-mass load",
          "[rhs_integrator]")
{
  // For a constant source field the assembled load vector must equal the
  // target consistent mass matrix applied to the constant nodal vector.
  Omega_h::Library lib;
  auto source_mesh = BuildUnitSquare(lib, 1);
  auto target_mesh = BuildUnitSquare(lib, 0);

  auto source_space = pcms::LagrangeFunctionSpace::FromMesh(
    source_mesh, 1, 1, pcms::CoordinateSystem::Cartesian, "global",
    pcms::LagrangeFunctionSpace::Backend::OmegaH);
  auto target_space = pcms::LagrangeFunctionSpace::FromMesh(
    target_mesh, 1, 1, pcms::CoordinateSystem::Cartesian, "global",
    pcms::LagrangeFunctionSpace::Backend::OmegaH);

  const double c = 2.0;
  auto source_field = source_space.CreateField<pcms::Real>();
  pcms::test::SetField(
    source_field, OMEGA_H_LAMBDA(pcms::Real, pcms::Real) { return c; });

  const auto target_layout =
    std::dynamic_pointer_cast<const pcms::OmegaHLagrangeLayout>(
      target_space.GetLayout());
  const std::vector<pcms::Real> g(target_mesh.nverts(), c);
  const auto expected = ExpectedLoadByGid(target_mesh, *target_layout, g);

  auto integrator =
    pcms::BuildOmegaHConservativeRHSIntegrator(source_space, target_space);
  CheckAssembledLoadMatches(source_space, *integrator, source_field, expected);
}

TEST_CASE(
  "OmegaHIntersectionRHSIntegrator: linear field matches consistent-mass load",
  "[rhs_integrator]")
{
  // f(x,y) = x + y exercises non-trivial quadrature paths that a constant
  // field cannot catch (a constant integrates exactly under any rule, while a
  // linear field requires at least 1st-order accuracy). The field is affine,
  // hence exactly representable in the target P1 space, so the assembled load
  // vector must equal the target consistent mass matrix applied to the nodal
  // values of x + y.
  Omega_h::Library lib;
  auto source_mesh = BuildUnitSquare(lib, 1);
  auto target_mesh = BuildUnitSquare(lib, 0);

  auto source_space = pcms::LagrangeFunctionSpace::FromMesh(
    source_mesh, 1, 1, pcms::CoordinateSystem::Cartesian, "global",
    pcms::LagrangeFunctionSpace::Backend::OmegaH);
  auto target_space = pcms::LagrangeFunctionSpace::FromMesh(
    target_mesh, 1, 1, pcms::CoordinateSystem::Cartesian, "global",
    pcms::LagrangeFunctionSpace::Backend::OmegaH);

  auto source_field = source_space.CreateField<pcms::Real>();
  pcms::test::SetField(
    source_field, OMEGA_H_LAMBDA(pcms::Real x, pcms::Real y) { return x + y; });

  const auto target_layout =
    std::dynamic_pointer_cast<const pcms::OmegaHLagrangeLayout>(
      target_space.GetLayout());
  const auto tgt_coords_h =
    Omega_h::HostRead<Omega_h::Real>(target_mesh.coords());
  std::vector<pcms::Real> g(target_mesh.nverts());
  for (int i = 0; i < target_mesh.nverts(); ++i) {
    g[i] = tgt_coords_h[2 * i + 0] + tgt_coords_h[2 * i + 1];
  }
  const auto expected = ExpectedLoadByGid(target_mesh, *target_layout, g);

  auto integrator =
    pcms::BuildOmegaHConservativeRHSIntegrator(source_space, target_space);
  CheckAssembledLoadMatches(source_space, *integrator, source_field, expected);
}

TEST_CASE("OmegaHIntersectionRHSIntegrator: rejects invalid layouts",
          "[rhs_integrator]")
{
  Omega_h::Library lib;
  auto source_mesh = BuildUnitSquare(lib, 1);
  auto target_mesh = BuildUnitSquare(lib, 0);

  SECTION("multi-component source space throws")
  {
    auto source_space = pcms::LagrangeFunctionSpace::FromMesh(
      source_mesh, 1, 2, pcms::CoordinateSystem::Cartesian, "global",
      pcms::LagrangeFunctionSpace::Backend::OmegaH);
    auto target_space = pcms::LagrangeFunctionSpace::FromMesh(
      target_mesh, 1, 1, pcms::CoordinateSystem::Cartesian, "global",
      pcms::LagrangeFunctionSpace::Backend::OmegaH);
    REQUIRE_THROWS(
      pcms::BuildOmegaHConservativeRHSIntegrator(source_space, target_space));
  }

  SECTION("multi-component target space throws")
  {
    auto source_space = pcms::LagrangeFunctionSpace::FromMesh(
      source_mesh, 1, 1, pcms::CoordinateSystem::Cartesian, "global",
      pcms::LagrangeFunctionSpace::Backend::OmegaH);
    auto target_space = pcms::LagrangeFunctionSpace::FromMesh(
      target_mesh, 1, 2, pcms::CoordinateSystem::Cartesian, "global",
      pcms::LagrangeFunctionSpace::Backend::OmegaH);
    REQUIRE_THROWS(
      pcms::BuildOmegaHConservativeRHSIntegrator(source_space, target_space));
  }

  SECTION("non-Cartesian source coordinate system throws")
  {
    auto source_space = pcms::LagrangeFunctionSpace::FromMesh(
      source_mesh, 1, 1, pcms::CoordinateSystem::Cylindrical, "global",
      pcms::LagrangeFunctionSpace::Backend::OmegaH);
    auto target_space = pcms::LagrangeFunctionSpace::FromMesh(
      target_mesh, 1, 1, pcms::CoordinateSystem::Cartesian, "global",
      pcms::LagrangeFunctionSpace::Backend::OmegaH);
    REQUIRE_THROWS(
      pcms::BuildOmegaHConservativeRHSIntegrator(source_space, target_space));
  }

  SECTION("non-Cartesian target coordinate system throws")
  {
    auto source_space = pcms::LagrangeFunctionSpace::FromMesh(
      source_mesh, 1, 1, pcms::CoordinateSystem::Cartesian, "global",
      pcms::LagrangeFunctionSpace::Backend::OmegaH);
    auto target_space = pcms::LagrangeFunctionSpace::FromMesh(
      target_mesh, 1, 1, pcms::CoordinateSystem::Cylindrical, "global",
      pcms::LagrangeFunctionSpace::Backend::OmegaH);
    REQUIRE_THROWS(
      pcms::BuildOmegaHConservativeRHSIntegrator(source_space, target_space));
  }
}
