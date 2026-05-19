#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <pcms/field/evaluation_request.h>
#include <pcms/field/function_space/lagrange.h>
#include <pcms/field/layout/omega_h_lagrange.h>
#include <pcms/transfer/load_vector_integrator.hpp>
#include <pcms/transfer/mesh_intersection.hpp>
#include <pcms/transfer/omega_h_rhs_integrator.hpp>
#include "field_test_utils.h"
#include <petscksp.h>
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
  Omega_h::LOs ev2v = (diagonal == 0)
    ? Omega_h::LOs({0, 1, 3, 1, 2, 3})
    : Omega_h::LOs({0, 1, 2, 0, 2, 3});
  Omega_h::Mesh mesh(&lib);
  Omega_h::build_from_elems_and_coords(&mesh, OMEGA_H_SIMPLEX, 2, ev2v,
                                       coords);
  mesh.add_tag<Omega_h::I8>(0, "class_dim", 1,
                            Omega_h::Read<Omega_h::I8>(mesh.nverts(),
                                                       Omega_h::I8(0)));
  mesh.add_tag<Omega_h::ClassId>(
    0, "class_id", 1,
    Omega_h::Read<Omega_h::ClassId>(mesh.nverts(), Omega_h::ClassId(0)));
  mesh.add_tag<Omega_h::I8>(1, "class_dim", 1,
                            Omega_h::Read<Omega_h::I8>(mesh.nedges(),
                                                       Omega_h::I8(1)));
  mesh.add_tag<Omega_h::ClassId>(
    1, "class_id", 1,
    Omega_h::Read<Omega_h::ClassId>(mesh.nedges(), Omega_h::ClassId(0)));
  mesh.add_tag<Omega_h::I8>(2, "class_dim", 1,
                            Omega_h::Read<Omega_h::I8>(mesh.nelems(),
                                                       Omega_h::I8(2)));
  mesh.add_tag<Omega_h::ClassId>(
    2, "class_id", 1,
    Omega_h::Read<Omega_h::ClassId>(mesh.nelems(), Omega_h::ClassId(0)));
  return mesh;
}

} // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("OmegaHRHSIntegrator: integration points lie inside target domain",
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
  auto raw_coords_view = Kokkos::View<
    const pcms::Real**, Kokkos::LayoutRight, pcms::DeviceMemorySpace,
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

TEST_CASE("OmegaHRHSIntegrator: zero source field gives zero assembled vector",
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

TEST_CASE(
  "OmegaHRHSIntegrator: constant field matches buildLoadVector reference",
  "[rhs_integrator]")
{
  // Verifies that the staged integrator produces the same assembled load vector
  // as the legacy buildLoadVector() for a constant source field of 2.0.
  Omega_h::Library lib;
  auto source_mesh = BuildUnitSquare(lib, 1);
  auto target_mesh = BuildUnitSquare(lib, 0);

  auto source_space = pcms::LagrangeFunctionSpace::FromMesh(
    source_mesh, 1, 1, pcms::CoordinateSystem::Cartesian, "global",
    pcms::LagrangeFunctionSpace::Backend::OmegaH);
  auto target_space = pcms::LagrangeFunctionSpace::FromMesh(
    target_mesh, 1, 1, pcms::CoordinateSystem::Cartesian, "global",
    pcms::LagrangeFunctionSpace::Backend::OmegaH);

  // Set constant source field = 2.0.
  auto source_field = source_space.CreateField<pcms::Real>();
  {
    const int n = source_mesh.nverts();
    std::vector<pcms::Real> vals(n, 2.0);
    source_field.SetDOFHolderDataHost(
      pcms::Rank1View<const pcms::Real, pcms::HostMemorySpace>(vals.data(), n));
  }

  // --- Reference via legacy buildLoadVector ---
  Omega_h::Write<Omega_h::Real> src_vals_oh(source_mesh.nverts(), 2.0);
  const auto intersection = pcms::intersectTargets(source_mesh, target_mesh);
  auto ref_flat =
    pcms::buildLoadVector(target_mesh, source_mesh, intersection, src_vals_oh);
  auto ref_host =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ref_flat);

  // Scatter ref_flat into per-global-node totals.
  const auto& tgt_faces2nodes =
    target_mesh.ask_down(Omega_h::FACE, Omega_h::VERT).ab2b;
  const auto target_layout =
    std::dynamic_pointer_cast<const pcms::OmegaHLagrangeLayout>(
      target_space.GetLayout());
  const auto& tgt_gids = target_layout->GetGidsHost();

  std::unordered_map<pcms::GO, pcms::Real> ref_by_gid;
  for (int e = 0; e < target_mesh.nelems(); ++e) {
    const auto verts = Omega_h::gather_verts<3>(tgt_faces2nodes, e);
    for (int k = 0; k < 3; ++k) {
      ref_by_gid[static_cast<pcms::GO>(tgt_gids[verts[k]])] +=
        static_cast<pcms::Real>(ref_host(e * 3 + k));
    }
  }

  // --- New staged integrator ---
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

  const PetscScalar* vec_array = nullptr;
  VecGetArrayRead(vec, &vec_array);

  // Compare per-node totals.
  for (const auto& [gid, ref_val] : ref_by_gid) {
    const pcms::Real got = static_cast<pcms::Real>(vec_array[gid]);
    CAPTURE(gid, ref_val, got);
    CHECK(got == Catch::Approx(ref_val).epsilon(1e-10));
  }
  VecRestoreArrayRead(vec, &vec_array);
  // vec is owned by integrator; no VecDestroy here.
}

TEST_CASE("OmegaHRHSIntegrator: linear field matches buildLoadVector reference",
          "[rhs_integrator]")
{
  // f(x,y) = x + y exercises non-trivial quadrature paths that a constant
  // field cannot catch (the constant case integrates any quadrature rule
  // exactly, while a linear field requires at least 1st-order accuracy).
  Omega_h::Library lib;
  auto source_mesh = BuildUnitSquare(lib, 1);
  auto target_mesh = BuildUnitSquare(lib, 0);

  auto source_space = pcms::LagrangeFunctionSpace::FromMesh(
    source_mesh, 1, 1, pcms::CoordinateSystem::Cartesian, "global",
    pcms::LagrangeFunctionSpace::Backend::OmegaH);
  auto target_space = pcms::LagrangeFunctionSpace::FromMesh(
    target_mesh, 1, 1, pcms::CoordinateSystem::Cartesian, "global",
    pcms::LagrangeFunctionSpace::Backend::OmegaH);

  // Set source field = x + y at each vertex.
  auto source_field = source_space.CreateField<pcms::Real>();
  pcms::test::SetField(
    source_field,
    OMEGA_H_LAMBDA(pcms::Real x, pcms::Real y) { return x + y; });

  // --- Reference via legacy buildLoadVector ---
  const auto src_coords_h =
    Omega_h::HostRead<Omega_h::Real>(source_mesh.coords());
  Omega_h::Write<Omega_h::Real> src_vals_oh(source_mesh.nverts());
  Omega_h::parallel_for(
    source_mesh.nverts(), OMEGA_H_LAMBDA(int i) {
      src_vals_oh[i] = src_coords_h[2 * i + 0] + src_coords_h[2 * i + 1];
    });

  const auto intersection = pcms::intersectTargets(source_mesh, target_mesh);
  auto ref_flat =
    pcms::buildLoadVector(target_mesh, source_mesh, intersection, src_vals_oh);
  auto ref_host =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ref_flat);

  const auto& tgt_faces2nodes =
    target_mesh.ask_down(Omega_h::FACE, Omega_h::VERT).ab2b;
  const auto target_layout =
    std::dynamic_pointer_cast<const pcms::OmegaHLagrangeLayout>(
      target_space.GetLayout());
  const auto& tgt_gids = target_layout->GetGidsHost();

  std::unordered_map<pcms::GO, pcms::Real> ref_by_gid;
  for (int e = 0; e < target_mesh.nelems(); ++e) {
    const auto verts = Omega_h::gather_verts<3>(tgt_faces2nodes, e);
    for (int k = 0; k < 3; ++k) {
      ref_by_gid[static_cast<pcms::GO>(tgt_gids[verts[k]])] +=
        static_cast<pcms::Real>(ref_host(e * 3 + k));
    }
  }

  // --- New staged integrator ---
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

  const PetscScalar* vec_array = nullptr;
  VecGetArrayRead(vec, &vec_array);

  for (const auto& [gid, ref_val] : ref_by_gid) {
    const pcms::Real got = static_cast<pcms::Real>(vec_array[gid]);
    CAPTURE(gid, ref_val, got);
    CHECK(got == Catch::Approx(ref_val).epsilon(1e-10));
  }
  VecRestoreArrayRead(vec, &vec_array);
  // vec is owned by integrator; no VecDestroy here.
}

TEST_CASE("OmegaHRHSIntegrator: rejects invalid layouts", "[rhs_integrator]")
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
