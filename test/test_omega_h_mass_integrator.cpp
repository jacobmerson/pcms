#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <pcms/field/function_space/lagrange.h>
#include <pcms/field/layout/omega_h_lagrange.h>
#include <pcms/transfer/mass_matrix_integrator.hpp>
#include <pcms/transfer/omega_h_mass_integrator.hpp>
#include <KokkosController.hpp>
#include <MeshField.hpp>
#include <MeshField_Element.hpp>
#include <petscksp.h>
#include <map>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace
{

Omega_h::Mesh BuildUnitSquare(Omega_h::Library& lib)
{
  const Omega_h::Reals coords({0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0});
  Omega_h::LOs ev2v({0, 1, 3, 1, 2, 3});
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

std::map<std::pair<pcms::GO, pcms::GO>, pcms::Real> BuildReferenceMassMap(
  Omega_h::Mesh& mesh, const pcms::FunctionSpace& space)
{
  using namespace pcms;

  MeshField::OmegahMeshField<DefaultExecutionSpace, 2,
                             MeshField::KokkosController>
    omf(mesh);
  auto coordField = omf.getCoordField();
  const auto [shp, map] = MeshField::Omegah::getTriangleElement<1>(mesh);
  MeshField::FieldElement coordFe(mesh.nelems(), coordField, shp, map);
  auto elm_mass_dev = buildElementMassMatrix(mesh, coordFe);
  auto elm_mass_host =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, elm_mass_dev);

  const auto layout =
    std::dynamic_pointer_cast<const OmegaHLagrangeLayout>(space.GetLayout());
  REQUIRE(layout != nullptr);
  const auto& faces2nodes = mesh.ask_down(Omega_h::FACE, Omega_h::VERT).ab2b;
  const auto& gids = layout->GetGidsHost();

  std::map<std::pair<GO, GO>, Real> ref;
  for (int e = 0; e < mesh.nelems(); ++e) {
    const auto verts = Omega_h::gather_verts<3>(faces2nodes, e);
    for (int i = 0; i < 3; ++i) {
      const GO row = static_cast<GO>(gids[verts[i]]);
      for (int j = 0; j < 3; ++j) {
        const GO col = static_cast<GO>(gids[verts[j]]);
        ref[{row, col}] += elm_mass_host(e * 9 + i * 3 + j);
      }
    }
  }
  return ref;
}

} // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE(
  "OmegaHMassIntegrator: assembled matrix matches buildElementMassMatrix",
  "[mass_integrator]")
{
  Omega_h::Library lib;
  auto mesh = BuildUnitSquare(lib);

  auto space = pcms::LagrangeFunctionSpace::FromMesh(
    mesh, 1, 1, pcms::CoordinateSystem::Cartesian, "global",
    pcms::LagrangeFunctionSpace::Backend::OmegaH);

  const auto ref = BuildReferenceMassMap(mesh, *space);

  auto integrator = pcms::BuildOmegaHMassIntegrator(*space);
  Mat mat = integrator->GetMatrix();

  for (const auto& [key, ref_val] : ref) {
    PetscInt r = static_cast<PetscInt>(key.first);
    PetscInt c = static_cast<PetscInt>(key.second);
    PetscScalar got = 0.0;
    MatGetValues(mat, 1, &r, 1, &c, &got);
    CAPTURE(key.first, key.second, ref_val, got);
    CHECK(static_cast<pcms::Real>(got) ==
          Catch::Approx(ref_val).epsilon(1e-12));
  }
  // mat is owned by integrator; no MatDestroy here.
}

TEST_CASE("OmegaHMassIntegrator: row sums match lumped mass",
          "[mass_integrator]")
{
  // For a consistent mass matrix M, the row sums equal the lumped mass at
  // each node: row_sum(i) = integral(N_i dx). For a regular mesh of unit
  // area with uniform nodal distribution the sum over all nodes equals 1.
  Omega_h::Library lib;
  auto mesh = BuildUnitSquare(lib);

  auto space = pcms::LagrangeFunctionSpace::FromMesh(
    mesh, 1, 1, pcms::CoordinateSystem::Cartesian, "global",
    pcms::LagrangeFunctionSpace::Backend::OmegaH);

  const auto layout =
    std::dynamic_pointer_cast<const pcms::OmegaHLagrangeLayout>(
      space->GetLayout());
  const auto& gids = layout->GetGidsHost();
  const int nverts = mesh.nverts();

  auto integrator = pcms::BuildOmegaHMassIntegrator(*space);
  Mat mat = integrator->GetMatrix();

  // Sum all row sums.
  pcms::Real grand_total = 0.0;
  for (int v = 0; v < nverts; ++v) {
    const pcms::GO row = static_cast<pcms::GO>(gids[v]);
    pcms::Real row_sum = 0.0;
    for (int u = 0; u < nverts; ++u) {
      const pcms::GO col = static_cast<pcms::GO>(gids[u]);
      PetscInt r = static_cast<PetscInt>(row);
      PetscInt c = static_cast<PetscInt>(col);
      PetscScalar val = 0.0;
      MatGetValues(mat, 1, &r, 1, &c, &val);
      row_sum += static_cast<pcms::Real>(val);
    }
    CAPTURE(v, row, row_sum);
    CHECK(row_sum > 0.0); // each row sum must be positive
    grand_total += row_sum;
  }

  // Total == area of domain == 1.
  CHECK(grand_total == Catch::Approx(1.0).epsilon(1e-10));
  // mat is owned by integrator; no MatDestroy here.
}

TEST_CASE("OmegaHMassIntegrator: rejects invalid layouts", "[mass_integrator]")
{
  Omega_h::Library lib;
  auto mesh = BuildUnitSquare(lib);

  SECTION("multi-component space throws")
  {
    auto space = pcms::LagrangeFunctionSpace::FromMesh(
      mesh, 1, 2, pcms::CoordinateSystem::Cartesian, "global",
      pcms::LagrangeFunctionSpace::Backend::OmegaH);
    REQUIRE_THROWS(pcms::BuildOmegaHMassIntegrator(*space));
  }

  SECTION("non-Cartesian coordinate system throws")
  {
    auto space = pcms::LagrangeFunctionSpace::FromMesh(
      mesh, 1, 1, pcms::CoordinateSystem::Cylindrical, "global",
      pcms::LagrangeFunctionSpace::Backend::OmegaH);
    REQUIRE_THROWS(pcms::BuildOmegaHMassIntegrator(*space));
  }
}
