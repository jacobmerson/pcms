#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_shape.hpp>

#include <pcms/utility/coordinate_systems/cartesian.hpp>
#include <pcms/backends/omega_h/omega_h_geometry.hpp>
#include <pcms/backends/omega_h/mesh_intersection.hpp>
#include <pcms/utility/assert.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include "field_test_utils.hpp"

namespace pcms
{
using omega_h::IntersectionResults;
using omega_h::intersectTargets;
using omega_h::OmegaHGeometry;
using omega_h::simplices_have_degenerate_overlap;
} // namespace pcms

namespace
{
/// Assert the two intersection maps are identical: same per-target counts and
/// the same source elements in the same order.
void require_intersections_identical(const pcms::IntersectionResults& filtered,
                                     const pcms::IntersectionResults& exact)
{
  const auto filtered_offsets = Omega_h::HostRead(filtered.tgt2src_offsets);
  const auto exact_offsets = Omega_h::HostRead(exact.tgt2src_offsets);
  REQUIRE(filtered_offsets.size() == exact_offsets.size());
  for (int i = 0; i < filtered_offsets.size(); ++i) {
    REQUIRE(filtered_offsets[i] == exact_offsets[i]);
  }

  const auto filtered_indices = Omega_h::HostRead(filtered.tgt2src_indices);
  const auto exact_indices = Omega_h::HostRead(exact.tgt2src_indices);
  REQUIRE(filtered_indices.size() == exact_indices.size());
  for (int i = 0; i < filtered_indices.size(); ++i) {
    REQUIRE(filtered_indices[i] == exact_indices[i]);
  }
}
} // namespace

namespace
{
/// Rotates the vertices by `angle` about the z axis through (0.5, 0.5), then
/// moves them by `shift` along x.
void Move(Omega_h::Mesh& mesh, double angle, double shift)
{
  const int dim = mesh.dim();
  const auto coords = Omega_h::HostRead(mesh.coords());
  Omega_h::HostWrite<Omega_h::Real> moved(coords.size());
  for (int i = 0; i < coords.size(); ++i) {
    moved[i] = coords[i];
  }
  for (int v = 0; v < mesh.nverts(); ++v) {
    const double x = coords[v * dim] - 0.5;
    const double y = coords[v * dim + 1] - 0.5;
    moved[v * dim] = 0.5 + std::cos(angle) * x - std::sin(angle) * y + shift;
    moved[v * dim + 1] = 0.5 + std::sin(angle) * x + std::cos(angle) * y;
  }
  mesh.set_coords(Omega_h::Reals(moved.write()));
}

/// Requires each row of the intersection, with and without the prefilter, to
/// be the source elements whose clip with the target element has volume (by
/// the intersection's tolerances), ascending.
/// @return the total overlap volume
template <int Dim>
double RequireExhaustiveRows(Omega_h::Mesh& source, Omega_h::Mesh& target)
{
  const auto geometry = std::make_shared<pcms::OmegaHGeometry>(
    source, pcms::csys::CreateCartesian(Dim));
  const auto source_coords = Omega_h::HostRead(source.coords());
  const auto target_coords = Omega_h::HostRead(target.coords());
  const auto source_nodes = Omega_h::HostRead(source.ask_elem_verts());
  const auto target_nodes = Omega_h::HostRead(target.ask_elem_verts());
  const auto source_measures =
    Omega_h::HostRead(Omega_h::measure_elements_real(&source));
  const auto target_measures =
    Omega_h::HostRead(Omega_h::measure_elements_real(&target));

  double total_measure = 0;
  for (bool prefilter : {false, true}) {
    CAPTURE(Dim, prefilter);
    const auto result = pcms::intersectTargets(*geometry, target, prefilter);
    const auto offsets = Omega_h::HostRead(result.tgt2src_offsets);
    const auto indices = Omega_h::HostRead(result.tgt2src_indices);
    total_measure = 0;
    for (int t = 0; t < target.nelems(); ++t) {
      r3d::Few<r3d::Vector<Dim>, Dim + 1> tgt;
      for (int v = 0; v <= Dim; ++v) {
        for (int d = 0; d < Dim; ++d) {
          tgt[v][d] = target_coords[target_nodes[t * (Dim + 1) + v] * Dim + d];
        }
      }
      std::vector<int> expected;
      for (int s = 0; s < source.nelems(); ++s) {
        r3d::Few<r3d::Vector<Dim>, Dim + 1> src;
        for (int v = 0; v <= Dim; ++v) {
          for (int d = 0; d < Dim; ++d) {
            src[v][d] =
              source_coords[source_nodes[s * (Dim + 1) + v] * Dim + d];
          }
        }
        r3d::Polytope<Dim> intersection;
        r3d::intersect_simplices(intersection, tgt, src);
        const double measure = std::abs(r3d::measure(intersection));
        const double eps =
          std::max(PCMS_INTERSECTION_ABS_TOL,
                   PCMS_INTERSECTION_REL_TOL *
                     std::max(target_measures[t], source_measures[s]));
        if (intersection.nverts >= Dim + 1 && measure >= eps) {
          expected.push_back(s);
          total_measure += measure;
        }
      }
      const std::vector<int> row(indices.data() + offsets[t],
                                 indices.data() + offsets[t + 1]);
      CAPTURE(t);
      REQUIRE(row == expected);
    }
  }
  return total_measure;
}
} // namespace

TEST_CASE("intersection rows are the overlapping source elements, ascending",
          "[intersection]")
{
  Omega_h::Library lib;
  auto world = lib.world();
  // Every target centroid is outside the source, but real overlaps remain.
  SECTION("2D, target centroids outside")
  {
    auto source =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 0, 2, 2, 0, false);
    auto target =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 0, 1, 1, 0, false);
    Move(target, 0, 0.9);
    // the slab [0.9, 1] x [0, 1]
    REQUIRE(RequireExhaustiveRows<2>(source, target) ==
            Catch::Approx(0.1).margin(1e-12));
  }
  SECTION("3D, target centroids outside")
  {
    auto source = pcms::test::BuildReferenceTet(lib);
    auto target = pcms::test::BuildReferenceTet(lib);
    Move(target, 0, 0.9);
    // a tet of edge 0.1
    REQUIRE(RequireExhaustiveRows<3>(source, target) ==
            Catch::Approx(0.001 / 6).margin(1e-12));
  }
  SECTION("2D, rotated target partly outside")
  {
    auto source =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 0, 8, 8, 0, false);
    auto target =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 0, 5, 5, 0, false);
    Move(target, 0.5236, 0.3);
    RequireExhaustiveRows<2>(source, target);
  }
  SECTION("3D, rotated target partly outside")
  {
    auto source =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 4, 4, 4, false);
    auto target =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 3, 3, 3, false);
    Move(target, 0.4363, 0.3);
    RequireExhaustiveRows<3>(source, target);
  }
}

// A pre-filter that never fires would satisfy the equivalence test below
// while saving nothing, so pin down that it actually rejects the case it
// exists for, and that it does not reject a real overlap.
TEST_CASE("intersection prefilter rejects degenerate overlaps",
          "[intersection]")
{
  SECTION("3D, face-adjacent tets are rejected")
  {
    // Two tets sharing the face x=0; separated exactly by that plane.
    const r3d::Few<r3d::Vector<3>, 4> left{
      {{0, 0, 0}, {0, 1, 0}, {0, 0, 1}, {-1, 0, 0}}};
    const r3d::Few<r3d::Vector<3>, 4> right{
      {{0, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}}};
    REQUIRE(pcms::simplices_have_degenerate_overlap<3>(left, right));
  }
  SECTION("3D, overlapping tets are not rejected")
  {
    const r3d::Few<r3d::Vector<3>, 4> a{
      {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    // Same tet shifted slightly; the overlap has positive volume.
    const r3d::Few<r3d::Vector<3>, 4> b{
      {{0.1, 0.1, 0.1}, {1.1, 0.1, 0.1}, {0.1, 1.1, 0.1}, {0.1, 0.1, 1.1}}};
    REQUIRE_FALSE(pcms::simplices_have_degenerate_overlap<3>(a, b));
  }
  SECTION("2D, edge-adjacent triangles are rejected")
  {
    const r3d::Few<r3d::Vector<2>, 3> left{{{0, 0}, {0, 1}, {-1, 0}}};
    const r3d::Few<r3d::Vector<2>, 3> right{{{0, 0}, {0, 1}, {1, 0}}};
    REQUIRE(pcms::simplices_have_degenerate_overlap<2>(left, right));
  }
  SECTION("2D, overlapping triangles are not rejected")
  {
    const r3d::Few<r3d::Vector<2>, 3> a{{{0, 0}, {1, 0}, {0, 1}}};
    const r3d::Few<r3d::Vector<2>, 3> b{{{0.1, 0.1}, {1.1, 0.1}, {0.1, 1.1}}};
    REQUIRE_FALSE(pcms::simplices_have_degenerate_overlap<2>(a, b));
  }
}

// The separating-plane pre-filter must not change which source elements are
// found, only how quickly the degenerate pairs are discarded. The unfiltered
// path is the oracle.
TEST_CASE("intersection prefilter preserves the intersection map",
          "[intersection]")
{
  Omega_h::Library lib;
  auto world = lib.world();

  SECTION("2D, identical meshes")
  {
    auto source =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 0, 8, 8, 0, false);
    auto target =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 0, 8, 8, 0, false);
    const auto geometry = std::make_shared<pcms::OmegaHGeometry>(
      source, pcms::csys::CreateCartesian(2));
    require_intersections_identical(
      pcms::intersectTargets(*geometry, target, true),
      pcms::intersectTargets(*geometry, target, false));
  }
  SECTION("2D, nested resolutions")
  {
    auto source =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 0, 12, 12, 0, false);
    auto target =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 0, 4, 4, 0, false);
    const auto geometry = std::make_shared<pcms::OmegaHGeometry>(
      source, pcms::csys::CreateCartesian(2));
    require_intersections_identical(
      pcms::intersectTargets(*geometry, target, true),
      pcms::intersectTargets(*geometry, target, false));
  }
  SECTION("2D, non-nested resolutions")
  {
    auto source =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 0, 7, 11, 0, false);
    auto target =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 0, 5, 3, 0, false);
    const auto geometry = std::make_shared<pcms::OmegaHGeometry>(
      source, pcms::csys::CreateCartesian(2));
    require_intersections_identical(
      pcms::intersectTargets(*geometry, target, true),
      pcms::intersectTargets(*geometry, target, false));
  }
  SECTION("3D, identical meshes")
  {
    auto source =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 4, 4, 4, false);
    auto target =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 4, 4, 4, false);
    const auto geometry = std::make_shared<pcms::OmegaHGeometry>(
      source, pcms::csys::CreateCartesian(3));
    require_intersections_identical(
      pcms::intersectTargets(*geometry, target, true),
      pcms::intersectTargets(*geometry, target, false));
  }
  SECTION("3D, nested resolutions")
  {
    auto source =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 6, 6, 6, false);
    auto target =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 3, 3, 3, false);
    const auto geometry = std::make_shared<pcms::OmegaHGeometry>(
      source, pcms::csys::CreateCartesian(3));
    require_intersections_identical(
      pcms::intersectTargets(*geometry, target, true),
      pcms::intersectTargets(*geometry, target, false));
  }
  SECTION("3D, non-nested resolutions")
  {
    auto source =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 5, 3, 4, false);
    auto target =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 2, 4, 3, false);
    const auto geometry = std::make_shared<pcms::OmegaHGeometry>(
      source, pcms::csys::CreateCartesian(3));
    require_intersections_identical(
      pcms::intersectTargets(*geometry, target, true),
      pcms::intersectTargets(*geometry, target, false));
  }
}

// The source elements per target are a property of the two meshes: at what
// grid resolution the source geometry searches must not show in them.
TEST_CASE("intersection map is independent of the source search resolution",
          "[intersection]")
{
  Omega_h::Library lib;
  auto world = lib.world();

  SECTION("2D, coarse geometry search grid")
  {
    auto source =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 0, 7, 11, 0, false);
    auto target =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 0, 5, 3, 0, false);
    const auto coarse = std::make_shared<pcms::OmegaHGeometry>(
      source, pcms::csys::CreateCartesian(2),
      pcms::omega_h::OmegaHGeometryOptions{.search = pcms::SearchOptions{3}});
    const auto geometry = std::make_shared<pcms::OmegaHGeometry>(
      source, pcms::csys::CreateCartesian(2));
    require_intersections_identical(pcms::intersectTargets(*coarse, target),
                                    pcms::intersectTargets(*geometry, target));
  }
  SECTION("3D, coarse geometry search grid")
  {
    auto source =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 5, 3, 4, false);
    auto target =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 2, 4, 3, false);
    const auto coarse = std::make_shared<pcms::OmegaHGeometry>(
      source, pcms::csys::CreateCartesian(3),
      pcms::omega_h::OmegaHGeometryOptions{.search = pcms::SearchOptions{2}});
    const auto geometry = std::make_shared<pcms::OmegaHGeometry>(
      source, pcms::csys::CreateCartesian(3));
    require_intersections_identical(pcms::intersectTargets(*coarse, target),
                                    pcms::intersectTargets(*geometry, target));
  }
  SECTION("a 2D geometry cannot intersect a 3D target")
  {
    auto flat =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 0, 2, 2, 0, false);
    auto target =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 2, 2, 2, false);
    const auto geometry = std::make_shared<pcms::OmegaHGeometry>(
      flat, pcms::csys::CreateCartesian(2));
    REQUIRE_THROWS_AS(pcms::intersectTargets(*geometry, target),
                      pcms::pcms_error);
  }
}
