#include <catch2/catch_test_macros.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>

#include <pcms/transfer/mesh_intersection.hpp>

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
    require_intersections_identical(
      pcms::intersectTargets(source, target, true),
      pcms::intersectTargets(source, target, false));
  }
  SECTION("2D, nested resolutions")
  {
    auto source =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 0, 12, 12, 0, false);
    auto target =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 0, 4, 4, 0, false);
    require_intersections_identical(
      pcms::intersectTargets(source, target, true),
      pcms::intersectTargets(source, target, false));
  }
  SECTION("2D, non-nested resolutions")
  {
    auto source =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 0, 7, 11, 0, false);
    auto target =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 0, 5, 3, 0, false);
    require_intersections_identical(
      pcms::intersectTargets(source, target, true),
      pcms::intersectTargets(source, target, false));
  }
  SECTION("3D, identical meshes")
  {
    auto source =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 4, 4, 4, false);
    auto target =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 4, 4, 4, false);
    require_intersections_identical(
      pcms::intersectTargets(source, target, true),
      pcms::intersectTargets(source, target, false));
  }
  SECTION("3D, nested resolutions")
  {
    auto source =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 6, 6, 6, false);
    auto target =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 3, 3, 3, false);
    require_intersections_identical(
      pcms::intersectTargets(source, target, true),
      pcms::intersectTargets(source, target, false));
  }
  SECTION("3D, non-nested resolutions")
  {
    auto source =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 5, 3, 4, false);
    auto target =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 2, 4, 3, false);
    require_intersections_identical(
      pcms::intersectTargets(source, target, true),
      pcms::intersectTargets(source, target, false));
  }
}
