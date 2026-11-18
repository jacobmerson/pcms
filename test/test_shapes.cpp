#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <pcms/utility/shapes.hpp>
#include <pcms/localization/overlap.hpp>

#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>

#include <type_traits>

// Tests of the shapes and of the overlap operations on them: every operation is
// checked on its own, and each case of the exact triangle-box test is reached
// through a configuration only that case can detect.

using pcms::AABBox;
using pcms::Point;
using pcms::Real;
using pcms::Simplex;

namespace
{
Simplex<2> Triangle(Real x0, Real y0, Real x1, Real y1, Real x2, Real y2)
{
  Simplex<2> s;
  s.vertices(0, 0) = x0;
  s.vertices(1, 0) = y0;
  s.vertices(0, 1) = x1;
  s.vertices(1, 1) = y1;
  s.vertices(0, 2) = x2;
  s.vertices(1, 2) = y2;
  return s;
}

AABBox<2> Box(Real lo_x, Real hi_x, Real lo_y, Real hi_y)
{
  return {.center = {(lo_x + hi_x) / 2, (lo_y + hi_y) / 2},
          .half_width = {(hi_x - lo_x) / 2, (hi_y - lo_y) / 2}};
}
} // namespace

TEST_CASE("shape traits give the tag and the dimension")
{
  static_assert(std::is_same_v<pcms::tag_t<Point<2>>, pcms::PointTag>);
  static_assert(std::is_same_v<pcms::tag_t<AABBox<3>>, pcms::BoxTag>);
  static_assert(std::is_same_v<pcms::tag_t<Simplex<2>>, pcms::SimplexTag>);
  static_assert(
    std::is_same_v<pcms::tag_t<pcms::Segment<2>>, pcms::SegmentTag>);
  static_assert(pcms::dimension_v<Point<4>> == 4);
  static_assert(pcms::dimension_v<AABBox<3>> == 3);
  static_assert(pcms::dimension_v<Simplex<2>> == 2);
  SUCCEED();
}

TEST_CASE("point against box and simplex")
{
  const auto box = Box(0, 1, 0, 1);
  const auto tri = Triangle(0, 0, 1, 0, 0, 1);

  REQUIRE(pcms::overlaps(Point<2>{{0.5, 0.5}}, box));
  REQUIRE(pcms::overlaps(Point<2>{{1.0, 0.5}}, box)); // closed: on the edge
  REQUIRE_FALSE(pcms::overlaps(Point<2>{{1.5, 0.5}}, box));

  REQUIRE(pcms::overlaps(Point<2>{{0.2, 0.2}}, tri));
  REQUIRE_FALSE(pcms::overlaps(Point<2>{{0.8, 0.8}}, tri));

  // Either operand order.
  REQUIRE(pcms::overlaps(box, Point<2>{{0.5, 0.5}}));
  REQUIRE(pcms::overlaps(tri, Point<2>{{0.2, 0.2}}));
  REQUIRE_FALSE(pcms::overlaps(tri, Point<2>{{0.8, 0.8}}));
}

TEST_CASE("segment against box")
{
  using pcms::Segment;
  const auto box = Box(0, 1, 0, 1);
  // Crossing the box without either end inside it.
  REQUIRE(pcms::overlaps(Segment<2>{{-1.0, 0.5}, {2.0, 0.5}}, box));
  // Entirely inside.
  REQUIRE(pcms::overlaps(Segment<2>{{0.2, 0.2}, {0.8, 0.8}}, box));
  // Passing beside the box.
  REQUIRE_FALSE(pcms::overlaps(Segment<2>{{-1.0, 1.5}, {2.0, 1.5}}, box));
  // Closed: touching an edge counts.
  REQUIRE(pcms::overlaps(Segment<2>{{1.0, -1.0}, {1.0, 2.0}}, box));
  // Either operand order.
  REQUIRE(pcms::overlaps(box, Segment<2>{{-1.0, 0.5}, {2.0, 0.5}}));
  REQUIRE_FALSE(pcms::overlaps(box, Segment<2>{{-1.0, 1.5}, {2.0, 1.5}}));
}

TEST_CASE("exact triangle-box test, one case at a time")
{
  const auto tri = Triangle(0, 0, 4, 0, 0, 4);

  SECTION("a triangle vertex inside the box")
  {
    REQUIRE(pcms::overlaps(tri, Box(3.5, 5, -1, 0.5)));
  }
  SECTION("a box corner inside the triangle")
  {
    REQUIRE(pcms::overlaps(tri, Box(1, 2, 1, 2)));
  }
  SECTION("only an edge crosses the box")
  {
    // A vertical strip across the base: no triangle vertex is in it, and none
    // of its corners is in the triangle.
    const auto flat = Triangle(0, 0, 4, 0, 0, 1);
    REQUIRE(pcms::overlaps(flat, Box(1.9, 2.1, -1, 3)));
  }
  SECTION("disjoint although the bounding boxes overlap")
  {
    const auto corner = Box(3, 4, 3, 4);
    REQUIRE_FALSE(pcms::overlaps(tri, corner));
    // Their bounding boxes do overlap: in 2D the triangle itself is tested.
    REQUIRE(
      pcms::overlaps(pcms::detail::simplex_bbox<2>(tri.vertices), corner));
  }
  SECTION("either operand order")
  {
    REQUIRE(pcms::overlaps(Box(1, 2, 1, 2), tri));
    REQUIRE_FALSE(pcms::overlaps(Box(3, 4, 3, 4), tri));
  }
}

TEST_CASE("simplex-box overlap outside 2D tests the bounding box")
{
  // The tetrahedron (0,0,0), (1,0,0), (0,1,0), (0,0,1) and a box in its
  // bounding box's far corner, which the tetrahedron itself does not reach.
  Simplex<3> tet{};
  for (int k = 1; k < 4; ++k) {
    tet.vertices(k - 1, k) = 1.0;
  }
  const AABBox<3> far_corner{.center = {0.9, 0.9, 0.9},
                             .half_width = {0.05, 0.05, 0.05}};
  REQUIRE(pcms::overlaps(tet, far_corner));
  const AABBox<3> outside{.center = {2.0, 2.0, 2.0},
                          .half_width = {0.5, 0.5, 0.5}};
  REQUIRE_FALSE(pcms::overlaps(tet, outside));
}

TEST_CASE("mesh simplices are the mesh's elements")
{
  auto lib = Omega_h::Library{};
  auto mesh =
    Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1, 1, 0, 1, 1, 0, false);
  const auto coords = mesh.coords();
  const auto elems2verts = mesh.ask_elem_verts();
  const auto nelems = mesh.nelems();

  // Built inside a kernel, as the candidate map does. Both triangles of the
  // unit square span the whole square.
  Kokkos::View<Real* [4]> boxes("element boxes", nelems);
  Kokkos::parallel_for(
    nelems, KOKKOS_LAMBDA(int e) {
      const auto bbox = pcms::detail::simplex_bbox<2>(
        Simplex<2>::FromElement(coords, elems2verts, e).vertices);
      boxes(e, 0) = bbox.center[0];
      boxes(e, 1) = bbox.center[1];
      boxes(e, 2) = bbox.half_width[0];
      boxes(e, 3) = bbox.half_width[1];
    });
  const auto boxes_h =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, boxes);
  REQUIRE(nelems == 2);
  for (int e = 0; e < nelems; ++e) {
    for (int k = 0; k < 4; ++k) {
      REQUIRE(boxes_h(e, k) == Catch::Approx(0.5));
    }
  }
}

TEST_CASE("simplex-box overlap above 3D", "[nd]")
{
  SECTION("4D simplex-vs-box overlap needs every axis to overlap")
  {
    const AABBox<4> box{.center = {1.0, 1.0, 1.0, 1.0},
                        .half_width = {0.5, 0.5, 0.5, 0.5}};
    // A simplex spanning [0.8, 1.2] on axes 0-2. On axis 3 it either reaches
    // into the box or stops short of it.
    const auto simplex = [](pcms::Real axis3_lo, pcms::Real axis3_hi) {
      Omega_h::Matrix<4, 5> coords{};
      for (int v = 0; v < 5; ++v) {
        for (int d = 0; d < 3; ++d) {
          coords(d, v) = (v % 2 == 0) ? 0.8 : 1.2;
        }
        coords(3, v) = (v == 0) ? axis3_lo : axis3_hi;
      }
      return coords;
    };
    REQUIRE(pcms::overlaps(pcms::Simplex<4>{simplex(1.4, 2.0)}, box));
    // Overlapping on axes 0-2 but entirely past the box on axis 3.
    REQUIRE_FALSE(pcms::overlaps(pcms::Simplex<4>{simplex(1.6, 2.0)}, box));
  }
  SECTION("simplex-box overlap in 4D")
  {
    const AABBox<4> box{.center = {1.0, 1.0, 1.0, 1.0},
                        .half_width = {0.5, 0.5, 0.5, 0.5}};

    // A degenerate simplex: all 5 vertices at a point inside the box.
    Omega_h::Matrix<4, 5> inside{};
    for (int v = 0; v < 5; ++v) {
      for (int d = 0; d < 4; ++d) {
        inside(d, v) = 1.0;
      }
    }
    REQUIRE(overlaps(Simplex<4>{inside}, box));

    // Far outside on every axis.
    Omega_h::Matrix<4, 5> outside{};
    for (int v = 0; v < 5; ++v) {
      for (int d = 0; d < 4; ++d) {
        outside(d, v) = 100.0;
      }
    }
    REQUIRE_FALSE(overlaps(Simplex<4>{outside}, box));

    // Straddling the box on one axis only: the bounding-box test is true.
    Omega_h::Matrix<4, 5> straddling{};
    for (int v = 0; v < 5; ++v) {
      for (int d = 0; d < 4; ++d) {
        straddling(d, v) = 1.0;
      }
    }
    straddling(3, 0) = 0.5;
    straddling(3, 1) = 1.9;
    REQUIRE(overlaps(Simplex<4>{straddling}, box));
  }

  SECTION("simplex-box overlap in 5D")
  {
    const AABBox<5> box{.center = {2.0, 2.0, 2.0, 2.0, 2.0},
                        .half_width = {1.0, 1.0, 1.0, 1.0, 1.0}};
    Omega_h::Matrix<5, 6> inside{};
    for (int v = 0; v < 6; ++v) {
      for (int d = 0; d < 5; ++d) {
        inside(d, v) = 2.0;
      }
    }
    REQUIRE(overlaps(Simplex<5>{inside}, box));
  }
}
