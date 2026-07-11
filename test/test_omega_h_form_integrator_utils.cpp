#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <pcms/transfer/omega_h_form_integrator_utils.hpp>

namespace
{

r3d::Polytope<2> MakePoly(int nverts,
                          std::initializer_list<std::pair<double, double>> pts)
{
  r3d::Polytope<2> poly;
  poly.nverts = nverts;
  int i = 0;
  for (auto [x, y] : pts) {
    poly.verts[i].pos[0] = x;
    poly.verts[i].pos[1] = y;
    poly.verts[i].pnbrs[0] = -1;
    poly.verts[i].pnbrs[1] = -1;
    ++i;
  }
  return poly;
}

} // namespace

TEST_CASE(
  "RemoveDuplicateVerticesAndFixLinks: no duplicates leaves triangle intact",
  "[form_integrator_utils]")
{
  // A clean CCW triangle — nothing to remove, area should equal 0.5.
  auto poly = MakePoly(3, {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}});
  const int n = pcms::detail::RemoveDuplicateVerticesAndFixLinks(poly);
  REQUIRE(n == 3);
  REQUIRE(poly.nverts == 3);
  // The function sets correct pnbrs, so r3d::measure is valid after the call.
  const double area = r3d::measure(poly);
  REQUIRE(area == Catch::Approx(0.5).epsilon(1e-12));
}

TEST_CASE("RemoveDuplicateVerticesAndFixLinks: one duplicate vertex is removed",
          "[form_integrator_utils]")
{
  // 4-vertex polygon; last vertex duplicates the first within the default tol.
  // After dedup: 3 unique vertices, area of the underlying triangle preserved.
  auto poly = MakePoly(4, {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {5e-13, 0.0}});
  const int n = pcms::detail::RemoveDuplicateVerticesAndFixLinks(poly);
  REQUIRE(n == 3);
  REQUIRE(poly.nverts == 3);
  const double area = r3d::measure(poly);
  REQUIRE(area == Catch::Approx(0.5).epsilon(1e-10));
}

TEST_CASE(
  "RemoveDuplicateVerticesAndFixLinks: polygon becomes degenerate after dedup",
  "[form_integrator_utils]")
{
  // 3 vertices where two positions are identical — only 2 unique remain.
  // The function must mark the polygon degenerate: all pnbrs set to -1.
  auto poly = MakePoly(3, {{0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}});
  const int n = pcms::detail::RemoveDuplicateVerticesAndFixLinks(poly);
  REQUIRE(n == 2);
  REQUIRE(poly.nverts == 2);
  for (int i = 0; i < 2; ++i) {
    CAPTURE(i);
    CHECK(poly.verts[i].pnbrs[0] == -1);
    CHECK(poly.verts[i].pnbrs[1] == -1);
  }
}

TEST_CASE(
  "RemoveDuplicateVerticesAndFixLinks: near-tolerance vertex is retained",
  "[form_integrator_utils]")
{
  // Vertex at (0, 2e-12) differs from (0, 0) by 2e-12 in y, which exceeds the
  // default tol=1e-12, so it must NOT be treated as a duplicate.
  auto poly = MakePoly(3, {{0.0, 0.0}, {1.0, 0.0}, {0.0, 2e-12}});
  const int n = pcms::detail::RemoveDuplicateVerticesAndFixLinks(poly);
  REQUIRE(n == 3);
  REQUIRE(poly.nverts == 3);
}
