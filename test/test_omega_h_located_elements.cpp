#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <pcms/backends/omega_h/omega_h.hpp>
#include <pcms/backends/uniform_grid/geometry.hpp>
#include <pcms/utility/coordinate_systems/cartesian.hpp>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <vector>

using pcms::LO;
using pcms::Real;

namespace
{

std::shared_ptr<const pcms::BoundPoints> BindPoints(
  const pcms::Geometry& geometry, const std::vector<Real>& points)
{
  const auto n = points.size() / 2;
  Kokkos::View<Real**, Kokkos::LayoutRight, pcms::HostMemorySpace> host("host",
                                                                        n, 2);
  for (size_t i = 0; i < n; ++i) {
    host(i, 0) = points[2 * i];
    host(i, 1) = points[2 * i + 1];
  }
  Kokkos::View<Real**, pcms::DeviceMemorySpace> device("device", n, 2);
  pcms::DeepCopyMismatchLayouts(device, host);
  return geometry.Bind(
    pcms::CoordinateView<pcms::DeviceMemorySpace>(
      geometry.GetCoordinateSystem(), pcms::MakeRank2View(device)),
    {pcms::OutOfBoundsMode::FILL, 0.0});
}

// The lowest element whose closure holds (x, y).
LO LowestContainingElement(Omega_h::Mesh& mesh, Real x, Real y)
{
  const Omega_h::HostRead<Omega_h::Real> coords(mesh.coords());
  const Omega_h::HostRead<Omega_h::LO> verts(mesh.ask_elem_verts());
  for (LO e = 0; e < mesh.nelems(); ++e) {
    Real px[3], py[3];
    for (int v = 0; v < 3; ++v) {
      px[v] = coords[2 * verts[3 * e + v]];
      py[v] = coords[2 * verts[3 * e + v] + 1];
    }
    const Real det =
      (px[1] - px[0]) * (py[2] - py[0]) - (px[2] - px[0]) * (py[1] - py[0]);
    const Real l1 =
      ((x - px[0]) * (py[2] - py[0]) - (px[2] - px[0]) * (y - py[0])) / det;
    const Real l2 =
      ((px[1] - px[0]) * (y - py[0]) - (x - px[0]) * (py[1] - py[0])) / det;
    if (l1 >= -1e-12 && l2 >= -1e-12 && 1 - l1 - l2 >= -1e-12) {
      return e;
    }
  }
  return -1;
}

} // namespace

TEST_CASE("omega_h::LocatedElements reports the element of each found point")
{
  auto lib = Omega_h::Library{};
  auto mesh =
    Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1, 1, 0, 2, 2, 0, false);
  const auto geometry =
    pcms::omega_h::CreateGeometry(mesh, pcms::csys::CreateCartesian(2));
  // inside an element, outside, on the edge y = 0.5 between two elements
  const std::vector<Real> points = {0.3, 0.1, 1.5, 0.5, 0.25, 0.5};
  const auto bound = BindPoints(*geometry, points);
  const auto found = Kokkos::create_mirror_view_and_copy(
    pcms::HostMemorySpace(), bound->FoundPoints());
  const auto elements = Kokkos::create_mirror_view_and_copy(
    pcms::HostMemorySpace(), pcms::omega_h::LocatedElements(*bound));
  REQUIRE(found.extent(0) == 2);
  REQUIRE(elements.extent(0) == 2);
  REQUIRE(found(0) == 0);
  REQUIRE(found(1) == 2);
  for (size_t k = 0; k < found.extent(0); ++k) {
    const auto i = static_cast<size_t>(found(k));
    CAPTURE(k, i);
    REQUIRE(elements(k) ==
            LowestContainingElement(mesh, points[2 * i], points[2 * i + 1]));
  }
}

TEST_CASE("omega_h::LocatedElements throws for other geometries")
{
  pcms::UniformGrid<2> grid;
  grid.bot_left = {0.0, 0.0};
  grid.edge_length = {1.0, 1.0};
  grid.divisions = {2, 2};
  const auto geometry =
    pcms::uniform_grid::CreateGeometry(grid, pcms::csys::CreateCartesian(2));
  const auto bound = BindPoints(*geometry, {0.3, 0.1});
  REQUIRE_THROWS_WITH(
    pcms::omega_h::LocatedElements(*bound),
    Catch::Matchers::ContainsSubstring("not bound on an Omega_h geometry"));
}
