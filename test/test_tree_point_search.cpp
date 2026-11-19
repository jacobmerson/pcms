// The ArborX bounding volume hierarchy point search reports the records of the
// grid search (built with PCMS_ENABLE_ARBORX).
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <pcms/backends/omega_h/point_search.hpp>
#include <pcms/backends/omega_h/omega_h_geometry.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_build.hpp>
#include <array>
#include <random>
#include <vector>

namespace
{
using pcms::LO;
using pcms::Real;

template <int Dim>
Omega_h::Mesh BuildBox(Omega_h::Library& lib, int n)
{
  auto world = lib.world();
  if constexpr (Dim == 2) {
    return Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, n, n, 0, false);
  } else {
    return Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, n, n, n, false);
  }
}

template <int Dim>
using Points = std::vector<std::array<Real, Dim>>;

/// Random points in and around the unit box, the mesh's vertices and the
/// midpoints of its edges.
template <int Dim>
Points<Dim> MakePoints(Omega_h::Mesh& mesh)
{
  Points<Dim> pts;
  std::mt19937 gen(42);
  std::uniform_real_distribution<Real> dist(-0.2, 1.2);
  for (int i = 0; i < 400; ++i) {
    std::array<Real, Dim> p;
    for (auto& x : p) {
      x = dist(gen);
    }
    pts.push_back(p);
  }
  const auto coords = Omega_h::HostRead<Real>(mesh.coords());
  for (LO v = 0; v < mesh.nverts(); ++v) {
    std::array<Real, Dim> p;
    for (int d = 0; d < Dim; ++d) {
      p[d] = coords[v * Dim + d];
    }
    pts.push_back(p);
  }
  const auto edge_verts =
    Omega_h::HostRead<LO>(mesh.ask_down(Omega_h::EDGE, Omega_h::VERT).ab2b);
  for (LO e = 0; e < mesh.nedges(); ++e) {
    std::array<Real, Dim> p;
    for (int d = 0; d < Dim; ++d) {
      p[d] = 0.5 * (coords[edge_verts[2 * e] * Dim + d] +
                    coords[edge_verts[2 * e + 1] * Dim + d]);
    }
    pts.push_back(p);
  }
  return pts;
}

template <int Dim, typename Search>
auto SearchOnHost(const Search& search, const Points<Dim>& pts,
                  const std::vector<LO>& guess_values = {})
{
  Kokkos::View<Real* [Dim]> points("points", pts.size());
  auto points_h = Kokkos::create_mirror_view(points);
  for (std::size_t i = 0; i < pts.size(); ++i) {
    for (int d = 0; d < Dim; ++d) {
      points_h(i, d) = pts[i][d];
    }
  }
  Kokkos::deep_copy(points, points_h);
  using Result = typename Search::Result;
  Kokkos::View<Result*> results("results", pts.size());
  if (guess_values.empty()) {
    results = search(points);
  } else {
    Kokkos::View<LO*> guesses("guesses", guess_values.size());
    auto guesses_h = Kokkos::create_mirror_view(guesses);
    for (std::size_t i = 0; i < guess_values.size(); ++i) {
      guesses_h(i) = guess_values[i];
    }
    Kokkos::deep_copy(guesses, guesses_h);
    search(points, guesses, results);
  }
  auto results_h = Kokkos::create_mirror_view(results);
  Kokkos::deep_copy(results_h, results);
  return results_h;
}

/// The records agree for every point in the mesh; a point outside is outside
/// for both.
template <int Dim, typename Grid, typename Tree>
void CheckSame(const Grid& grid, const Tree& tree, const Points<Dim>& pts,
               const std::vector<LO>& guesses = {})
{
  const auto g = SearchOnHost<Dim>(grid, pts, guesses);
  const auto t = SearchOnHost<Dim>(tree, pts, guesses);
  int inside = 0;
  for (std::size_t i = 0; i < pts.size(); ++i) {
    if (g(i).IsOutside()) {
      CHECK(t(i).IsOutside());
      continue;
    }
    ++inside;
    CHECK(t(i).dimensionality == g(i).dimensionality);
    CHECK(t(i).entity == g(i).entity);
    for (int d = 0; d < Dim; ++d) {
      CHECK(t(i).parametric[d] == Catch::Approx(g(i).parametric[d]));
    }
  }
  CHECK(inside > static_cast<int>(pts.size()) / 4);
}
} // namespace

TEST_CASE("bounding volume hierarchy search equals the grid search",
          "[point_search][arborx]")
{
  auto lib = Omega_h::Library{};
  const Real tau = 1e-9;
  SECTION("2D")
  {
    auto mesh = BuildBox<2>(lib, 10);
    pcms::omega_h::GridPointSearch2D grid{mesh, 10, 10, tau};
    pcms::omega_h::TreePointSearch2D tree{mesh, tau};
    CheckSame<2>(grid, tree, MakePoints<2>(mesh));
  }
  SECTION("3D")
  {
    auto mesh = BuildBox<3>(lib, 4);
    pcms::omega_h::GridPointSearch3D grid{mesh, 5, 5, 5, tau};
    pcms::omega_h::TreePointSearch3D tree{mesh, tau};
    CheckSame<3>(grid, tree, MakePoints<3>(mesh));
  }
}

TEST_CASE("bounding volume hierarchy search with guessed elements",
          "[point_search][arborx]")
{
  auto lib = Omega_h::Library{};
  const Real tau = 1e-9;
  auto mesh = BuildBox<2>(lib, 8);
  pcms::omega_h::GridPointSearch2D grid{mesh, 8, 8, tau};
  pcms::omega_h::TreePointSearch2D tree{mesh, tau};
  const auto pts = MakePoints<2>(mesh);
  // every other point with a wrong guess, the others with none
  std::vector<LO> guesses(pts.size(), -1);
  for (std::size_t i = 0; i < pts.size(); i += 2) {
    guesses[i] = static_cast<LO>(i % mesh.nelems());
  }
  CheckSame<2>(grid, tree, pts, guesses);
}

TEST_CASE("bounding volume hierarchy search outside the mesh",
          "[point_search][arborx]")
{
  auto lib = Omega_h::Library{};
  auto mesh = BuildBox<2>(lib, 4);
  pcms::omega_h::TreePointSearch2D tree{mesh, 1e-9};
  const auto r = SearchOnHost<2>(tree, Points<2>{{5., 5.}, {-3., 0.5}});
  for (int i = 0; i < 2; ++i) {
    CHECK(r(i).IsOutside());
    CHECK(
      r(i).dimensionality ==
      pcms::omega_h::TreePointSearch2D::Result::Dimensionality::NO_INTERSECT);
  }
}
