#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <pcms/localization/point_search.h>
#include <pcms/localization/uniform_grid_search.hpp>
#include <pcms/utility/mesh_geometry.h>
#include <pcms/utility/uniform_grid.h>
#include "field_test_utils.h"
#include <pcms/field/evaluator/omega_h_geometry.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_build.hpp>
#include <algorithm>
#include <utility>
#include <vector>

using pcms::AABBox;
using pcms::Uniform2DGrid;

namespace
{

/// The element and barycentric coordinates ([point][vertex]) of the results,
/// lifted as the Omega_h binding does, on the host.
template <int Dim, typename Result>
auto LiftOnHost(Omega_h::Mesh& mesh, Kokkos::View<Result*> results)
{
  auto lifted = pcms::detail::LiftResults<Dim>(
    mesh, Kokkos::View<const Result*, pcms::DeviceMemorySpace>(results));
  return std::make_pair(
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, lifted.first),
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, lifted.second));
}

/// Whether `elem` has all of `verts` among its vertices.
bool ElementHasVertices(Omega_h::Mesh& mesh, Omega_h::LO elem,
                        const std::vector<Omega_h::LO>& verts)
{
  const auto elem_verts = Omega_h::HostRead<Omega_h::LO>(mesh.ask_elem_verts());
  for (const auto v : verts) {
    bool found = false;
    for (int k = 0; k <= mesh.dim(); ++k) {
      found = found || elem_verts[elem * (mesh.dim() + 1) + k] == v;
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

/// The point the barycentric coordinates `bary` give in element `elem`.
std::vector<pcms::Real> PositionInElement(
  Omega_h::Mesh& mesh, Omega_h::LO elem,
  Kokkos::View<const pcms::Real**, Kokkos::HostSpace> bary, int row)
{
  const int dim = mesh.dim();
  const auto elem_verts = Omega_h::HostRead<Omega_h::LO>(mesh.ask_elem_verts());
  const auto coords = Omega_h::HostRead<Omega_h::Real>(mesh.coords());
  std::vector<pcms::Real> x(dim, 0.0);
  for (int k = 0; k <= dim; ++k) {
    for (int d = 0; d < dim; ++d) {
      x[d] += bary(row, k) * coords[elem_verts[elem * (dim + 1) + k] * dim + d];
    }
  }
  return x;
}

/// The lowest element adjacent to entity `entity` of dimension `entity_dim`.
Omega_h::LO LowestAdjacentElement(Omega_h::Mesh& mesh, int entity_dim,
                                  Omega_h::LO entity)
{
  const auto up = mesh.ask_up(entity_dim, mesh.dim());
  const auto a2ab = Omega_h::HostRead<Omega_h::LO>(up.a2ab);
  const auto ab2b = Omega_h::HostRead<Omega_h::LO>(up.ab2b);
  Omega_h::LO lowest = ab2b[a2ab[entity]];
  for (Omega_h::LO i = a2ab[entity]; i < a2ab[entity + 1]; ++i) {
    lowest = std::min(lowest, ab2b[i]);
  }
  return lowest;
}

} // namespace

TEST_CASE("global to local")
{
  Omega_h::Matrix<2, 3> coords{{0.0, 0.0}, {1, 0}, {0.5, 1}};
  SECTION("check verts")
  {
    for (int i = 0; i < coords.size(); ++i) {
      auto xi = Omega_h::barycentric_from_global<2, 2>(coords[i], coords);
      Omega_h::Vector<3> hand_xi{0, 0, 0};
      hand_xi[i] = 1;
      printf("[%f,%f,%f] == [%f,%f,%f]\n", xi[0], xi[1], xi[2], hand_xi[0],
             hand_xi[1], hand_xi[2]);
      REQUIRE(Omega_h::are_close(xi, hand_xi));
    }
  }
  SECTION("check point")
  {
    Omega_h::Vector<2> point{0.5, 0.5};
    auto xi = Omega_h::barycentric_from_global<2, 2>(point, coords);
    Omega_h::Vector<3> hand_xi{0.25, 0.25, 0.5};
    printf("[%f,%f,%f] == [%f,%f,%f]\n", xi[0], xi[1], xi[2], hand_xi[0],
           hand_xi[1], hand_xi[2]);
    REQUIRE(are_close(xi, hand_xi));
  }
}
TEST_CASE("Triangle AABB intersection")
{
  using pcms::triangle_intersects_bbox;
  AABBox<2> unit_square{.center = {0, 0}, .half_width = {0.5, 0.5}};
  SECTION("Triangle outside bbox")
  {
    REQUIRE(triangle_intersects_bbox({{1, 1}, {2, 1}, {2, 2}}, unit_square) ==
            false);
  }
  SECTION("BBOX inside Triangle")
  {
    REQUIRE(triangle_intersects_bbox({{-10, -10}, {10, -10}, {10, 10}},
                                     unit_square) == true);
  }
  SECTION("All verts inside BBOX")
  {
    REQUIRE(triangle_intersects_bbox({{-10, -10}, {10, -10}, {10, 10}},
                                     unit_square) == true);
  }
  SECTION("Single vert inside bbox")
  {
    REQUIRE(triangle_intersects_bbox({{0, 0}, {1, 0}, {1, 1}}, unit_square) ==
            true);
    REQUIRE(triangle_intersects_bbox({{1, 1}, {0, 0}, {1, 0}}, unit_square) ==
            true);
    REQUIRE(triangle_intersects_bbox({{1, 0}, {1, 1}, {0, 0}}, unit_square) ==
            true);
  }
  SECTION("edge crossed bbox")
  {
    // Triangle 1 with all vert permutations
    REQUIRE(triangle_intersects_bbox({{-0.6, 0}, {0, 0.6}, {-1, 1}},
                                     unit_square) == true);
    REQUIRE(triangle_intersects_bbox({{-1, 1}, {-0.6, 0}, {0, 0.6}},
                                     unit_square) == true);
    REQUIRE(triangle_intersects_bbox({{0, 0.6}, {-1, 1}, {-0.6, 0}},
                                     unit_square) == true);
    // triangle 2 with all vert permutations
    REQUIRE(triangle_intersects_bbox({{-0.6, 0}, {0.6, 0}, {0, -1}},
                                     unit_square) == true);
    REQUIRE(triangle_intersects_bbox({{0, -1}, {-0.6, 0}, {0.6, 0}},
                                     unit_square) == true);
    REQUIRE(triangle_intersects_bbox({{0.6, 0}, {0, -1}, {-0.6, 0}},
                                     unit_square) == true);
  }
}
TEST_CASE("Triangle BBox Intersection Regression")
{
  using pcms::triangle_intersects_bbox;
  AABBox<2> bbox{.center = {0.10833333, 0.10833333},
                 .half_width = {0.00833333, 0.00833333}};
  REQUIRE(triangle_intersects_bbox({{0, 0}, {0.1, 0}, {0.1, 0.1}}, bbox) ==
          true);
  REQUIRE(triangle_intersects_bbox({{0.1, 0}, {0.2, 0}, {0.2, 0.1}}, bbox) ==
          false);
  REQUIRE(triangle_intersects_bbox({{0.1, 0.1}, {0, 0.1}, {0, 0}}, bbox) ==
          true);
  REQUIRE(triangle_intersects_bbox({{0.1, 0.1}, {0.2, 0.1}, {0.2, 0.2}},
                                   bbox) == true);
  REQUIRE(triangle_intersects_bbox({{0, 0.1}, {0.1, 0.1}, {0.1, 0.2}}, bbox) ==
          true);
  REQUIRE(triangle_intersects_bbox({{0.1, 0.2}, {0, 0.2}, {0, 0.1}}, bbox) ==
          false);
  REQUIRE(triangle_intersects_bbox({{0.2, 0.2}, {0.1, 0.2}, {0.1, 0.1}},
                                   bbox) == true);
  REQUIRE(triangle_intersects_bbox({{0.2, 0.1}, {0.1, 0.1}, {0.1, 0}}, bbox) ==
          true);
}

template <typename T>
bool num_candidates_within_range(const T& intersection_map, pcms::LO min,
                                 pcms::LO max)
{
  using size_type = typename T::size_type;
  using MinMax = Kokkos::MinMax<size_type, Kokkos::HostSpace>;
  using result_type = typename MinMax::value_type;
  result_type result;
  Kokkos::parallel_reduce(
    intersection_map.numRows(),
    KOKKOS_LAMBDA(const int i, result_type& update) {
      auto num_candidates =
        intersection_map.row_map(i + 1) - intersection_map.row_map(i);
      if (num_candidates > update.max_val)
        update.max_val = num_candidates;
      if (num_candidates < update.min_val)
        update.min_val = num_candidates;
    },
    MinMax{result});
  bool within_range = (result.min_val >= min) && (result.max_val <= max);
  if (!within_range)
    std::cerr << result.min_val << ' ' << result.max_val << '\n';
  return within_range;
}
// extern Omega_h::Library omega_h_library;

TEST_CASE("construct intersection map")
{
  // auto world = omega_h_library.world();
  auto lib = Omega_h::Library{};
  auto world = lib.world();
  auto mesh =
    Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 10, 10, 0, false);
  REQUIRE(mesh.dim() == 2);
  SECTION("grid bbox overlap")
  {
    Kokkos::View<Uniform2DGrid[1]> grid_d("uniform grid");
    auto grid_h = Kokkos::create_mirror_view(grid_d);
    grid_h(0) = Uniform2DGrid{
      .edge_length{1, 1}, .bot_left = {0, 0}, .divisions = {10, 10}};
    Kokkos::deep_copy(grid_d, grid_h);
    auto intersection_map = pcms::detail::construct_intersection_map_2d(
      mesh, grid_d, grid_h(0).GetNumCells());
    // assert(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(intersection_map.numRows() == 100);
    REQUIRE(num_candidates_within_range(intersection_map, 2, 16));
  }
  SECTION("fine grid")
  {
    Kokkos::View<Uniform2DGrid[1]> grid_d("uniform grid");
    auto grid_h = Kokkos::create_mirror_view(grid_d);
    grid_h(0) = Uniform2DGrid{
      .edge_length{1, 1}, .bot_left = {0, 0}, .divisions = {60, 60}};
    Kokkos::deep_copy(grid_d, grid_h);
    // require number of candidates is >=1 and <=6
    auto intersection_map = pcms::detail::construct_intersection_map_2d(
      mesh, grid_d, grid_h(0).GetNumCells());

    REQUIRE(intersection_map.numRows() == 3600);
    REQUIRE(num_candidates_within_range(intersection_map, 1, 6));
  }
}
namespace
{
/// Assert two candidate maps are identical: same row offsets and the same
/// entries in the same order within every row.
template <typename T>
void require_maps_identical(const T& fast, const T& reference)
{
  REQUIRE(fast.numRows() == reference.numRows());
  REQUIRE(fast.entries.size() == reference.entries.size());

  auto fast_row_map =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fast.row_map);
  auto ref_row_map =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, reference.row_map);
  auto fast_entries =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fast.entries);
  auto ref_entries =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, reference.entries);

  for (std::size_t i = 0; i < fast_row_map.extent(0); ++i) {
    REQUIRE(fast_row_map(i) == ref_row_map(i));
  }
  for (std::size_t i = 0; i < fast_entries.extent(0); ++i) {
    REQUIRE(fast_entries(i) == ref_entries(i));
  }
}
} // namespace

// The element-major candidate map must produce exactly what the brute-force
// cell-major build does. Entry order matters as well as membership: the search
// returns the first candidate containing the point, so ordering decides which
// element wins a shared-face tie and which is reported for an out-of-bounds
// query.
TEST_CASE("candidate map matches brute-force reference")
{
  auto lib = Omega_h::Library{};
  auto world = lib.world();

  SECTION("2D")
  {
    auto mesh =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 0, 10, 10, 0, false);
    REQUIRE(mesh.dim() == 2);
    for (pcms::LO divisions : {3, 10, 37}) {
      Kokkos::View<Uniform2DGrid[1]> grid_d("uniform grid");
      auto grid_h = Kokkos::create_mirror_view(grid_d);
      grid_h(0) = Uniform2DGrid{.edge_length{1, 1},
                                .bot_left = {0, 0},
                                .divisions = {divisions, divisions}};
      Kokkos::deep_copy(grid_d, grid_h);
      const auto ncells = grid_h(0).GetNumCells();
      require_maps_identical(
        pcms::detail::construct_intersection_map_2d(mesh, grid_d, ncells),
        pcms::detail::construct_intersection_map_reference_2d(mesh, grid_d,
                                                              ncells));
    }
  }
  SECTION("3D")
  {
    auto mesh =
      Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 5, 5, 5, false);
    REQUIRE(mesh.dim() == 3);
    for (pcms::LO divisions : {2, 5, 11}) {
      Kokkos::View<pcms::Uniform3DGrid[1]> grid_d("uniform grid");
      auto grid_h = Kokkos::create_mirror_view(grid_d);
      grid_h(0) =
        pcms::Uniform3DGrid{.edge_length{1, 1, 1},
                            .bot_left = {0, 0, 0},
                            .divisions = {divisions, divisions, divisions}};
      Kokkos::deep_copy(grid_d, grid_h);
      const auto ncells = grid_h(0).GetNumCells();
      require_maps_identical(
        pcms::detail::construct_intersection_map_3d(mesh, grid_d, ncells),
        pcms::detail::construct_intersection_map_reference_3d(mesh, grid_d,
                                                              ncells));
    }
  }
}

TEST_CASE("uniform grid search")
{
  using pcms::GridPointSearch2D;
  auto lib = Omega_h::Library{};
  auto world = lib.world();
  auto mesh =
    Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 10, 10, 0, false);
  auto tolerances =
    GridPointSearch2D::PointSearchTolerances{"point search 2d tolerances"};
  auto tolerances_h = Kokkos::create_mirror_view(tolerances);
  tolerances_h(0) = 0.01;
  tolerances_h(1) = 0.01;
  Kokkos::deep_copy(tolerances, tolerances_h);

  GridPointSearch2D search{mesh, 10, 10, tolerances};

  Kokkos::View<pcms::Real* [2]> points("test_points", 8);
  // Kokkos::View<pcms::Real*[2]> points("test_points", 1);
  auto points_h = Kokkos::create_mirror_view(points);
  points_h(0, 0) = 0;
  points_h(0, 1) = 0;
  points_h(1, 0) = 0.55;
  points_h(1, 1) = 0.54;
  points_h(2, 0) = 100;
  points_h(2, 1) = 100;
  points_h(3, 0) = 1;
  points_h(3, 1) = 1;
  points_h(4, 0) = -1;
  points_h(4, 1) = -1;
  points_h(5, 0) = 1.01;
  points_h(5, 1) = 0.95;
  points_h(6, 0) = 0.05;
  points_h(6, 1) = -0.01;
  points_h(7, 0) = 0.05;
  points_h(7, 1) = 0.02;
  Kokkos::deep_copy(points, points_h);
  auto results = search(points);
  auto results_h = Kokkos::create_mirror_view(results);
  Kokkos::deep_copy(results_h, results);
  auto [elems_h, bary_h] = LiftOnHost<2>(mesh, results);
  const auto edge_verts = Omega_h::HostRead<Omega_h::LO>(
    mesh.ask_down(Omega_h::EDGE, Omega_h::VERT).ab2b);
  const auto mesh_coords = Omega_h::HostRead<Omega_h::Real>(mesh.coords());
  SECTION("global coordinate within mesh")
  {
    {
      auto [dim, idx, parametric] = results_h(0);
      const auto elem = elems_h(0);

      CAPTURE(idx);

      REQUIRE(dim == GridPointSearch2D::Result::Dimensionality::VERTEX);
      REQUIRE(idx == 0);
      // a vertex has no coordinates of its own
      REQUIRE(parametric[0] == 0);
      REQUIRE(parametric[1] == 0);
      // the lift is the lowest element around the vertex, with all weight on
      // the vertex
      REQUIRE(elem == LowestAdjacentElement(mesh, 0, idx));
      REQUIRE(ElementHasVertices(mesh, elem, {idx}));
      const auto x = PositionInElement(mesh, elem, bary_h, 0);
      REQUIRE(x[0] == Catch::Approx(0).margin(1e-14));
      REQUIRE(x[1] == Catch::Approx(0).margin(1e-14));
    }
    {
      auto [dim, idx, parametric] = results_h(1);
      const auto elem = elems_h(1);
      REQUIRE(dim == GridPointSearch2D::Result::Dimensionality::EDGE);
      REQUIRE(idx == 156);
      // the position along the edge is that of the projection of the point
      const auto va = edge_verts[idx * 2];
      const auto vb = edge_verts[idx * 2 + 1];
      const double abx = mesh_coords[vb * 2] - mesh_coords[va * 2];
      const double aby = mesh_coords[vb * 2 + 1] - mesh_coords[va * 2 + 1];
      const double t = ((0.55 - mesh_coords[va * 2]) * abx +
                        (0.54 - mesh_coords[va * 2 + 1]) * aby) /
                       (abx * abx + aby * aby);
      REQUIRE(parametric[0] == Catch::Approx(t));
      REQUIRE(parametric[1] == 0);
      // the lift is the lowest element around the edge, with the weight on the
      // edge's vertices
      REQUIRE(elem == LowestAdjacentElement(mesh, 1, idx));
      REQUIRE(ElementHasVertices(mesh, elem, {va, vb}));
      const auto x = PositionInElement(mesh, elem, bary_h, 1);
      REQUIRE(x[0] == Catch::Approx(mesh_coords[va * 2] + t * abx));
      REQUIRE(x[1] == Catch::Approx(mesh_coords[va * 2 + 1] + t * aby));
    }
    {
      auto [dim, idx, parametric] = results_h(7);
      const auto elem = elems_h(7);
      REQUIRE(dim == GridPointSearch2D::Result::Dimensionality::FACE);
      REQUIRE(idx == 0);
      // a face is its own element, with the reduced coordinates as weights
      REQUIRE(elem == idx);
      REQUIRE(bary_h(7, 0) == Catch::Approx(1 - parametric[0] - parametric[1]));
      REQUIRE(bary_h(7, 1) == Catch::Approx(parametric[0]));
      REQUIRE(bary_h(7, 2) == Catch::Approx(parametric[1]));
      const auto x = PositionInElement(mesh, elem, bary_h, 7);
      REQUIRE(x[0] == Catch::Approx(0.05));
      REQUIRE(x[1] == Catch::Approx(0.02));
    }
  }
  SECTION("Global coordinate outside mesh")
  {
    auto out_of_bounds = results_h(2);
    auto top_right = results_h(3);
    REQUIRE(out_of_bounds.dimensionality ==
            GridPointSearch2D::Result::Dimensionality::VERTEX);
    REQUIRE(out_of_bounds.IsOutside());
    REQUIRE(out_of_bounds.EntityId() == top_right.entity);
    REQUIRE(elems_h(2) >= 0);
    REQUIRE(ElementHasVertices(mesh, elems_h(2), {out_of_bounds.EntityId()}));

    out_of_bounds = results_h(4);
    auto bot_left = results_h(0);
    REQUIRE(out_of_bounds.dimensionality ==
            GridPointSearch2D::Result::Dimensionality::VERTEX);
    REQUIRE(out_of_bounds.IsOutside());
    REQUIRE(out_of_bounds.EntityId() == bot_left.entity);
    REQUIRE(elems_h(4) >= 0);
    REQUIRE(ElementHasVertices(mesh, elems_h(4), {out_of_bounds.EntityId()}));

    out_of_bounds = results_h(5);
    REQUIRE(out_of_bounds.dimensionality ==
            GridPointSearch2D::Result::Dimensionality::EDGE);
    REQUIRE(out_of_bounds.IsOutside());
    REQUIRE(out_of_bounds.EntityId() == 219);
    REQUIRE(elems_h(5) >= 0);
    REQUIRE(ElementHasVertices(mesh, elems_h(5),
                               {edge_verts[219 * 2], edge_verts[219 * 2 + 1]}));
    // the outside point's position on its nearest edge is its projection
    REQUIRE(out_of_bounds.parametric[0] >= 0);
    REQUIRE(out_of_bounds.parametric[0] <= 1);

    // (0.05, -0.01) is within the edge tolerance of the bottom edge, so it is
    // located on that edge rather than reported outside.
    auto on_edge = results_h(6);
    REQUIRE(on_edge.dimensionality ==
            GridPointSearch2D::Result::Dimensionality::EDGE);
    REQUIRE_FALSE(on_edge.IsOutside());
    REQUIRE(on_edge.entity == 0);
    REQUIRE(elems_h(6) >= 0);
    REQUIRE(
      ElementHasVertices(mesh, elems_h(6), {edge_verts[0], edge_verts[1]}));
  }
  SECTION("point on extension of an edge")
  {
    Kokkos::View<pcms::Real* [2]> ext_points("ext_test_points", 1);
    auto ext_h = Kokkos::create_mirror_view(ext_points);
    ext_h(0, 0) = 1.5;
    ext_h(0, 1) = 0.0;
    Kokkos::deep_copy(ext_points, ext_h);
    auto ext_results = search(ext_points);
    auto ext_results_h = Kokkos::create_mirror_view(ext_results);
    Kokkos::deep_copy(ext_results_h, ext_results);
    auto [ext_elems_h, ext_bary_h] = LiftOnHost<2>(mesh, ext_results);

    auto res = ext_results_h(0);
    // Must be out-of-bounds (negative entity id)
    REQUIRE(res.entity < 0);
    // Dimensionality must be VERTEX — the nearest entity is the
    // rightmost bottom vertex (1.0, 0).
    REQUIRE(res.dimensionality ==
            GridPointSearch2D::Result::Dimensionality::VERTEX);
    // The lift still resolves to a valid face around that vertex
    REQUIRE(ext_elems_h(0) >= 0);
    REQUIRE(ElementHasVertices(mesh, ext_elems_h(0), {res.EntityId()}));
  }
}

TEST_CASE("point search locates points on shared faces",
          "[point_search][3d][regression]")
{
  using pcms::GridPointSearch3D;
  auto lib = Omega_h::Library{};
  auto world = lib.world();

  const int source_divisions = 10;
  const int query_divisions = 5;
  auto mesh =
    Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, source_divisions,
                       source_divisions, source_divisions, false);
  auto query_mesh =
    Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, query_divisions,
                       query_divisions, query_divisions, false);

  // Same grid resolution the intersection search uses for this mesh.
  const auto n = pcms::DivisionsPerAxisForMesh<3>(mesh.nelems());
  GridPointSearch3D search{mesh, n, n, n};

  const auto centroids_h =
    Omega_h::HostRead<Omega_h::Real>(pcms::get_entity_centroids(query_mesh, 3));
  const auto npoints = query_mesh.nelems();

  Kokkos::View<pcms::Real* [3]> points("query points", npoints);
  auto points_h = Kokkos::create_mirror_view(points);
  for (Omega_h::LO i = 0; i < npoints; ++i) {
    for (int d = 0; d < 3; ++d) {
      points_h(i, d) = centroids_h[i * 3 + d];
    }
  }
  Kokkos::deep_copy(points, points_h);

  auto results = search(points);
  auto results_h = Kokkos::create_mirror_view(results);
  Kokkos::deep_copy(results_h, results);

  std::vector<Omega_h::LO> not_located;
  for (Omega_h::LO i = 0; i < npoints; ++i) {
    if (results_h(i).entity < 0) {
      not_located.push_back(i);
    }
  }
  CAPTURE(not_located.size(), npoints);
  REQUIRE(not_located.empty());
}

TEST_CASE("point search encodes outside points unambiguously",
          "[point_search][regression]")
{
  auto lib = Omega_h::Library{};

  SECTION("3D: nearest tetrahedron 0")
  {
    using pcms::GridPointSearch3D;
    using Dimensionality = GridPointSearch3D::Result::Dimensionality;
    auto mesh = pcms::test::BuildReferenceTet(lib);
    const auto n = pcms::DivisionsPerAxisForMesh<3>(mesh.nelems());
    GridPointSearch3D search{mesh, n, n, n};

    Kokkos::View<pcms::Real* [3]> points("points", 2);
    auto points_h = Kokkos::create_mirror_view(points);
    // inside the tet
    points_h(0, 0) = 0.1;
    points_h(0, 1) = 0.1;
    points_h(0, 2) = 0.1;
    // in the tet's bounding box but outside it: the nearest candidate is tet 0
    points_h(1, 0) = 0.5;
    points_h(1, 1) = 0.5;
    points_h(1, 2) = 0.5;
    Kokkos::deep_copy(points, points_h);

    auto results = search(points);
    auto results_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
    const auto owners_h = LiftOnHost<3>(mesh, results).first;

    REQUIRE(results_h(0).dimensionality == Dimensionality::REGION);
    REQUIRE_FALSE(results_h(0).IsOutside());
    REQUIRE(results_h(0).entity == 0);
    REQUIRE(results_h(0).EntityId() == 0);
    // a region is its own element
    REQUIRE(owners_h(0) == 0);

    REQUIRE(results_h(1).dimensionality == Dimensionality::REGION);
    REQUIRE(results_h(1).IsOutside());
    REQUIRE(results_h(1).EntityId() == 0);
    REQUIRE(owners_h(1) == 0);

    // The grid over a single tet has no empty cell, so decode the
    // no-candidate result directly.
    Kokkos::View<GridPointSearch3D::Result*> none("none", 1);
    Kokkos::deep_copy(none,
                      GridPointSearch3D::Result{Dimensionality::NO_INTERSECT,
                                                -1, Omega_h::zero_vector<3>()});
    auto none_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, none);
    const auto none_owners_h = LiftOnHost<3>(mesh, none).first;
    REQUIRE(none_h(0).IsOutside());
    REQUIRE(none_h(0).EntityId() == -1);
    REQUIRE(none_owners_h(0) == -1);
  }

  SECTION("2D: nearest vertex 0 and an empty query cell")
  {
    using pcms::GridPointSearch2D;
    using Dimensionality = GridPointSearch2D::Result::Dimensionality;
    // Only the triangle below the diagonal, so the cells above it hold no
    // candidates.
    auto mesh = pcms::test::BuildUnitSquare(lib, Omega_h::LOs({0, 1, 3}));
    const auto n = pcms::DivisionsPerAxisForMesh<2>(mesh.nelems());
    GridPointSearch2D search{mesh, n, n};

    Kokkos::View<pcms::Real* [2]> points("points", 3);
    auto points_h = Kokkos::create_mirror_view(points);
    // beyond vertex 0: the nearest entity is vertex 0
    points_h(0, 0) = -0.05;
    points_h(0, 1) = -0.05;
    // in a grid cell the triangle does not reach
    points_h(1, 0) = 0.6;
    points_h(1, 1) = 0.6;
    // inside the triangle
    points_h(2, 0) = 0.2;
    points_h(2, 1) = 0.2;
    Kokkos::deep_copy(points, points_h);

    auto results = search(points);
    auto results_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, results);
    const auto owners_h = LiftOnHost<2>(mesh, results).first;

    REQUIRE(results_h(0).dimensionality == Dimensionality::VERTEX);
    REQUIRE(results_h(0).IsOutside());
    REQUIRE(results_h(0).EntityId() == 0);
    REQUIRE(owners_h(0) == 0);

    REQUIRE(results_h(1).dimensionality == Dimensionality::NO_INTERSECT);
    REQUIRE(results_h(1).IsOutside());
    REQUIRE(results_h(1).EntityId() == -1);
    REQUIRE(owners_h(1) == -1);

    REQUIRE(results_h(2).dimensionality == Dimensionality::FACE);
    REQUIRE_FALSE(results_h(2).IsOutside());
    REQUIRE(results_h(2).entity == 0);
    REQUIRE(owners_h(2) == 0);
  }
}

TEST_CASE("uniform grid search reports the cell and the position in it",
          "[point_search][uniform_grid]")
{
  using Search = pcms::UniformGridPointSearch<2>;
  // 4 x 2 cells of size 0.5 x 1 over [0, 2] x [0, 2]
  const pcms::UniformGrid<2> grid{
    .edge_length = {2.0, 2.0}, .bot_left = {0.0, 0.0}, .divisions = {4, 2}};
  Kokkos::View<pcms::Real* [2]> points("points", 3);
  auto points_h = Kokkos::create_mirror_view(points);
  // in the cell at (1, 0) in axis order, a quarter and a half along its axes
  points_h(0, 0) = 0.5 + 0.125;
  points_h(0, 1) = 0.5;
  // on the grid's far corner
  points_h(1, 0) = 2.0;
  points_h(1, 1) = 2.0;
  // outside, a cell width beyond the near edge of the first cell
  points_h(2, 0) = -0.5;
  points_h(2, 1) = 0.5;
  Kokkos::deep_copy(points, points_h);

  const auto results = Kokkos::create_mirror_view_and_copy(
    Kokkos::HostSpace{},
    Search(grid)(
      Kokkos::View<const pcms::Real**, pcms::DeviceMemorySpace>(points)));

  const auto inside = results(0);
  REQUIRE_FALSE(inside.IsOutside());
  REQUIRE(inside.cell == grid.ClosestCellID(Omega_h::Vector<2>{0.625, 0.5}));
  REQUIRE(inside.parametric[0] == Catch::Approx(0.25));
  REQUIRE(inside.parametric[1] == Catch::Approx(0.5));

  // the grid is closed: its far corner is in its last cell, at the corner
  const auto corner = results(1);
  REQUIRE_FALSE(corner.IsOutside());
  REQUIRE(corner.cell == 7);
  REQUIRE(corner.parametric[0] == Catch::Approx(1.0));
  REQUIRE(corner.parametric[1] == Catch::Approx(1.0));

  // a point outside gets its nearest cell and its position relative to it
  const auto outside = results(2);
  REQUIRE(outside.IsOutside());
  REQUIRE(outside.CellId() == 0);
  REQUIRE(outside.cell == Search::Result::OutsideId(0));
  REQUIRE(outside.parametric[0] == Catch::Approx(-1.0));
  REQUIRE(outside.parametric[1] == Catch::Approx(0.5));
}
