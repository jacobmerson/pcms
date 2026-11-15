#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <Kokkos_Core.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_shape.hpp>
#include <pcms/configuration.h>
#include <pcms/field/coordinate_systems/cartesian.hpp>
#include <pcms/field/coordinate_systems/cylindrical.hpp>
#include <pcms/field/evaluator/mls_options.h>
#include <pcms/field/evaluator/omega_h_geometry.hpp>
#include <pcms/field/evaluator/uniform_grid_geometry.hpp>
#include <pcms/field/function_space/lagrange.h>
#include <pcms/geometry/geometry.hpp>
#include <pcms/localization/adj_search.hpp>
#include <pcms/localization/mesh_localization.h>
#include "field_test_utils.h"
#include <memory>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using pcms::DeviceMemorySpace;
using pcms::HostMemorySpace;
using pcms::LagrangeFunctionSpace;
using pcms::LO;
using pcms::Real;

namespace
{

Omega_h::Mesh BuildSquareMesh(Omega_h::Library& lib, int n)
{
  return Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 1.0, 1.0, 0.0, n, n,
                            0);
}

std::shared_ptr<pcms::OmegaHGeometry> MakeGeometry(
  Omega_h::Mesh& mesh, pcms::SearchOptions options = {})
{
  return pcms::OmegaHGeometry::Create(mesh, pcms::csys::Cartesian::Create(2),
                                      options);
}

std::shared_ptr<LagrangeFunctionSpace> MakeSpace(
  const std::shared_ptr<pcms::OmegaHGeometry>& geometry, int order,
  int num_components)
{
  return LagrangeFunctionSpace::FromGeometry(
    geometry, order, num_components, "global",
    LagrangeFunctionSpace::Backend::OmegaH);
}

/// Checks the orientation convention of the Jacobian of element `e`: row i is
/// x_{i+1} - x_0 for the element's vertices x_0, ..., x_Dim.
template <int Dim>
void CheckJacobianRows(Omega_h::Mesh& mesh,
                       const pcms::DifferentialView& differential, LO e)
{
  const auto jacobian = Kokkos::create_mirror_view_and_copy(
    HostMemorySpace(), differential.jacobian);
  const auto elem_verts = Omega_h::HostRead<Omega_h::LO>(mesh.ask_elem_verts());
  const auto coords = Omega_h::HostRead<Omega_h::Real>(mesh.coords());
  const LO v0 = elem_verts[(Dim + 1) * e];
  for (int i = 0; i < Dim; ++i) {
    const LO vi = elem_verts[(Dim + 1) * e + i + 1];
    for (int d = 0; d < Dim; ++d) {
      CAPTURE(e, i, d);
      REQUIRE_THAT(jacobian(e, i, d),
                   WithinAbs(coords[Dim * vi + d] - coords[Dim * v0 + d],
                             pcms::test::ExactTol));
    }
  }
}

KOKKOS_INLINE_FUNCTION Real Quadratic(Real x, Real y)
{
  return 1.0 + x - 2.0 * y + 3.0 * x * x - x * y + 0.5 * y * y;
}

} // namespace

TEST_CASE("One bind serves every space on a geometry")
{
  auto lib = Omega_h::Library{};
  auto mesh = BuildSquareMesh(lib, 8);
  auto geometry = MakeGeometry(mesh);
  auto scalars = MakeSpace(geometry, 1, 1);
  auto vectors = MakeSpace(geometry, 1, 3);
  auto cells = MakeSpace(geometry, 0, 1);
  REQUIRE(scalars->GetGeometry() == geometry);
  REQUIRE(vectors->GetGeometry() == geometry);
  REQUIRE(cells->GetGeometry() == geometry);

  auto f = scalars->CreateFunction<Real>("f");
  pcms::test::SetField(
    f, OMEGA_H_LAMBDA(Real x, Real y) { return pcms::test::linear_f(x, y); });
  const auto pts = pcms::test::StandardEvalCoords2D();
  auto coords = pcms::test::CreateDeviceCoordinateView(
    pts, geometry->GetCoordinateSystem());
  const auto bound = geometry->Bind(coords.coordinate_view);
  REQUIRE(bound->GetGeometry() == geometry);
  REQUIRE_NOTHROW(scalars->CreatePointEvaluator<Real>(bound));
  REQUIRE_NOTHROW(vectors->CreatePointEvaluator<Real>(bound));
  REQUIRE_NOTHROW(cells->CreatePointEvaluator<Real>(bound));
  pcms::test::CheckEvaluation(
    *scalars->CreatePointEvaluator<Real>(bound), f, pts,
    OMEGA_H_LAMBDA(Real x, Real y) { return pcms::test::linear_f(x, y); });
  // an evaluator still takes only fields of the space that created it
  REQUIRE_THROWS_WITH(
    pcms::test::EvaluateToHost(*cells->CreatePointEvaluator<Real>(bound), f,
                               f.GetData().GetValueBasis()),
    ContainsSubstring("not a field of the function space"));

  // the three components of the vector space see the same points
  auto g = vectors->CreateFunction<Real>("g");
  pcms::test::SetFieldComponents(g, [](Real x, Real y, Real* out) {
    for (int c = 0; c < 3; ++c) {
      out[c] = pcms::test::linear_f(x, y) + c;
    }
  });
  const auto values =
    pcms::test::EvaluateToHost(*vectors->CreatePointEvaluator<Real>(bound), g,
                               g.GetData().GetValueBasis(), 3);
  for (size_t i = 0; i < pts.size() / 2; ++i) {
    for (int c = 0; c < 3; ++c) {
      CAPTURE(i, c);
      REQUIRE_THAT(
        values(i, c),
        WithinAbs(pcms::test::linear_f(pts[2 * i], pts[2 * i + 1]) + c,
                  pcms::test::EvalTol));
    }
  }
}

TEST_CASE("Bound sources of another geometry are rejected")
{
  auto lib = Omega_h::Library{};
  auto mesh = BuildSquareMesh(lib, 4);
  auto geometry = MakeGeometry(mesh);
  auto other = MakeGeometry(mesh);
  auto space = MakeSpace(geometry, 1, 1);
  auto coords = pcms::test::CreateDeviceCoordinateView(
    pcms::test::StandardEvalCoords2D(), geometry->GetCoordinateSystem());
  const auto other_bound = other->Bind(coords.coordinate_view);
  REQUIRE_THROWS_WITH(space->CreatePointEvaluator<Real>(other_bound),
                      ContainsSubstring("another geometry"));
  // the same geometry, the same mesh: same entities
  REQUIRE(
    geometry->GetDiscretization()->SameEntities(*other->GetDiscretization()));
}

TEST_CASE("Parametric binds skip the search")
{
  auto lib = Omega_h::Library{};
  auto mesh = BuildSquareMesh(lib, 4);
  auto geometry = MakeGeometry(mesh);
  auto space = MakeSpace(geometry, 1, 1);
  auto f = space->CreateFunction<Real>("f");
  pcms::test::SetField(
    f, OMEGA_H_LAMBDA(Real x, Real y) { return pcms::test::linear_f(x, y); });

  // element centroids: barycentric coordinates 1/3 each
  const LO nelems = mesh.nelems();
  Kokkos::View<LO*, HostMemorySpace> ids_host("ids", nelems);
  Kokkos::View<Real**, HostMemorySpace> bary_host("bary", nelems, 3);
  for (LO e = 0; e < nelems; ++e) {
    ids_host(e) = e;
    for (int v = 0; v < 3; ++v) {
      bary_host(e, v) = 1.0 / 3.0;
    }
  }
  Kokkos::View<LO*, DeviceMemorySpace> ids("ids", nelems);
  Kokkos::deep_copy(ids, ids_host);
  // a device view of the device layout (the host mirror's layout need not
  // convert to the ParametricPoints view on a GPU)
  Kokkos::View<Real**, DeviceMemorySpace> bary("bary", nelems, 3);
  auto bary_mirror = Kokkos::create_mirror_view(bary);
  for (LO e = 0; e < nelems; ++e) {
    for (int v = 0; v < 3; ++v) {
      bary_mirror(e, v) = bary_host(e, v);
    }
  }
  Kokkos::deep_copy(bary, bary_mirror);
  const auto bound = geometry->Bind(pcms::ParametricPoints{ids, bary});
  REQUIRE(bound->NumPoints() == nelems);
  REQUIRE_FALSE(bound->HasPointValues());
  REQUIRE(bound->FoundPoints().extent(0) == static_cast<size_t>(nelems));
  REQUIRE(bound->MissingPoints().extent(0) == 0);

  const auto values = pcms::test::EvaluateToHost(
    *space->CreatePointEvaluator<Real>(bound), f, f.GetData().GetValueBasis());
  const auto elem_verts = Omega_h::HostRead<Omega_h::LO>(mesh.ask_elem_verts());
  const auto mesh_coords = Omega_h::HostRead<Omega_h::Real>(mesh.coords());
  for (LO e = 0; e < nelems; ++e) {
    Real x = 0, y = 0;
    for (int v = 0; v < 3; ++v) {
      const LO vert = elem_verts[3 * e + v];
      x += mesh_coords[2 * vert] / 3.0;
      y += mesh_coords[2 * vert + 1] / 3.0;
    }
    CAPTURE(e, x, y);
    REQUIRE_THAT(values(e, 0),
                 WithinAbs(pcms::test::linear_f(x, y), pcms::test::ExactTol));
  }

  // barycentric coordinates of a triangle have three columns
  Kokkos::View<Real**, DeviceMemorySpace> two_columns("two_columns", nelems, 2);
  REQUIRE_THROWS_WITH(geometry->Bind(pcms::ParametricPoints{ids, two_columns}),
                      ContainsSubstring("3 barycentric columns"));
  Kokkos::View<Real**, DeviceMemorySpace> short_coords("short", nelems - 1, 3);
  REQUIRE_THROWS_WITH(geometry->Bind(pcms::ParametricPoints{ids, short_coords}),
                      ContainsSubstring("differ in length"));

  // ids must be elements of the mesh
  Kokkos::View<LO*, DeviceMemorySpace> bad_ids("bad_ids", nelems);
  Kokkos::deep_copy(bad_ids, ids);
  Kokkos::deep_copy(Kokkos::subview(bad_ids, nelems - 1), nelems);
  REQUIRE_THROWS_WITH(
    geometry->Bind(pcms::ParametricPoints{bad_ids, bary}),
    ContainsSubstring("element ids span [0, " + std::to_string(nelems) +
                      "], outside [0, " + std::to_string(nelems) + ")"));
  Kokkos::deep_copy(Kokkos::subview(bad_ids, 0), -1);
  REQUIRE_THROWS_WITH(geometry->Bind(pcms::ParametricPoints{bad_ids, bary}),
                      ContainsSubstring("ids span [-1, "));

  // no points: an empty bound source that evaluates to nothing
  const auto empty = geometry->Bind(pcms::ParametricPoints{});
  REQUIRE(empty->NumPoints() == 0);
  REQUIRE(empty->FoundPoints().extent(0) == 0);
  REQUIRE(empty->MissingPoints().extent(0) == 0);
  const auto nothing = pcms::test::EvaluateToHost(
    *space->CreatePointEvaluator<Real>(empty), f, f.GetData().GetValueBasis());
  REQUIRE(nothing.extent(0) == 0);
#ifdef PCMS_ENABLE_MESHFIELDS
  auto p1 = LagrangeFunctionSpace::FromGeometry(
    geometry, 1, 1, "global", LagrangeFunctionSpace::Backend::MeshFields);
  auto h = p1->CreateFunction<Real>("h");
  REQUIRE(pcms::test::EvaluateToHost(*p1->CreatePointEvaluator<Real>(empty), h,
                                     h.GetData().GetValueBasis())
            .extent(0) == 0);
#endif
}

TEST_CASE("Parametric binds on a uniform grid check the cells")
{
  pcms::UniformGrid<2> grid;
  grid.bot_left = {0.0, 0.0};
  grid.edge_length = {1.0, 1.0};
  grid.divisions = {4, 3};
  auto geometry = pcms::UniformGridGeometry<2>::Create(
    grid, pcms::csys::Cartesian::Create(2));
  auto space = LagrangeFunctionSpace::FromGeometry(geometry, 1, 1);
  auto f = space->CreateFunction<Real>("f");

  Kokkos::View<LO*, DeviceMemorySpace> ids("ids", 2);
  Kokkos::View<Real**, DeviceMemorySpace> coords("coords", 2, 2);
  Kokkos::deep_copy(Kokkos::subview(ids, 1), 12);
  REQUIRE_THROWS_WITH(geometry->Bind(pcms::ParametricPoints{ids, coords}),
                      ContainsSubstring("cell ids span [0, 12], outside [0, "
                                        "12)"));
  Kokkos::deep_copy(Kokkos::subview(ids, 1), 11);
  REQUIRE(geometry->Bind(pcms::ParametricPoints{ids, coords})->NumPoints() ==
          2);

  const auto empty = geometry->Bind(pcms::ParametricPoints{});
  REQUIRE(empty->NumPoints() == 0);
  const auto nothing = pcms::test::EvaluateToHost(
    *space->CreatePointEvaluator<Real>(empty), f, f.GetData().GetValueBasis());
  REQUIRE(nothing.extent(0) == 0);
}

TEST_CASE("Parametric binds on a uniform grid evaluate at the given positions")
{
  pcms::UniformGrid<2> grid;
  grid.bot_left = {-1.0, 0.5};
  grid.edge_length = {2.0, 1.5};
  grid.divisions = {4, 3};
  auto geometry = pcms::UniformGridGeometry<2>::Create(
    grid, pcms::csys::Cartesian::Create(2));
  auto space = LagrangeFunctionSpace::FromGeometry(geometry, 1, 1);
  auto f = space->CreateFunction<Real>("f");
  // bilinear, so the multilinear interpolation reproduces it exactly
  pcms::test::SetField(
    f, OMEGA_H_LAMBDA(Real x, Real y) { return 1.0 + 2.0 * x - y + x * y; });

  // every cell at its centre and at an off-centre position (0.25, 0.75),
  // which tells the axes apart
  const LO ncells = grid.GetNumCells();
  const Real positions[2][2] = {{0.5, 0.5}, {0.25, 0.75}};
  const LO n = 2 * ncells;
  Kokkos::View<LO*, DeviceMemorySpace> ids("ids", n);
  Kokkos::View<Real**, DeviceMemorySpace> coords("coords", n, 2);
  auto ids_host = Kokkos::create_mirror_view(ids);
  auto coords_host = Kokkos::create_mirror_view(coords);
  std::vector<Real> physical;
  for (LO k = 0; k < n; ++k) {
    const LO cell = k % ncells;
    const auto* position = positions[k / ncells];
    ids_host(k) = cell;
    const auto bbox = grid.GetCellBBOX(cell);
    for (int d = 0; d < 2; ++d) {
      coords_host(k, d) = position[d];
      physical.push_back(bbox.center[d] - bbox.half_width[d] +
                         2.0 * bbox.half_width[d] * position[d]);
    }
  }
  Kokkos::deep_copy(ids, ids_host);
  Kokkos::deep_copy(coords, coords_host);

  const auto bound = geometry->Bind(pcms::ParametricPoints{ids, coords});
  REQUIRE_FALSE(bound->HasPointValues());
  const auto values = pcms::test::EvaluateToHost(
    *space->CreatePointEvaluator<Real>(bound), f, f.GetData().GetValueBasis());
  // the same points bound by their coordinates
  auto query = pcms::test::CreateDeviceCoordinateView(
    physical, geometry->GetCoordinateSystem());
  const auto located = pcms::test::EvaluateToHost(
    *space->CreatePointEvaluator<Real>(geometry->Bind(query.coordinate_view)),
    f, f.GetData().GetValueBasis());
  for (LO k = 0; k < n; ++k) {
    const Real x = physical[2 * k];
    const Real y = physical[2 * k + 1];
    CAPTURE(k, x, y);
    REQUIRE_THAT(values(k, 0),
                 WithinAbs(1.0 + 2.0 * x - y + x * y, pcms::test::ExactTol));
    REQUIRE_THAT(values(k, 0), WithinAbs(located(k, 0), pcms::test::ExactTol));
  }

  // positions in a 2D cell have two columns
  Kokkos::View<Real**, DeviceMemorySpace> three_columns("three", n, 3);
  REQUIRE_THROWS_WITH(
    geometry->Bind(pcms::ParametricPoints{ids, three_columns}),
    ContainsSubstring("2D grid need 2 columns, got 3"));
}

TEST_CASE("MapToPhysical gives the coordinates of parametric points")
{
  auto lib = Omega_h::Library{};
  auto mesh = BuildSquareMesh(lib, 3);
  auto geometry = MakeGeometry(mesh);
  // the centroid and the first vertex of every element
  const LO nelems = mesh.nelems();
  Kokkos::View<LO*, DeviceMemorySpace> ids("ids", 2 * nelems);
  Kokkos::View<Real**, DeviceMemorySpace> bary("bary", 2 * nelems, 3);
  auto ids_host = Kokkos::create_mirror_view(ids);
  auto bary_host = Kokkos::create_mirror_view(bary);
  for (LO e = 0; e < nelems; ++e) {
    ids_host(e) = e;
    ids_host(nelems + e) = e;
    for (int v = 0; v < 3; ++v) {
      bary_host(e, v) = 1.0 / 3.0;
      bary_host(nelems + e, v) = v == 0 ? 1.0 : 0.0;
    }
  }
  Kokkos::deep_copy(ids, ids_host);
  Kokkos::deep_copy(bary, bary_host);
  const auto physical = Kokkos::create_mirror_view_and_copy(
    HostMemorySpace(),
    geometry->MapToPhysical(pcms::ParametricPoints{ids, bary}));
  REQUIRE(physical.extent(0) == static_cast<size_t>(2 * nelems));
  REQUIRE(physical.extent(1) == 2);
  const auto elem_verts = Omega_h::HostRead<Omega_h::LO>(mesh.ask_elem_verts());
  const auto mesh_coords = Omega_h::HostRead<Omega_h::Real>(mesh.coords());
  for (LO e = 0; e < nelems; ++e) {
    for (int d = 0; d < 2; ++d) {
      Real centroid = 0;
      for (int v = 0; v < 3; ++v) {
        centroid += mesh_coords[2 * elem_verts[3 * e + v] + d] / 3.0;
      }
      CAPTURE(e, d);
      REQUIRE_THAT(physical(e, d), WithinAbs(centroid, pcms::test::ExactTol));
      REQUIRE_THAT(physical(nelems + e, d),
                   WithinAbs(mesh_coords[2 * elem_verts[3 * e] + d],
                             pcms::test::ExactTol));
    }
  }
  Kokkos::View<Real**, DeviceMemorySpace> two_columns("two", 2 * nelems, 2);
  REQUIRE_THROWS_WITH(
    geometry->MapToPhysical(pcms::ParametricPoints{ids, two_columns}),
    ContainsSubstring("MapToPhysical: parametric coordinates of a 2D mesh "
                      "need 3 barycentric columns"));
}

TEST_CASE("Differential gives the element measure")
{
  auto lib = Omega_h::Library{};
  SECTION("triangles")
  {
    auto mesh = BuildSquareMesh(lib, 5);
    auto geometry = MakeGeometry(mesh);
    auto coords = pcms::test::CreateDeviceCoordinateView(
      pcms::test::StandardEvalCoords2D(), geometry->GetCoordinateSystem());
    const auto differential =
      geometry->Differential(*geometry->Bind(coords.coordinate_view));
    REQUIRE(differential.per_element);
    REQUIRE(differential.jacobian.extent(0) ==
            static_cast<size_t>(mesh.nelems()));
    const auto det = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace(), differential.determinant);
    const auto measures =
      Omega_h::HostRead<Omega_h::Real>(Omega_h::measure_elements_real(&mesh));
    for (LO e = 0; e < mesh.nelems(); ++e) {
      CAPTURE(e);
      REQUIRE_THAT(det(e) / 2.0, WithinAbs(measures[e], pcms::test::ExactTol));
    }
    CheckJacobianRows<2>(mesh, differential, mesh.nelems() / 2);
    // the bound source must be one of this geometry
    auto other = MakeGeometry(mesh);
    REQUIRE_THROWS_WITH(
      geometry->Differential(*other->Bind(coords.coordinate_view)),
      ContainsSubstring("belongs to another geometry"));
  }
  SECTION("tetrahedra")
  {
    auto mesh = pcms::test::BuildUnitCube(lib, 2);
    auto geometry =
      pcms::OmegaHGeometry::Create(mesh, pcms::csys::Cartesian::Create(3));
    auto coords = pcms::test::CreateDeviceCoordinateView(
      {0.5, 0.5, 0.5}, geometry->GetCoordinateSystem(), 3);
    const auto differential =
      geometry->Differential(*geometry->Bind(coords.coordinate_view));
    const auto det = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace(), differential.determinant);
    const auto measures =
      Omega_h::HostRead<Omega_h::Real>(Omega_h::measure_elements_real(&mesh));
    for (LO e = 0; e < mesh.nelems(); ++e) {
      CAPTURE(e);
      REQUIRE_THAT(det(e) / 6.0, WithinAbs(measures[e], pcms::test::ExactTol));
    }
    CheckJacobianRows<3>(mesh, differential, mesh.nelems() - 1);
  }
}

TEST_CASE("Search options are honoured")
{
  auto lib = Omega_h::Library{};
  auto mesh = BuildSquareMesh(lib, 8);
  auto coarse = MakeGeometry(mesh, pcms::SearchOptions{3, 1e-12});
  auto derived = MakeGeometry(mesh);
  auto coarse_space = MakeSpace(coarse, 1, 1);
  auto derived_space = MakeSpace(derived, 1, 1);
  auto f = coarse_space->CreateFunction<Real>("f");
  pcms::test::SetField(
    f, OMEGA_H_LAMBDA(Real x, Real y) { return pcms::test::linear_f(x, y); });
  auto g = derived_space->CreateFunction<Real>("g");
  pcms::test::SetField(
    g, OMEGA_H_LAMBDA(Real x, Real y) { return pcms::test::linear_f(x, y); });

  // inside and outside points
  std::vector<Real> pts = pcms::test::StandardEvalCoords2D();
  for (Real x : pcms::test::StandardOutsideCoords2D()) {
    pts.push_back(x);
  }
  auto coords =
    pcms::test::CreateDeviceCoordinateView(pts, coarse->GetCoordinateSystem());
  const pcms::OutOfBoundsPolicy fill{pcms::OutOfBoundsMode::FILL, -1.0};
  const auto coarse_bound = coarse->Bind(coords.coordinate_view, fill);
  const auto derived_bound = derived->Bind(coords.coordinate_view, fill);
  const auto coarse_found = Kokkos::create_mirror_view_and_copy(
    HostMemorySpace(), coarse_bound->FoundPoints());
  const auto derived_found = Kokkos::create_mirror_view_and_copy(
    HostMemorySpace(), derived_bound->FoundPoints());
  REQUIRE(coarse_found.extent(0) == 5);
  REQUIRE(coarse_found.extent(0) == derived_found.extent(0));
  for (size_t k = 0; k < coarse_found.extent(0); ++k) {
    REQUIRE(coarse_found(k) == derived_found(k));
  }
  const auto a = pcms::test::EvaluateToHost(
    *coarse_space->CreatePointEvaluator<Real>(coarse_bound), f,
    f.GetData().GetValueBasis());
  const auto b = pcms::test::EvaluateToHost(
    *derived_space->CreatePointEvaluator<Real>(derived_bound), g,
    g.GetData().GetValueBasis());
  for (size_t i = 0; i < a.extent(0); ++i) {
    CAPTURE(i);
    REQUIRE_THAT(a(i, 0), WithinAbs(b(i, 0), pcms::test::ExactTol));
  }

  // the tolerance decides whether a point just outside the boundary is found
  auto near = pcms::test::CreateDeviceCoordinateView(
    {0.5, 0.5, 1.0 + 1e-8, 0.5}, coarse->GetCoordinateSystem());
  const auto loose = MakeGeometry(mesh, pcms::SearchOptions{.tolerance = 1e-6});
  const auto tight =
    MakeGeometry(mesh, pcms::SearchOptions{.tolerance = 1e-12});
  REQUIRE(loose->Bind(near.coordinate_view, fill)->FoundPoints().extent(0) ==
          2);
  const auto tight_bound = tight->Bind(near.coordinate_view, fill);
  REQUIRE(tight_bound->FoundPoints().extent(0) == 1);
  const auto tight_missing = Kokkos::create_mirror_view_and_copy(
    HostMemorySpace(), tight_bound->MissingPoints());
  REQUIRE(tight_missing.extent(0) == 1);
  REQUIRE(tight_missing(0) == 1);
}

TEST_CASE("Neighbourhood returns vertex supports")
{
  auto lib = Omega_h::Library{};
  auto mesh = BuildSquareMesh(lib, 8);
  auto geometry = MakeGeometry(mesh);
  auto coords = pcms::test::CreateDeviceCoordinateView(
    pcms::test::StandardEvalCoords2D(), geometry->GetCoordinateSystem());
  pcms::MLSOptions mls;
  mls.radius = 0.3;
  mls.min_req_supports = 6;
  // the search MLSOptions asks for: radius, 6 to 18 supports
  pcms::NeighbourhoodOptions options = mls.Localization();
  REQUIRE(options.max_supports == 18);
  const pcms::SupportResults actual =
    geometry->Neighbourhood(coords.coordinate_view, options);
  const pcms::SupportResults expected =
    pcms::AdjacencyLocalizationFactory(mesh, 0, mls)
      .Build(coords.coordinate_view);
  pcms::test::CheckSupportResultsEquivalent(actual, expected);
  // the options matter: the defaults give other supports
  const pcms::SupportResults defaults = geometry->Neighbourhood(
    coords.coordinate_view, pcms::NeighbourhoodOptions{});
  REQUIRE(defaults.supports_idx.size() != actual.supports_idx.size());

  // other upper bounds than 3 * min_supports are not honoured yet, so they
  // are refused
  options.max_supports = 12;
  REQUIRE_THROWS_WITH(
    geometry->Neighbourhood(coords.coordinate_view, options),
    ContainsSubstring(
      "max_supports must be 0 or 3 * min_supports (18), got 12"));
  // the query points must be in the geometry's coordinate system
  auto cylindrical = pcms::test::CreateDeviceCoordinateView(
    pcms::test::StandardEvalCoords2D(), pcms::csys::CylindricalRZ::Create());
  REQUIRE_THROWS_WITH(
    geometry->Neighbourhood(cylindrical.coordinate_view, mls.Localization()),
    ContainsSubstring("is not the geometry's coordinate system"));
}

#ifdef PCMS_ENABLE_MESHFIELDS
TEST_CASE("MeshFields evaluates P1 and P2 fields through the geometry's bind")
{
  // P1 reproduces a linear function and P2 a quadratic exactly, so any change
  // in the element coordinates MeshFields receives shows up here.
  auto lib = Omega_h::Library{};
  auto mesh = BuildSquareMesh(lib, 6);
  auto geometry = MakeGeometry(mesh);
  auto p1 = LagrangeFunctionSpace::FromGeometry(
    geometry, 1, 1, "global", LagrangeFunctionSpace::Backend::MeshFields);
  auto p2 = LagrangeFunctionSpace::FromGeometry(
    geometry, 2, 1, "global", LagrangeFunctionSpace::Backend::MeshFields);
  auto f1 = p1->CreateFunction<Real>("f1");
  pcms::test::SetField(
    f1, OMEGA_H_LAMBDA(Real x, Real y) { return pcms::test::linear_f(x, y); });
  auto f2 = p2->CreateFunction<Real>("f2");
  pcms::test::SetField(
    f2, OMEGA_H_LAMBDA(Real x, Real y) { return Quadratic(x, y); });

  const auto pts = pcms::test::StandardEvalCoords2D();
  auto coords = pcms::test::CreateDeviceCoordinateView(
    pts, geometry->GetCoordinateSystem());
  const auto bound = geometry->Bind(coords.coordinate_view);
  pcms::test::CheckEvaluation(
    *p1->CreatePointEvaluator<Real>(bound), f1, pts,
    OMEGA_H_LAMBDA(Real x, Real y) { return pcms::test::linear_f(x, y); },
    pcms::test::ExactTol);
  pcms::test::CheckEvaluation(
    *p2->CreatePointEvaluator<Real>(bound), f2, pts,
    OMEGA_H_LAMBDA(Real x, Real y) { return Quadratic(x, y); },
    pcms::test::ExactTol);
}
#endif
