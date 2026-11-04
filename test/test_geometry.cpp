#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <Kokkos_Core.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_shape.hpp>
#include <pcms/configuration.hpp>
#include <pcms/utility/coordinate_systems/cartesian.hpp>
#include <pcms/utility/coordinate_systems/cylindrical.hpp>
#include <pcms/backends/omega_h/geometry.hpp>
#include <pcms/backends/point_cloud/geometry.hpp>
#include <pcms/backends/uniform_grid/geometry.hpp>
#include <pcms/field/function_space/lagrange.hpp>
#include <pcms/field/function_space/polynomial_reconstruction.hpp>
#include <pcms/backends/omega_h/entity_layout.hpp>
#include <pcms/backends/omega_h/lagrange_layout.hpp>
#include <pcms/backends/point_cloud/layout.hpp>
#include <pcms/geometry/geometry.hpp>
#include <pcms/backends/omega_h/adj_search.hpp>
#include <pcms/localization/mls_support_helpers.hpp>
#include <pcms/utility/mesh_geometry.hpp>
#include <pcms/utility/omega_h_array_utils.hpp>
#include "field_test_utils.hpp"
#include <memory>
#include <string>
#include <variant>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using pcms::DeviceMemorySpace;
using pcms::FunctionSpace;
using pcms::HostMemorySpace;
using pcms::LagrangeBackend;
using pcms::LagrangeOptions;
using pcms::LO;
using pcms::Real;

namespace
{

std::shared_ptr<pcms::Geometry> MakeGeometry(Omega_h::Mesh& mesh,
                                             pcms::SearchOptions options = {})
{
  return pcms::backend::omega_h::CreateGeometry(
    mesh, pcms::csys::Cartesian::Create(2),
    pcms::backend::omega_h::OmegaHGeometryOptions{.search = options});
}

/// The element the grid search of `mesh` finds for each point, lifted from the
/// entity it reports: where the adjacency search of the point starts.
Kokkos::View<LO*> StartElements(Omega_h::Mesh& mesh,
                                Kokkos::View<const Real**> points)
{
  return std::visit(
    [&](const auto& s) {
      constexpr int Dim = std::decay_t<decltype(s)>::DIM;
      return pcms::detail::LiftResults<Dim>(
               mesh,
               Kokkos::View<const typename std::decay_t<decltype(s)>::Result*,
                            pcms::DeviceMemorySpace>(
                 pcms::LocatePoints(s, points).second))
        .first;
    },
    pcms::MakeGridSearch(mesh, {}));
}

std::shared_ptr<FunctionSpace> MakeSpace(
  const std::shared_ptr<pcms::Geometry>& geometry, int order,
  int num_components)
{
  return CreateLagrangeSpace(
    geometry, order, num_components,
    LagrangeOptions{.backend = LagrangeBackend::OmegaH});
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
  auto mesh = pcms::test::BuildSquareMesh(lib, 8);
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
  const auto bound = pcms::test::Bind(*geometry, pts);
  REQUIRE(bound->GetGeometry() == geometry);
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

TEST_CASE("Spaces on a geometry share its discretization")
{
  auto lib = Omega_h::Library{};
  auto mesh = pcms::test::BuildSquareMesh(lib, 4);
  auto geometry = MakeGeometry(mesh);
  const auto discretization = geometry->GetDiscretization();
  auto scalars = MakeSpace(geometry, 1, 1);
  auto cells = MakeSpace(geometry, 0, 1);
  REQUIRE(scalars->GetLayout()->GetDiscretization() == discretization);
  REQUIRE(cells->GetLayout()->GetDiscretization() == discretization);
  REQUIRE(scalars->GetDiscretization() == discretization);
#ifdef PCMS_ENABLE_MESHFIELDS
  auto p2 = CreateLagrangeSpace(
    geometry, 2, 1, LagrangeOptions{.backend = LagrangeBackend::MeshFields});
  REQUIRE(p2->GetLayout()->GetDiscretization() == discretization);
#endif
  // a space on another geometry of the same mesh has the same entities, but
  // not the same discretization object
  auto other = MakeSpace(MakeGeometry(mesh), 1, 1);
  REQUIRE(other->GetLayout()->GetDiscretization() != discretization);
  REQUIRE(
    other->GetLayout()->GetDiscretization()->SameEntities(*discretization));
  // a layout refuses a discretization of another mesh
  auto other_mesh = pcms::test::BuildSquareMesh(lib, 2);
  REQUIRE_THROWS_WITH(
    pcms::OmegaHLagrangeLayout(
      other_mesh, 1, 1, pcms::csys::Cartesian::Create(2),
      std::static_pointer_cast<const pcms::OmegaHDiscretization>(
        discretization)),
    ContainsSubstring("OmegaHLagrangeLayout: the discretization is of another "
                      "mesh"));
}

TEST_CASE("Known owners skip the search")
{
  auto lib = Omega_h::Library{};
  auto mesh = pcms::test::BuildSquareMesh(lib, 4);
  auto geometry = MakeGeometry(mesh);
  auto space = MakeSpace(geometry, 1, 1);
  auto f = space->CreateFunction<Real>("f");
  pcms::test::SetField(
    f, OMEGA_H_LAMBDA(Real x, Real y) { return pcms::test::linear_f(x, y); });

  // the element centroids, with their elements as owners
  const LO nelems = mesh.nelems();
  const auto centroids =
    pcms::ConvertCoordsTo2D(pcms::get_entity_centroids(mesh, 2), nelems, 2);
  const auto coords =
    pcms::test::MakeCoords(centroids, geometry->GetCoordinateSystem());
  Kokkos::View<LO*, DeviceMemorySpace> elements("elements", nelems);
  pcms::iota_view<LO>(elements);
  const auto searched = geometry->Bind(coords);
  const auto hinted =
    geometry->Bind(coords, pcms::KnownOwners{geometry, 2, elements});
  REQUIRE(hinted->FoundPoints().extent(0) == static_cast<size_t>(nelems));
  const auto element_ids = [](const auto& bound) {
    return Kokkos::create_mirror_view_and_copy(
      HostMemorySpace(),
      dynamic_cast<const pcms::OmegaHMeshBinding&>(*bound).ElementIds());
  };
  // inside the mesh, the hint gives what the search finds
  const auto searched_ids = element_ids(searched);
  const auto hinted_ids = element_ids(hinted);
  const auto searched_values =
    pcms::test::EvaluateToHost(*space->CreatePointEvaluator<Real>(searched), f,
                               f.GetData().GetValueBasis());
  const auto hinted_values = pcms::test::EvaluateToHost(
    *space->CreatePointEvaluator<Real>(hinted), f, f.GetData().GetValueBasis());
  for (LO e = 0; e < nelems; ++e) {
    CAPTURE(e);
    REQUIRE(searched_ids(e) == e);
    REQUIRE(hinted_ids(e) == e);
    REQUIRE_THAT(hinted_values(e, 0),
                 WithinAbs(searched_values(e, 0), pcms::test::ExactTol));
  }

  // ids must be entities of the geometry
  Kokkos::View<LO*, DeviceMemorySpace> beyond("beyond", nelems);
  Kokkos::deep_copy(beyond, elements);
  Kokkos::deep_copy(Kokkos::subview(beyond, nelems - 1), nelems);
  REQUIRE_THROWS_WITH(
    geometry->Bind(coords, pcms::KnownOwners{geometry, 2, beyond}),
    ContainsSubstring("element ids span [0, " + std::to_string(nelems) +
                      "], outside [0, " + std::to_string(nelems) + ")"));

  // a hint for another geometry is ignored: the points are searched
  Kokkos::View<LO*, DeviceMemorySpace> wrong("wrong", nelems);
  const auto ignored =
    geometry->Bind(coords, pcms::KnownOwners{MakeGeometry(mesh), 2, wrong});
  const auto ignored_ids = element_ids(ignored);
  for (LO e = 0; e < nelems; ++e) {
    REQUIRE(ignored_ids(e) == e);
  }
}

TEST_CASE("Differential gives the element measure")
{
  auto lib = Omega_h::Library{};
  SECTION("triangles")
  {
    auto mesh = pcms::test::BuildSquareMesh(lib, 5);
    auto geometry = MakeGeometry(mesh);
    const auto pts = pcms::test::StandardEvalCoords2D();
    const auto differential =
      geometry->Differential(*pcms::test::Bind(*geometry, pts));
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
    REQUIRE_THROWS_WITH(geometry->Differential(*pcms::test::Bind(*other, pts)),
                        ContainsSubstring("belongs to another geometry"));
  }
  SECTION("tetrahedra")
  {
    auto mesh = pcms::test::BuildUnitCube(lib, 2);
    auto geometry = pcms::backend::omega_h::CreateGeometry(
      mesh, pcms::csys::Cartesian::Create(3));
    const auto differential =
      geometry->Differential(*pcms::test::Bind(*geometry, {0.5, 0.5, 0.5}));
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
  auto mesh = pcms::test::BuildSquareMesh(lib, 8);
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
  const pcms::OutOfBoundsPolicy fill{pcms::OutOfBoundsMode::FILL, -1.0};
  const auto coarse_bound = pcms::test::Bind(*coarse, pts, fill);
  const auto derived_bound = pcms::test::Bind(*derived, pts, fill);
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
  const std::vector<Real> near = {0.5, 0.5, 1.0 + 1e-8, 0.5};
  const auto loose = MakeGeometry(mesh, pcms::SearchOptions{.tolerance = 1e-6});
  const auto tight =
    MakeGeometry(mesh, pcms::SearchOptions{.tolerance = 1e-12});
  REQUIRE(pcms::test::Bind(*loose, near, fill)->FoundPoints().extent(0) == 2);
  const auto tight_bound = pcms::test::Bind(*tight, near, fill);
  REQUIRE(tight_bound->FoundPoints().extent(0) == 1);
  const auto tight_missing = Kokkos::create_mirror_view_and_copy(
    HostMemorySpace(), tight_bound->MissingPoints());
  REQUIRE(tight_missing.extent(0) == 1);
  REQUIRE(tight_missing(0) == 1);
}

TEST_CASE("Neighbourhood returns vertex supports")
{
  auto lib = Omega_h::Library{};
  auto mesh = pcms::test::BuildSquareMesh(lib, 8);
  auto geometry = MakeGeometry(mesh);
  auto coords = pcms::test::CreateDeviceCoordinateView(
    pcms::test::StandardEvalCoords2D(), geometry->GetCoordinateSystem());
  pcms::NeighbourhoodOptions options{.radius = 0.3, .min_supports = 6};
  const pcms::SupportResults actual =
    geometry->Neighbourhood(coords.coordinate_view, options);
  Omega_h::Real radius_sq = 0.09;
  const pcms::SupportResults expected = pcms::searchNeighbors(
    mesh, 0, pcms::flatten_to_omega_h_reals(coords.coordinate_view.GetValues()),
    2, StartElements(mesh, coords.view), radius_sq, 6, 18);
  pcms::test::CheckSupportResultsEquivalent(actual, expected);
  // a point outside the mesh starts from the element nearest it
  auto outside = pcms::test::CreateDeviceCoordinateView(
    {1.05, 0.5}, geometry->GetCoordinateSystem());
  REQUIRE(geometry->Neighbourhood(outside.coordinate_view, options)
            .supports_idx.size() >= 6);
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
    geometry->Neighbourhood(cylindrical.coordinate_view, {}),
    ContainsSubstring("is not the geometry's coordinate system"));
}

TEST_CASE("PointCloudGeometry::Neighbourhood takes the search of its sites")
{
  auto lib = Omega_h::Library{};
  auto mesh = pcms::test::BuildSquareMesh(lib, 8);
  const auto system = pcms::csys::Cartesian::Create(2);
  auto vertices =
    std::make_shared<pcms::OmegaHEntityLayout>(mesh, pcms::Vertex, 1, system);
  auto faces =
    std::make_shared<pcms::OmegaHEntityLayout>(mesh, pcms::Face, 1, system);
  const auto centroids = pcms::get_entity_centroids(mesh, pcms::Face);
  auto cloud = std::make_shared<pcms::PointCloudLayout>(
    2, pcms::ConvertCoordsTo2D(centroids, mesh.nfaces(), 2), system);
  const auto query = vertices->GetDOFHolderCoordinates();
  pcms::NeighbourhoodOptions options{.radius = 0.25, .min_supports = 8};
  Omega_h::Real radius_sq = options.radius * options.radius;
  const auto distance_search = pcms::BuildPointCloudSupports(
    centroids, mesh.coords(), 2, options.radius, 8, true);

  // a point cloud, and element centroids queried at arbitrary points: the
  // distance search
  pcms::test::CheckSupportResultsEquivalent(
    pcms::CreatePointCloudGeometry(cloud)->Neighbourhood(query, options),
    distance_search);
  const auto centroid_sites = pcms::CreatePointCloudGeometry(faces);
  pcms::test::CheckSupportResultsEquivalent(
    centroid_sites->Neighbourhood(query, options), distance_search);
  // element centroids queried at the vertices of their mesh, given as the
  // owners: the adjacency search from the elements around each vertex, which
  // on this mesh finds what the distance search finds
  const auto vertex_owners = pcms::DOFHolderOwners(*vertices, centroid_sites);
  REQUIRE(vertex_owners.entity_dim == pcms::Vertex);
  pcms::test::CheckSupportResultsEquivalent(
    centroid_sites->Neighbourhood(query, options, vertex_owners),
    distance_search);
  // the search starts from the owners: from a vertex far from the point, a
  // fixed radius finds nothing
  auto origin = pcms::test::CreateDeviceCoordinateView({0.0, 0.0}, system);
  Kokkos::View<LO*, DeviceMemorySpace> far("far", 1);
  Kokkos::deep_copy(far, mesh.nverts() - 1);
  const pcms::NeighbourhoodOptions fixed{
    .radius = 0.25, .min_supports = 8, .adapt_radius = false};
  REQUIRE(centroid_sites->Neighbourhood(origin.coordinate_view, fixed)
            .supports_idx.size() > 0);
  REQUIRE(centroid_sites
            ->Neighbourhood(origin.coordinate_view, fixed,
                            pcms::KnownOwners{centroid_sites, 0, far})
            .supports_idx.size() == 0);
  // vertices: the adjacency search from the element of each query point
  pcms::test::CheckSupportResultsEquivalent(
    pcms::CreatePointCloudGeometry(vertices)->Neighbourhood(query, options),
    pcms::searchNeighbors(
      mesh, 0, mesh.coords(), 2,
      StartElements(mesh,
                    pcms::ConvertCoordsTo2D(mesh.coords(), mesh.nverts(), 2)),
      radius_sq, 8, 24));
}

TEST_CASE("Points the adjacency search cannot start from are missing")
{
  auto lib = Omega_h::Library{};
  // two triangles far apart: the search cells between them hold no element
  Omega_h::Mesh mesh(&lib);
  Omega_h::build_from_elems_and_coords(
    &mesh, OMEGA_H_SIMPLEX, 2, Omega_h::LOs({0, 1, 2, 3, 4, 5}),
    Omega_h::Reals({0, 0, 1, 0, 0, 1, 9, 9, 10, 9, 9, 10}));
  pcms::test::AddDefaultClassification(mesh);
  const auto system = pcms::csys::Cartesian::Create(2);
  auto vertices =
    std::make_shared<pcms::OmegaHEntityLayout>(mesh, pcms::Vertex, 1, system);
  auto space = pcms::CreatePolynomialReconstructionSpace(
    pcms::CreatePointCloudGeometry(
      vertices, pcms::NeighbourhoodOptions{.radius = 0.5, .min_supports = 3}),
    pcms::MLSOptions{.degree = 1});
  auto f = space->CreateFunction<Real>();
  pcms::test::SetField(
    f, OMEGA_H_LAMBDA(Real x, Real y) { return pcms::test::linear_f(x, y); });
  const std::vector<Real> pts{0.25, 0.25, 5.5, 5.5};
  auto coords = pcms::test::CreateDeviceCoordinateView(pts, system);
  const auto bound = space->GetGeometry()->Bind(
    coords.coordinate_view,
    pcms::OutOfBoundsPolicy{pcms::OutOfBoundsMode::FILL, -1.0});
  REQUIRE(bound->MissingPoints().extent(0) == 1);
  const auto values = pcms::test::EvaluateToHost(
    *space->CreatePointEvaluator<Real>(bound), f, f.GetData().GetValueBasis());
  REQUIRE_THAT(values(0, 0), WithinAbs(pcms::test::linear_f(0.25, 0.25),
                                       pcms::test::EvalTol));
  REQUIRE(values(1, 0) == -1.0);
}

TEST_CASE("An empty bind evaluates to nothing")
{
  auto lib = Omega_h::Library{};
  auto mesh = pcms::test::BuildSquareMesh(lib, 2);
  auto geometry = MakeGeometry(mesh);
  pcms::UniformGrid<2> grid;
  grid.bot_left = {0.0, 0.0};
  grid.edge_length = {1.0, 1.0};
  grid.divisions = {2, 2};
  std::vector<std::shared_ptr<FunctionSpace>> spaces{
    MakeSpace(geometry, 1, 1),
    CreateLagrangeSpace(
      pcms::CreateUniformGridGeometry(grid, pcms::csys::Cartesian::Create(2)),
      1, 1)};
#ifdef PCMS_ENABLE_MESHFIELDS
  spaces.push_back(CreateLagrangeSpace(
    geometry, 1, 1, LagrangeOptions{.backend = LagrangeBackend::MeshFields}));
#endif
  const Kokkos::View<Real**, DeviceMemorySpace> none("none", 0, 2);
  for (const auto& space : spaces) {
    auto f = space->CreateFunction<Real>("f");
    const auto bound = space->GetGeometry()->Bind(
      pcms::test::MakeCoords(none, space->GetCoordinateSystem()));
    REQUIRE(
      pcms::test::EvaluateToHost(*space->CreatePointEvaluator<Real>(bound), f,
                                 f.GetData().GetValueBasis())
        .extent(0) == 0);
  }
}

TEST_CASE("DOFHolderOwners gives the entity of each DOF holder")
{
  auto lib = Omega_h::Library{};
  auto mesh = pcms::test::BuildSquareMesh(lib, 3);
  auto geometry = MakeGeometry(mesh);
  const auto entities = [&](const pcms::FunctionSpace& space, int dim) {
    const auto owners = pcms::DOFHolderOwners(*space.GetLayout(), geometry);
    REQUIRE(owners.geometry == geometry);
    REQUIRE(owners.entity_dim == dim);
    const auto ids =
      Kokkos::create_mirror_view_and_copy(HostMemorySpace(), owners.entities);
    REQUIRE(ids.extent(0) == static_cast<size_t>(mesh.nents(dim)));
    for (LO i = 0; i < mesh.nents(dim); ++i) {
      REQUIRE(ids(i) == i);
    }
  };
  entities(*MakeSpace(geometry, 1, 1), pcms::Vertex);
  entities(*MakeSpace(geometry, 0, 1), pcms::Face);
#ifdef PCMS_ENABLE_MESHFIELDS
  entities(
    *CreateLagrangeSpace(
      geometry, 1, 1, LagrangeOptions{.backend = LagrangeBackend::MeshFields}),
    pcms::Vertex);
#endif
  // with ghosts, DOF holders are not the entities
  Omega_h::Write<Omega_h::I8> mask(mesh.nverts(), 1);
  Kokkos::deep_copy(Kokkos::View<Omega_h::I8*>(mask.data(), 1), 0);
  const auto ghosted = CreateLagrangeSpace(
    pcms::backend::omega_h::CreateGeometry(
      mesh, pcms::csys::Cartesian::Create(2),
      pcms::backend::omega_h::OmegaHGeometryOptions{.owned_mask =
                                                      Omega_h::Read(mask)}),
    1, 1, LagrangeOptions{.backend = LagrangeBackend::OmegaH});
  REQUIRE(ghosted->GetLayout()->HasGhosts());
  REQUIRE(
    pcms::DOFHolderOwners(*ghosted->GetLayout(), geometry).entities.extent(0) ==
    0);
}

TEST_CASE("Hinted neighbourhoods equal searched ones inside the mesh")
{
  auto lib = Omega_h::Library{};
  auto mesh = pcms::test::BuildSquareMesh(lib, 6);
  auto geometry = MakeGeometry(mesh);
  const pcms::NeighbourhoodOptions options{.radius = 0.3, .min_supports = 6};
  for (const int dim : {pcms::Vertex, pcms::Face}) {
    CAPTURE(dim);
    const auto points = pcms::ConvertCoordsTo2D(
      pcms::get_entity_centroids(mesh, dim), mesh.nents(dim), 2);
    const auto coords =
      pcms::test::MakeCoords(points, geometry->GetCoordinateSystem());
    Kokkos::View<LO*, DeviceMemorySpace> ids("ids", mesh.nents(dim));
    pcms::iota_view<LO>(ids);
    pcms::test::CheckSupportResultsEquivalent(
      geometry->Neighbourhood(coords, options,
                              pcms::KnownOwners{geometry, dim, ids}),
      geometry->Neighbourhood(coords, options));
  }
}

TEST_CASE("Binding with owners keeps the policy")
{
  auto lib = Omega_h::Library{};
  auto mesh = pcms::test::BuildSquareMesh(lib, 4);
  auto geometry = MakeGeometry(mesh);
  auto space = MakeSpace(geometry, 1, 1);
  auto f = space->CreateFunction<Real>("f");
  // the second point is outside; without a usable hint it is searched
  const std::vector<Real> pts{0.5, 0.5, 2.0, 2.0};
  auto coords = pcms::test::CreateDeviceCoordinateView(
    pts, geometry->GetCoordinateSystem());
  const auto bound =
    geometry->Bind(coords.coordinate_view, pcms::KnownOwners{},
                   pcms::OutOfBoundsPolicy{pcms::OutOfBoundsMode::FILL, -1.0});
  REQUIRE(bound->Policy().fill_value == -1.0);
  REQUIRE(bound->MissingPoints().extent(0) == 1);
  const auto values = pcms::test::EvaluateToHost(
    *space->CreatePointEvaluator<Real>(bound), f, f.GetData().GetValueBasis());
  REQUIRE(values(1, 0) == -1.0);
}

#ifdef PCMS_ENABLE_MESHFIELDS
TEST_CASE("MeshFields evaluates P2 fields through the geometry's bind")
{
  // P2 reproduces a quadratic exactly, so any change in the element
  // coordinates MeshFields receives shows up here.
  auto lib = Omega_h::Library{};
  auto mesh = pcms::test::BuildSquareMesh(lib, 6);
  auto geometry = MakeGeometry(mesh);
  auto p2 = CreateLagrangeSpace(
    geometry, 2, 1, LagrangeOptions{.backend = LagrangeBackend::MeshFields});
  auto f2 = p2->CreateFunction<Real>("f2");
  pcms::test::SetField(
    f2, OMEGA_H_LAMBDA(Real x, Real y) { return Quadratic(x, y); });

  const auto pts = pcms::test::StandardEvalCoords2D();
  const auto bound = pcms::test::Bind(*geometry, pts);
  pcms::test::CheckEvaluation(
    *p2->CreatePointEvaluator<Real>(bound), f2, pts,
    OMEGA_H_LAMBDA(Real x, Real y) { return Quadratic(x, y); },
    pcms::test::ExactTol);
}
#endif
