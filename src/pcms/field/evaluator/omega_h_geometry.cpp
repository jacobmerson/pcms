#include "pcms/field/evaluator/omega_h_geometry.hpp"
#include "pcms/field/coordinate_system.hpp"
#include "pcms/field/evaluator/mls_options.h"
#include "pcms/localization/adj_search.hpp"
#include "pcms/localization/mesh_localization.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/profile.h"
#include "pcms/utility/uniform_grid.h"
#include <stdexcept>
#include <string>
#include <tuple>
#include <variant>

namespace pcms
{

OmegaHMeshBinding::OmegaHMeshBinding(std::shared_ptr<const Geometry> geometry,
                                     const BindRequest& request,
                                     const GridPointSearchVariant& search,
                                     Omega_h::Mesh& mesh)
  : BoundSource(std::move(geometry), request)
{
  std::visit(
    [&](const auto& s) {
      constexpr int Dim = std::decay_t<decltype(s)>::DIM;
      auto [found, owning_ids] = FindOwningElements(s, LocatedPointValues());
      SetFound(found);
      std::tie(elem_ids_, bary_) = detail::BarycentricInElements<Dim>(
        mesh, LocatedPointValues(), FoundPoints(), owning_ids);
    },
    search);
}

namespace
{

/// Throws unless there are no points or every point has dim + 1 barycentric
/// coordinates and an element of `mesh`.
void CheckBarycentricPoints(const ParametricPoints& points, Omega_h::Mesh& mesh,
                            const std::string& caller)
{
  if (points.NumPoints() == 0) {
    return;
  }
  const int dim = mesh.dim();
  if (static_cast<int>(points.coords.extent(1)) != dim + 1) {
    throw pcms_error(caller + ": parametric coordinates of a " +
                     std::to_string(dim) + "D mesh need " +
                     std::to_string(dim + 1) + " barycentric columns, got " +
                     std::to_string(points.coords.extent(1)));
  }
  detail::CheckParametricIds(points, mesh.nelems(), caller, "element");
}

template <int Dim>
Kokkos::View<Real**, DeviceMemorySpace> PhysicalFromBarycentric(
  Omega_h::Mesh& mesh, const ParametricPoints& points)
{
  const LO n = points.NumPoints();
  Kokkos::View<Real**, DeviceMemorySpace> physical(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "omega_h_physical_points"),
    n, Dim);
  const auto elem_verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto ids = points.ids;
  const auto bary = points.coords;
  Kokkos::parallel_for(
    "OmegaHGeometry::MapToPhysical",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, n),
    KOKKOS_LAMBDA(const LO i) {
      const LO elem = ids(i);
      Real x[Dim] = {};
      for (int v = 0; v < Dim + 1; ++v) {
        const LO vert = elem_verts[elem * (Dim + 1) + v];
        for (int d = 0; d < Dim; ++d) {
          x[d] += bary(i, v) * coords[vert * Dim + d];
        }
      }
      for (int d = 0; d < Dim; ++d) {
        physical(i, d) = x[d];
      }
    });
  return physical;
}

} // namespace

OmegaHMeshBinding::OmegaHMeshBinding(std::shared_ptr<const Geometry> geometry,
                                     const ParametricPoints& points,
                                     OutOfBoundsPolicy policy,
                                     Omega_h::Mesh& mesh)
  : BoundSource(std::move(geometry), points, policy)
{
  const int dim = mesh.dim();
  const LO n = points.NumPoints();
  elem_ids_ = Kokkos::View<LO*, DeviceMemorySpace>(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "omega_h_elem_ids"), n);
  bary_ = Kokkos::View<Real**, DeviceMemorySpace>(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "omega_h_bary"), n,
    dim + 1);
  if (n == 0) {
    return;
  }
  CheckBarycentricPoints(points, mesh, "OmegaHMeshBinding");
  Kokkos::deep_copy(elem_ids_, points.ids);
  Kokkos::deep_copy(bary_, points.coords);
}

GridPointSearchVariant MakeGridSearch(Omega_h::Mesh& mesh,
                                      const SearchOptions& options)
{
  if (mesh.dim() == 2) {
    const LO n = options.divisions > 0
                   ? options.divisions
                   : DivisionsPerAxisForMesh<2>(mesh.nelems());
    GridPointSearch2D::PointSearchTolerances tolerances(
      "point search 2d tolerances");
    Kokkos::deep_copy(tolerances, options.tolerance);
    return GridPointSearch2D(mesh, n, n, tolerances);
  }
  if (mesh.dim() == 3) {
    const LO n = options.divisions > 0
                   ? options.divisions
                   : DivisionsPerAxisForMesh<3>(mesh.nelems());
    GridPointSearch3D::PointSearchTolerances tolerances(
      "point search 3d tolerances");
    Kokkos::deep_copy(tolerances, options.tolerance);
    return GridPointSearch3D(mesh, n, n, n, tolerances);
  }
  throw std::invalid_argument(
    "MakeGridSearch: only 2D and 3D meshes are supported");
}

OmegaHGeometry::OmegaHGeometry(Omega_h::Mesh& mesh,
                               std::shared_ptr<const CoordinateSystem> system,
                               SearchOptions options)
  : mesh_(mesh),
    discretization_(std::make_shared<OmegaHDiscretization>(mesh)),
    system_(ResolveCoordinateSystem(std::move(system), mesh.dim())),
    search_(MakeGridSearch(mesh, options))
{
}

std::shared_ptr<OmegaHGeometry> OmegaHGeometry::Create(
  Omega_h::Mesh& mesh, std::shared_ptr<const CoordinateSystem> system,
  SearchOptions options)
{
  return std::make_shared<OmegaHGeometry>(mesh, std::move(system), options);
}

Distribution OmegaHGeometry::GetDistribution() const noexcept
{
  return mesh_.comm()->size() > 1 ? Distribution::Partitioned
                                  : Distribution::Replicated;
}

std::shared_ptr<BoundSource> OmegaHGeometry::BindImpl(
  const BindRequest& request) const
{
  PCMS_FUNCTION_TIMER;
  return std::make_shared<OmegaHMeshBinding>(shared_from_this(), request,
                                             search_, mesh_);
}

std::shared_ptr<BoundSource> OmegaHGeometry::BindParametricImpl(
  const ParametricPoints& points, OutOfBoundsPolicy policy) const
{
  return std::make_shared<OmegaHMeshBinding>(shared_from_this(), points, policy,
                                             mesh_);
}

Kokkos::View<Real**, DeviceMemorySpace> OmegaHGeometry::MapToPhysical(
  const ParametricPoints& points) const
{
  if (points.coords.extent(0) != points.ids.extent(0)) {
    throw pcms_error(
      "OmegaHGeometry::MapToPhysical: parametric ids and coords differ in "
      "length");
  }
  CheckBarycentricPoints(points, mesh_, "OmegaHGeometry::MapToPhysical");
  return mesh_.dim() == 2 ? PhysicalFromBarycentric<2>(mesh_, points)
                          : PhysicalFromBarycentric<3>(mesh_, points);
}

SupportResults OmegaHGeometry::Neighbourhood(
  CoordinateView<DeviceMemorySpace> coords,
  const NeighbourhoodOptions& options) const
{
  if (options.max_supports != NeighbourhoodOptions{}.max_supports) {
    throw pcms_error("OmegaHGeometry::Neighbourhood: max_supports is not "
                     "honoured yet; leave it at its default");
  }
  MLSOptions mls;
  mls.radius = options.radius;
  mls.min_req_supports = static_cast<unsigned>(options.min_supports);
  mls.adapt_radius = options.adapt_radius;
  return AdjacencyLocalizationFactory(mesh_, /*source_entity_dim=*/0, mls)
    .Build(std::move(coords));
}

namespace
{

/// Per-element Jacobian rows J_i = x_{i+1} - x_0 of the affine simplex map
/// and its determinant.
template <int Dim>
DifferentialView SimplexDifferential(Omega_h::Mesh& mesh)
{
  const LO nelems = mesh.nelems();
  Kokkos::View<Real***, DeviceMemorySpace> jacobian("omega_h_jacobian", nelems,
                                                    Dim, Dim);
  Kokkos::View<Real*, DeviceMemorySpace> determinant("omega_h_jacobian_det",
                                                     nelems);
  const auto elem_verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  Kokkos::parallel_for(
    "OmegaHGeometry::Differential",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, nelems),
    KOKKOS_LAMBDA(const LO e) {
      const LO v0 = elem_verts[e * (Dim + 1)];
      Real j[Dim][Dim];
      for (int i = 0; i < Dim; ++i) {
        const LO vi = elem_verts[e * (Dim + 1) + i + 1];
        for (int d = 0; d < Dim; ++d) {
          j[i][d] = coords[vi * Dim + d] - coords[v0 * Dim + d];
          jacobian(e, i, d) = j[i][d];
        }
      }
      if constexpr (Dim == 2) {
        determinant(e) = j[0][0] * j[1][1] - j[0][1] * j[1][0];
      } else {
        determinant(e) = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1]) -
                         j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0]) +
                         j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
      }
    });
  return DifferentialView{jacobian, determinant, /*per_element=*/true};
}

} // namespace

DifferentialView OmegaHGeometry::Differential(const BoundSource& bound) const
{
  if (bound.GetGeometry().get() != this) {
    throw pcms_error(
      "Differential: the bound source belongs to another geometry");
  }
  std::call_once(differential_once_, [this] {
    differential_ = mesh_.dim() == 2 ? SimplexDifferential<2>(mesh_)
                                     : SimplexDifferential<3>(mesh_);
  });
  return differential_;
}

} // namespace pcms
