#include "pcms/backends/omega_h/geometry.hpp"
#include "pcms/utility/coordinate_system.hpp"
#include "pcms/backends/omega_h/adj_search.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/omega_h_array_utils.hpp"
#include "pcms/utility/detail/profile.hpp"
#include "pcms/utility/uniform_grid.hpp"
#include <stdexcept>
#include <string>
#include <variant>

namespace pcms
{
using utility::ConvertCoordsTo2D;
using utility::flatten_to_omega_h_reals;
} // namespace pcms

namespace pcms::backend::omega_h
{

OmegaHMeshBinding::OmegaHMeshBinding(std::shared_ptr<const Geometry> geometry,
                                     const BindRequest& request,
                                     const GridPointSearchVariant& search,
                                     Omega_h::Mesh& mesh)
  : BoundSource(std::move(geometry), request), search_(search), mesh_(mesh)
{
  const bool hinted = request.owners.entity_dim == mesh.dim() &&
                      request.owners.entities.extent(0) > 0;
  if (hinted) {
    pcms::detail::CheckEntityIds(request.owners.entities, mesh.nelems(),
                                 "OmegaHMeshBinding", "element");
  }
  Locate(hinted ? request.owners.entities
                : Kokkos::View<const LO*, DeviceMemorySpace>());
}

void OmegaHMeshBinding::RebindImpl()
{
  Locate({});
}

void OmegaHMeshBinding::Locate(
  Kokkos::View<const LO*, DeviceMemorySpace> owners)
{
  std::visit(
    [&](const auto& s) {
      constexpr int Dim = std::decay_t<decltype(s)>::DIM;
      if (owners.extent(0) > 0) {
        detail::BarycentricInElements<Dim>(
          mesh_, LocatedPointValues(), FoundPoints(), owners, elem_ids_, bary_);
        return;
      }
      auto [found, results] = LocatePoints(s, LocatedPointValues());
      SetFound(found);
      const auto [lifted_elems, lifted_bary] = detail::LiftResults<Dim>(
        mesh_, Kokkos::View<const typename std::decay_t<decltype(s)>::Result*,
                            DeviceMemorySpace>(results));
      detail::GatherFoundLifts<Dim>(FoundPoints(), lifted_elems, lifted_bary,
                                    elem_ids_, bary_);
    },
    search_);
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
                               OmegaHGeometryOptions options)
  : mesh_(mesh),
    discretization_(std::make_shared<OmegaHDiscretization>(
      mesh, std::move(options.global_id_name), std::move(options.owned_mask))),
    system_(ResolveCoordinateSystem(std::move(system), mesh.dim())),
    search_(MakeGridSearch(mesh, options.search))
{
}

std::shared_ptr<BoundSource> OmegaHGeometry::BindImpl(
  const BindRequest& request) const
{
  PCMS_FUNCTION_TIMER;
  return std::make_shared<OmegaHMeshBinding>(shared_from_this(), request,
                                             search_, mesh_);
}

Kokkos::View<const LO*, DeviceMemorySpace> OmegaHGeometry::FindElementSeeds(
  Kokkos::View<const Real**, DeviceMemorySpace> points) const
{
  if (points.extent(1) != static_cast<std::size_t>(mesh_.dim())) {
    throw pcms_error("OmegaHGeometry::FindElementSeeds: point dimension "
                     "differs from the mesh dimension");
  }
  return std::visit(
    [&](const auto& s) -> Kokkos::View<const LO*, DeviceMemorySpace> {
      constexpr int Dim = std::decay_t<decltype(s)>::DIM;
      return detail::LiftElements<Dim>(
        mesh_, Kokkos::View<const typename std::decay_t<decltype(s)>::Result*,
                            DeviceMemorySpace>(s(points)));
    },
    search_);
}

SupportResults OmegaHGeometry::Neighbourhood(
  CoordinateView<DeviceMemorySpace> coords, const NeighbourhoodOptions& options,
  const KnownOwners& owners) const
{
  pcms::detail::CheckNeighbourhoodArguments(*this, coords, options,
                                            "OmegaHGeometry::Neighbourhood");
  const auto flat_points = flatten_to_omega_h_reals(coords.GetValues());
  const LO n = static_cast<LO>(coords.GetValues().extent(0));
  const int dim = mesh_.dim();
  int seed_dim = dim;
  Kokkos::View<const LO*, DeviceMemorySpace> seeds;
  if (owners.Hints(*this, n) &&
      (owners.entity_dim == 0 || owners.entity_dim == dim)) {
    seed_dim = owners.entity_dim;
    seeds = owners.entities;
    pcms::detail::CheckEntityIds(seeds, mesh_.nents(seed_dim),
                                 "OmegaHGeometry::Neighbourhood",
                                 seed_dim == 0 ? "vertex" : "element");
  } else {
    // the element the bind's search finds for each point (for a point outside
    // the mesh, the element nearest it; -1, and so no supports, if its grid
    // cell has none)
    const auto points = ConvertCoordsTo2D(flat_points, n, dim);
    seeds = FindElementSeeds(points);
  }
  return searchNeighbors(mesh_, /*site_dim=*/0, flat_points, seed_dim, seeds,
                         options.radius * options.radius, options.min_supports,
                         3 * options.min_supports, options.adapt_radius);
}

namespace
{

/// Determinant of a Dim x Dim matrix (Dim 2 or 3).
/// (A function of its own: CUDA does not allow a device lambda to capture a
/// variable for the first time inside an if constexpr.)
template <int Dim>
KOKKOS_INLINE_FUNCTION Real SimplexDeterminant(const Real (&j)[Dim][Dim])
{
  if constexpr (Dim == 2) {
    return j[0][0] * j[1][1] - j[0][1] * j[1][0];
  } else {
    return j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1]) -
           j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0]) +
           j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
  }
}

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
      determinant(e) = SimplexDeterminant<Dim>(j);
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

} // namespace pcms::backend::omega_h
