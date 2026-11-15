#ifndef PCMS_FIELD_EVALUATOR_OMEGA_H_GEOMETRY_HPP
#define PCMS_FIELD_EVALUATOR_OMEGA_H_GEOMETRY_HPP

#include "pcms/discretization/discretization/omega_h.hpp"
#include "pcms/geometry/geometry.hpp"
#include "pcms/localization/point_search.h"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/types.h"
#include <Kokkos_Core.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_shape.hpp>
#include <memory>
#include <mutex>
#include <utility>

namespace pcms
{

namespace detail
{

/// Writes the owning element and barycentric coordinates ([row][vertex]) of
/// the points `rows` of `points`, given the owning element of each point, to
/// the first rows of `elem_ids` and `bary`, which are reallocated only when
/// they are too small (detail::ReserveRows).
template <int Dim>
void BarycentricInElements(
  Omega_h::Mesh& mesh, Kokkos::View<const Real**, DeviceMemorySpace> points,
  Kokkos::View<const LO*, DeviceMemorySpace> rows,
  Kokkos::View<const LO*, DeviceMemorySpace> owning_ids,
  Kokkos::View<LO*, DeviceMemorySpace>& elem_ids,
  Kokkos::View<Real**, DeviceMemorySpace>& bary)
{
  const auto n = static_cast<LO>(rows.extent(0));
  ReserveRows(elem_ids, "omega_h_elem_ids", n);
  ReserveRows(bary, "omega_h_bary", n, Dim + 1);
  const auto elem_verts = mesh.ask_elem_verts();
  const auto mesh_coords = mesh.coords();
  Kokkos::parallel_for(
    "BarycentricInElements",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, n),
    KOKKOS_LAMBDA(const LO k) {
      const LO i = rows(k);
      const LO elem = owning_ids(i);
      elem_ids(k) = elem;
      Omega_h::Few<Omega_h::Vector<Dim>, Dim + 1> vertex_coords;
      for (int v = 0; v < Dim + 1; ++v) {
        vertex_coords[v] = Omega_h::get_vector<Dim>(
          mesh_coords, elem_verts[elem * (Dim + 1) + v]);
      }
      Omega_h::Vector<Dim> point;
      for (int d = 0; d < Dim; ++d) {
        point[d] = points(i, d);
      }
      const auto local =
        Omega_h::barycentric_from_global<Dim, Dim>(point, vertex_coords);
      for (int v = 0; v < Dim + 1; ++v) {
        bary(k, v) = local[v];
      }
    });
}

} // namespace detail

/// Bound source of an Omega_h simplex mesh: element and barycentric
/// coordinates of each found point, in FoundPoints() order.
class OmegaHMeshBinding final : public BoundSource
{
public:
  /// Locates `request.coords` with `search`, or, if `request.owners` gives
  /// the element of every point, takes those elements without a search (the
  /// points then all count as found). `search` and `mesh` are the geometry's;
  /// a rebind searches with them again.
  OmegaHMeshBinding(std::shared_ptr<const Geometry> geometry,
                    const BindRequest& request,
                    const GridPointSearchVariant& search, Omega_h::Mesh& mesh);

  [[nodiscard]] Kokkos::View<const LO*, DeviceMemorySpace> ElementIds()
    const noexcept
  {
    return Kokkos::subview(elem_ids_, Kokkos::make_pair(LO{0}, NumFound()));
  }

  /// [found point][vertex]
  [[nodiscard]] Kokkos::View<const Real**, DeviceMemorySpace> Barycentric()
    const noexcept
  {
    return Kokkos::subview(bary_, Kokkos::make_pair(LO{0}, NumFound()),
                           Kokkos::ALL);
  }

protected:
  void RebindImpl() override;

  [[nodiscard]] bool UnsharedImpl() const noexcept override
  {
    return elem_ids_.use_count() <= 1 && bary_.use_count() <= 1;
  }

private:
  [[nodiscard]] LO NumFound() const noexcept
  {
    return static_cast<LO>(FoundPoints().extent(0));
  }

  /// Fills elem_ids_ and bary_ for the elements `owners` gives, or, if it is
  /// empty, for the elements the search finds (and sets the found points).
  void Locate(Kokkos::View<const LO*, DeviceMemorySpace> owners);

  const GridPointSearchVariant& search_;
  Omega_h::Mesh& mesh_;
  Kokkos::View<LO*, DeviceMemorySpace> elem_ids_;
  Kokkos::View<Real**, DeviceMemorySpace> bary_;
};

/// Geometry of a 2D or 3D Omega_h simplex mesh (with classification) with
/// straight-sided elements. Construction builds the containing-element search
/// over the mesh, which costs about as much as a bind: create one geometry per
/// mesh and share it.
class OmegaHGeometry final : public Geometry
{
public:
  [[nodiscard]] static std::shared_ptr<OmegaHGeometry> Create(
    Omega_h::Mesh& mesh, std::shared_ptr<const CoordinateSystem> system,
    SearchOptions options = {});

  /// Use Create or std::make_shared: Bind uses shared_from_this().
  OmegaHGeometry(Omega_h::Mesh& mesh,
                 std::shared_ptr<const CoordinateSystem> system,
                 SearchOptions options = {});

  [[nodiscard]] std::shared_ptr<const Discretization> GetDiscretization()
    const noexcept override
  {
    return discretization_;
  }

  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>&
  GetCoordinateSystem() const noexcept override
  {
    return system_;
  }

  [[nodiscard]] Omega_h::Mesh& GetMesh() const noexcept { return mesh_; }

  /// Supports among the mesh vertices, by the mesh-adjacency search from the
  /// owner of each point: a cell or vertex hint of this geometry, or else the
  /// element Bind's search finds. A point outside the mesh starts from the
  /// element nearest it, so it gets supports near the boundary; a point whose
  /// search cell holds no element gets none.
  [[nodiscard]] SupportResults Neighbourhood(
    CoordinateView<DeviceMemorySpace> coords,
    const NeighbourhoodOptions& options,
    const KnownOwners& owners = {}) const override;

  /// The per-element constant Jacobian of the affine simplex map, computed
  /// for all elements on the first call.
  [[nodiscard]] DifferentialView Differential(
    const BoundSource& bound) const override;

protected:
  [[nodiscard]] std::shared_ptr<BoundSource> BindImpl(
    const BindRequest& request) const override;

private:
  Omega_h::Mesh& mesh_;
  std::shared_ptr<const OmegaHDiscretization> discretization_;
  std::shared_ptr<const CoordinateSystem> system_;
  GridPointSearchVariant search_;
  // Differential is computed once, lazily; the geometry is const and shared,
  // hence the once_flag. This is the one mutable member and it is a cache.
  mutable std::once_flag differential_once_;
  mutable DifferentialView differential_;
};

/// The grid search over a mesh, with options.divisions divisions per axis when
/// nonzero and DivisionsPerAxisForMesh otherwise.
/// @throws std::invalid_argument unless the mesh is 2D or 3D
GridPointSearchVariant MakeGridSearch(Omega_h::Mesh& mesh,
                                      const SearchOptions& options);

} // namespace pcms

#endif // PCMS_FIELD_EVALUATOR_OMEGA_H_GEOMETRY_HPP
