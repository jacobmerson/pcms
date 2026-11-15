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

/// Owning element and barycentric coordinates ([row][vertex]) of the points
/// `rows` of `points`, given the owning element of each point.
template <int Dim>
std::pair<Kokkos::View<LO*, DeviceMemorySpace>,
          Kokkos::View<Real**, DeviceMemorySpace>>
BarycentricInElements(Omega_h::Mesh& mesh,
                      Kokkos::View<const Real**, DeviceMemorySpace> points,
                      Kokkos::View<const LO*, DeviceMemorySpace> rows,
                      Kokkos::View<const LO*, DeviceMemorySpace> owning_ids)
{
  const auto n = static_cast<LO>(rows.extent(0));
  Kokkos::View<LO*, DeviceMemorySpace> elem_ids("omega_h_elem_ids", n);
  Kokkos::View<Real**, DeviceMemorySpace> bary("omega_h_bary", n, Dim + 1);
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
  return {elem_ids, bary};
}

} // namespace detail

/// Bound source of an Omega_h simplex mesh: element and barycentric
/// coordinates of each found point, in FoundPoints() order.
class OmegaHMeshBinding final : public BoundSource
{
public:
  /// Locates `request.coords` with `search`.
  OmegaHMeshBinding(std::shared_ptr<const Geometry> geometry,
                    const BindRequest& request,
                    const GridPointSearchVariant& search, Omega_h::Mesh& mesh);

  /// Takes element ids and barycentric coordinates as given.
  OmegaHMeshBinding(std::shared_ptr<const Geometry> geometry,
                    const ParametricPoints& points, OutOfBoundsPolicy policy,
                    Omega_h::Mesh& mesh);

  [[nodiscard]] Kokkos::View<const LO*, DeviceMemorySpace> ElementIds()
    const noexcept
  {
    return elem_ids_;
  }

  /// [found point][vertex]
  [[nodiscard]] Kokkos::View<const Real**, DeviceMemorySpace> Barycentric()
    const noexcept
  {
    return bary_;
  }

private:
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

  /// Partitioned if the mesh's communicator has more than one rank.
  [[nodiscard]] Distribution GetDistribution() const noexcept override;

  [[nodiscard]] Omega_h::Mesh& GetMesh() const noexcept { return mesh_; }

  [[nodiscard]] const GridPointSearchVariant& GetSearch() const noexcept
  {
    return search_;
  }

  /// x = sum_v lambda_v x_v for each point given by element and barycentric
  /// coordinates; throws on the same malformed points as Bind.
  [[nodiscard]] Kokkos::View<Real**, DeviceMemorySpace> MapToPhysical(
    const ParametricPoints& points) const;

  /// Supports among the mesh vertices, by the mesh-adjacency search.
  /// `options.query_layout` is not used: the same-mesh search from element
  /// centroids does not apply to vertex sites.
  [[nodiscard]] SupportResults Neighbourhood(
    CoordinateView<DeviceMemorySpace> coords,
    const NeighbourhoodOptions& options) const override;

  /// The per-element constant Jacobian of the affine simplex map, computed
  /// for all elements on the first call.
  [[nodiscard]] DifferentialView Differential(
    const BoundSource& bound) const override;

protected:
  [[nodiscard]] std::shared_ptr<BoundSource> BindImpl(
    const BindRequest& request) const override;
  [[nodiscard]] std::shared_ptr<BoundSource> BindParametricImpl(
    const ParametricPoints& points, OutOfBoundsPolicy policy) const override;

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
