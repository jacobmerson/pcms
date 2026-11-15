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

/// @brief Barycentric coordinates of points in their owning elements.
/// @tparam Dim mesh dimension
/// @param mesh simplex mesh
/// @param points query points, one row per point
/// @param rows points to compute, as indices into `points`
/// @param owning_ids owning element of each point
/// @return owning element and barycentric coordinates ([row][vertex]) of each
///         row
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

/// @brief Bound source of an Omega_h simplex mesh: element and barycentric
///        coordinates of each found point.
class OmegaHMeshBinding final : public BoundSource
{
public:
  /// @brief Locates `request.coords` with `search`.
  /// @param geometry geometry the points are localized on
  /// @param request query points and out-of-bounds policy
  /// @param search containing-element search over `mesh`
  /// @param mesh simplex mesh of the geometry
  OmegaHMeshBinding(std::shared_ptr<const Geometry> geometry,
                    const BindRequest& request,
                    const GridPointSearchVariant& search, Omega_h::Mesh& mesh);

  /// @brief Takes element ids and barycentric coordinates as given.
  /// @param geometry geometry the points are localized on
  /// @param points element of each point and its barycentric coordinates
  ///        (dim + 1 columns); no points gives an empty bound source
  /// @param policy out-of-bounds policy of the evaluators
  /// @param mesh simplex mesh of the geometry
  /// @throws pcms_error if there are points and `points.coords` does not have
  ///         dim + 1 columns or an id is not an element of `mesh`
  OmegaHMeshBinding(std::shared_ptr<const Geometry> geometry,
                    const ParametricPoints& points, OutOfBoundsPolicy policy,
                    Omega_h::Mesh& mesh);

  /// @return element of each found point, in FoundPoints() order
  [[nodiscard]] Kokkos::View<const LO*, DeviceMemorySpace> ElementIds()
    const noexcept
  {
    return elem_ids_;
  }

  /// @return barycentric coordinates of each found point in its element
  ///         ([found point][vertex]), in FoundPoints() order
  [[nodiscard]] Kokkos::View<const Real**, DeviceMemorySpace> Barycentric()
    const noexcept
  {
    return bary_;
  }

private:
  Kokkos::View<LO*, DeviceMemorySpace> elem_ids_;
  Kokkos::View<Real**, DeviceMemorySpace> bary_;
};

/// @brief Geometry of an Omega_h simplex mesh with straight-sided elements.
class OmegaHGeometry final : public Geometry
{
public:
  /// @brief Creates the geometry owned by a std::shared_ptr, as Bind needs.
  /// @param mesh 2D or 3D simplex mesh with classification; must outlive the
  ///        geometry
  /// @param system coordinate system of the mesh coordinates
  /// @param options search options
  [[nodiscard]] static std::shared_ptr<OmegaHGeometry> Create(
    Omega_h::Mesh& mesh, std::shared_ptr<const CoordinateSystem> system,
    SearchOptions options = {});

  /// @brief Create it with Create or std::make_shared: Bind uses
  ///        shared_from_this().
  /// @param mesh 2D or 3D simplex mesh with classification; must outlive the
  ///        geometry
  /// @param system coordinate system of the mesh coordinates
  /// @param options search options
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

  /// @return Partitioned if the mesh's communicator has more than one rank
  [[nodiscard]] Distribution GetDistribution() const noexcept override;

  /// @return the mesh of this geometry
  [[nodiscard]] Omega_h::Mesh& GetMesh() const noexcept { return mesh_; }

  /// @return the containing-element search over the mesh
  [[nodiscard]] const GridPointSearchVariant& GetSearch() const noexcept
  {
    return search_;
  }

  /// @brief MLS supports of the query points among the mesh vertices.
  /// @throws pcms_error if `options.max_supports` is not the default
  [[nodiscard]] SupportResults Neighbourhood(
    CoordinateView<DeviceMemorySpace> coords,
    const NeighbourhoodOptions& options) const override;

  /// @brief Per-element constant Jacobian of the affine simplex map and its
  ///        determinant; computed on first call.
  /// @throws pcms_error if `bound` belongs to another geometry
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

/// @brief The grid search over a mesh, with options.divisions divisions per
///        axis when nonzero and DivisionsPerAxisForMesh otherwise.
/// @throws std::invalid_argument unless the mesh is 2D or 3D
GridPointSearchVariant MakeGridSearch(Omega_h::Mesh& mesh,
                                      const SearchOptions& options);

} // namespace pcms

#endif // PCMS_FIELD_EVALUATOR_OMEGA_H_GEOMETRY_HPP
