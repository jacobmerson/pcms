#ifndef PCMS_COUPLING_POINT_SEARCH_H
#define PCMS_COUPLING_POINT_SEARCH_H
#include <cassert>

#include <utility>
#include <variant>

#include <Kokkos_Core.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_bbox.hpp>
#include <Omega_h_shape.hpp>

#include "pcms/utility/memory_spaces.hpp"
#include "pcms/utility/types.hpp"
#include "pcms/utility/uniform_grid.hpp"
#include "pcms/utility/bounding_box.hpp"

namespace pcms::localization
{

//
// TODO take a bounding box as we may want a bbox that's bigger than the mesh!
// this function is in the public header for testing, but should not be directly
// used
namespace detail
{
Kokkos::Crs<LO, Kokkos::DefaultExecutionSpace, void, LO>
construct_intersection_map_2d(Omega_h::Mesh& mesh,
                              Kokkos::View<Uniform2DGrid[1]> grid,
                              int num_grid_cells);

Kokkos::Crs<LO, Kokkos::DefaultExecutionSpace, void, LO>
construct_intersection_map_3d(Omega_h::Mesh& mesh,
                              Kokkos::View<Uniform3DGrid[1]> grid,
                              int num_grid_cells);

/// O(num_cells * nelems) brute-force candidate map. Test oracle for the
/// element-major build above; never use in production.
Kokkos::Crs<LO, Kokkos::DefaultExecutionSpace, void, LO>
construct_intersection_map_reference_2d(Omega_h::Mesh& mesh,
                                        Kokkos::View<Uniform2DGrid[1]> grid,
                                        int num_grid_cells);

/// \see construct_intersection_map_reference_2d
Kokkos::Crs<LO, Kokkos::DefaultExecutionSpace, void, LO>
construct_intersection_map_reference_3d(Omega_h::Mesh& mesh,
                                        Kokkos::View<Uniform3DGrid[1]> grid,
                                        int num_grid_cells);
} // namespace detail

[[nodiscard]] KOKKOS_FUNCTION bool triangle_intersects_bbox(
  const Omega_h::Matrix<2, 3>& coords, const AABBox<2>& bbox);

/// The search reports the entity the point lies on and the point's parametric
/// coordinates on that entity. It does not choose an element; a consumer that
/// needs one lifts the point into an adjacent element (see OmegaHMeshBinding).
template <int dim>
class PointLocalizationSearch
{
public:
  struct Result
  {
    enum class Dimensionality
    {
      VERTEX = 0,
      EDGE = 1,
      FACE = 2,
      REGION = 3,
      NO_INTERSECT = 4
    };

    Dimensionality dimensionality;
    /// id of the entity of that dimension that contains the point;
    /// OutsideId(nearest entity) for a point outside the mesh; -1 for
    /// NO_INTERSECT
    LO entity;
    /// The point on the entity in reduced coordinates: the entity's
    /// dimension many entries are used (none for a vertex, 1 on an edge, 2 on
    /// a face, 3 in a region), the others are 0. With lambda_0..lambda_d the
    /// barycentric coordinates of the point in the entity, in the order of
    /// the entity's vertices (an edge's from the edge-to-vertex adjacency, a
    /// face's or region's from the element-to-vertex adjacency), the entries
    /// are (lambda_1, ..., lambda_d): the point is
    /// x_0 + sum_j parametric[j] * (x_{j+1} - x_0). For a point outside the
    /// mesh it is the position of the point on the nearest entity where the
    /// search has one (the projection onto an edge, the extrapolated position
    /// in a region) and zeros otherwise.
    Omega_h::Vector<dim> parametric;

    /// `entity` of a point outside the mesh whose nearest entity is `id`.
    [[nodiscard]] KOKKOS_INLINE_FUNCTION static constexpr LO OutsideId(LO id)
    {
      return -id - 1;
    }

    /// True when the point was not located inside any entity.
    [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool IsOutside() const
    {
      return entity < 0;
    }

    /// Id of the entity of dimension `dimensionality`: the containing entity
    /// of a located point, the nearest entity of an outside point, or -1 for
    /// NO_INTERSECT.
    [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr LO EntityId() const
    {
      if (dimensionality == Dimensionality::NO_INTERSECT) {
        return -1;
      }
      return entity < 0 ? -entity - 1 : entity;
    }
  };

  static constexpr auto DIM = dim;

  using PointSearchTolerances = Kokkos::View<Real[DIM]>;

  explicit PointLocalizationSearch(const PointSearchTolerances& tolerances)
    : tolerances_(tolerances)
  {
    assert(tolerances_.is_allocated());
  }

  virtual Kokkos::View<Result*> operator()(
    Kokkos::View<const Real* [dim]> point) const = 0;

  virtual ~PointLocalizationSearch() = default;

protected:
  PointSearchTolerances tolerances_;
};

using PointLocalizationSearch2D = PointLocalizationSearch<2>;
using PointLocalizationSearch3D = PointLocalizationSearch<3>;

class GridPointSearch2D : public PointLocalizationSearch2D
{
  using CandidateMapT =
    Kokkos::Crs<LO, Kokkos::DefaultExecutionSpace, void, LO>;

public:
  using Result = PointLocalizationSearch2D::Result;

  GridPointSearch2D(Omega_h::Mesh& mesh, LO Nx, LO Ny);
  GridPointSearch2D(Omega_h::Mesh& mesh, LO Nx, LO Ny,
                    const PointSearchTolerances& tolerances);

  /**
   * Given points in global coordinates, returns for each the entity (vertex,
   * edge or face, chosen with the vertex and edge tolerances) containing it
   * and the point's position on that entity.
   */
  Kokkos::View<Result*> operator()(
    Kokkos::View<const Real* [DIM]> point) const override;

private:
  Omega_h::Adj tris2edges_adj_;
  Omega_h::Adj tris2verts_adj_;
  Omega_h::Adj edges2verts_adj_;
  Kokkos::View<Uniform2DGrid[1]> grid_{"uniform grid"};
  CandidateMapT candidate_map_;
  Omega_h::LOs tris2verts_;
  Omega_h::Reals coords_;
};

class GridPointSearch3D : public PointLocalizationSearch3D
{
  using CandidateMapT =
    Kokkos::Crs<LO, Kokkos::DefaultExecutionSpace, void, LO>;

public:
  using Result = PointLocalizationSearch3D::Result;

  GridPointSearch3D(Omega_h::Mesh& mesh, LO Nx, LO Ny, LO Nz);
  GridPointSearch3D(Omega_h::Mesh& mesh, LO Nx, LO Ny, LO Nz,
                    const PointSearchTolerances& tolerances);

  /**
   * Given points in global coordinates, returns for each the id of a
   * tetrahedron containing it and the point's reduced coordinates in that
   * tetrahedron. Containment is tested with a Cartesian tolerance band: the
   * point may lie up to `tolerances(2)` outside any face plane (the band is
   * applied per barycentric coordinate, scaled by the gradient norm of that
   * coordinate, so it is the same distance from both sides of a shared face).
   * The result is always labelled REGION. When several tetrahedra contain the
   * point (it lies on a shared face, edge or vertex) the smallest id is
   * returned; this is an interim tie-break that hides a genuine ambiguity,
   * see docs/point_search_shared_face_plan.md.
   *
   * If no candidate contains the point, the result is the candidate whose
   * face is closest, encoded with Result::OutsideId or NO_INTERSECT with
   * id -1 when the grid cell has no candidates at all.
   */
  Kokkos::View<Result*> operator()(
    Kokkos::View<const Real* [DIM]> point) const override;

private:
  Omega_h::Adj tris2edges_adj_;
  Omega_h::Adj tris2verts_adj_;
  Omega_h::Adj edges2verts_adj_;
  Kokkos::View<UniformGrid<DIM>[1]> grid_{"uniform grid"};
  CandidateMapT candidate_map_;
  Omega_h::LOs tris2verts_;
  Omega_h::Reals coords_;
};

/// A grid search of either spatial dimension, as owned by components that are
/// dimension-agnostic at compile time (e.g. an evaluator factory over an
/// Omega_h mesh) and handed to consumers that need to locate points on the same
/// mesh without building a second search.
using GridPointSearchVariant =
  std::variant<GridPointSearch2D, GridPointSearch3D>;

/// @brief Locates each query point.
/// @param search containing-entity search over a mesh
/// @param points query points, one row per point
/// @return per point, whether it is in the mesh (not outside and not
///         NO_INTERSECT), and the search's result
template <typename Search>
std::pair<Kokkos::View<bool*>, Kokkos::View<typename Search::Result*>>
LocatePoints(const Search& search, Kokkos::View<const Real**> points)
{
  const auto results = search(points);
  const auto n = static_cast<LO>(points.extent(0));
  Kokkos::View<bool*> found(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "found_in_mesh"), n);
  pcms::detail::ForEach(
    "LocatePoints", n, KOKKOS_LAMBDA(const LO i) {
      found(i) = !results(i).IsOutside() &&
                 results(i).dimensionality !=
                   Search::Result::Dimensionality::NO_INTERSECT;
    });
  return {found, results};
}

} // namespace pcms::localization
#endif // PCMS_COUPLING_POINT_SEARCH_H
