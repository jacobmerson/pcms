#ifndef PCMS_GEOMETRY_GEOMETRY_HPP
#define PCMS_GEOMETRY_GEOMETRY_HPP

#include "pcms/discretization/discretization.h"
#include "pcms/field/coordinate_system.hpp"
#include "pcms/field/coordinate_view.hpp"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/geometry/bound_source.hpp"
#include "pcms/geometry/parametric_points.hpp"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/types.h"
#include <Kokkos_Core.hpp>
#include <memory>

namespace pcms
{

struct SupportResults; // pcms/localization/adj_search.hpp

/// @brief Options of the containing-cell search a geometry builds.
struct SearchOptions
{
  /// grid divisions per axis; 0 derives them from the element count
  int divisions = 0;
  Real tolerance = 1e-12;
};

/// @brief How a geometry is laid out over MPI ranks.
enum class Distribution
{
  Replicated, ///< every rank holds the whole geometry
  Partitioned ///< each rank holds a part; a point may be owned elsewhere
};

/// @brief Options of Geometry::Neighbourhood.
struct NeighbourhoodOptions
{
  Real radius = 0.5;
  int min_supports = 10;
  /// not honoured yet: Neighbourhood throws unless it is the default
  int max_supports = 30;
  bool adapt_radius = true;
};

/// @brief Jacobian of the geometry map at bound points.
///
/// jacobian(r, i, j) = d x_j / d xi_i, determinant(r) = det. When
/// per_element is true, r is an element id; otherwise r indexes the bound
/// source's FoundPoints().
struct DifferentialView
{
  Kokkos::View<const Real***, DeviceMemorySpace> jacobian;
  Kokkos::View<const Real*, DeviceMemorySpace> determinant;
  bool per_element = false;
  [[nodiscard]] bool Empty() const noexcept { return jacobian.extent(0) == 0; }
};

/// @brief The coordinate function of a discretization and its inverse.
///
/// A Geometry owns one Discretization (the topology spaces are defined on),
/// a native coordinate system, the search that locates physical points in
/// the discretization, and, when the kind has one, the Jacobian of the map
/// from parametric to physical coordinates. Function spaces are created on a
/// geometry; Bind produces bound sources every space on the geometry accepts.
///
/// Lifetime: a geometry outlives its bound sources and the spaces created on
/// it. A bound source stays valid after later binds. Bind uses
/// shared_from_this(), so a geometry must be owned by a std::shared_ptr
/// (create it with the kind's Create or std::make_shared).
class Geometry : public std::enable_shared_from_this<Geometry>
{
public:
  virtual ~Geometry() noexcept = default;
  Geometry(const Geometry&) = delete;
  Geometry& operator=(const Geometry&) = delete;

  /// @return the topology spaces on this geometry are defined on
  [[nodiscard]] virtual std::shared_ptr<const Discretization>
  GetDiscretization() const noexcept = 0;

  /// @return the native coordinate system: Bind takes points in it
  [[nodiscard]] virtual const std::shared_ptr<const CoordinateSystem>&
  GetCoordinateSystem() const noexcept = 0;

  /// @return how the geometry is laid out over MPI ranks
  [[nodiscard]] virtual Distribution GetDistribution() const noexcept
  {
    return Distribution::Replicated;
  }

  /// @return whether Bind accepts OutOfBoundsMode::NEAREST_BOUNDARY
  [[nodiscard]] virtual bool SupportsNearestBoundary() const noexcept
  {
    return false;
  }

  /// @brief Localizes physical query points.
  /// @param coords points in GetCoordinateSystem(); may be released after
  /// @param policy what to do with points outside the geometry
  /// @return bound source for the evaluator sources on this geometry
  /// @throws pcms_error if the coordinate system differs, or the policy is
  ///         NEAREST_BOUNDARY and SupportsNearestBoundary() is false, or the
  ///         policy is ERROR and a point is outside
  [[nodiscard]] std::shared_ptr<const BoundSource> Bind(
    CoordinateView<DeviceMemorySpace> coords,
    OutOfBoundsPolicy policy = {}) const;

  /// @brief Bind for the DOF-holder coordinates of a layout; records the
  ///        layout on the bound source as QueryLayout(). The layout is an
  ///        opaque handle here (pcms_geometry does not depend on pcms_field);
  ///        it is never dereferenced in this library.
  /// @param dof_holder_coords DOF-holder coordinates of `query_layout`
  /// @param query_layout layout whose DOF holders are the query points
  /// @param policy what to do with points outside the geometry
  [[nodiscard]] std::shared_ptr<const BoundSource> Bind(
    CoordinateView<DeviceMemorySpace> dof_holder_coords,
    std::shared_ptr<const FieldLayout> query_layout,
    OutOfBoundsPolicy policy = {}) const;

  /// @brief Binds points given in parametric coordinates; no search.
  /// @param points parametric points of this geometry's kind
  /// @param policy out-of-bounds policy of the evaluators created from the
  ///        result
  /// @throws pcms_error if `points.ids` and `points.coords` differ in length,
  ///         or the kind has no parametric coordinates, or there are points
  ///         and an id or the number of coordinate columns does not fit the
  ///         kind (no points give an empty bound source)
  [[nodiscard]] std::shared_ptr<const BoundSource> Bind(
    const ParametricPoints& points, OutOfBoundsPolicy policy = {}) const;

  /// @brief Support sets of query points among this geometry's sites, for
  ///        reconstruction spaces.
  /// @throws pcms_error unless the kind overrides it, or if
  ///         `options.max_supports` is not the default (not honoured yet)
  [[nodiscard]] virtual SupportResults Neighbourhood(
    CoordinateView<DeviceMemorySpace> coords,
    const NeighbourhoodOptions& options) const;

  /// @brief Jacobian at the bound points.
  /// @param bound a bound source of this geometry
  /// @return an empty view unless the kind overrides it
  /// @throws pcms_error from the kinds that override it if `bound` belongs to
  ///         another geometry
  [[nodiscard]] virtual DifferentialView Differential(
    const BoundSource& bound) const;

protected:
  Geometry() = default;

  /// @param request points already checked to be in GetCoordinateSystem()
  /// @return the bound source of the kind
  [[nodiscard]] virtual std::shared_ptr<BoundSource> BindImpl(
    const BindRequest& request) const = 0;

  /// @param points parametric points with ids and coords of one length
  /// @param policy out-of-bounds policy of the evaluators
  /// @return the bound source of the kind
  [[nodiscard]] virtual std::shared_ptr<BoundSource> BindParametricImpl(
    const ParametricPoints& points, OutOfBoundsPolicy policy) const = 0;
};

} // namespace pcms

#endif // PCMS_GEOMETRY_GEOMETRY_HPP
