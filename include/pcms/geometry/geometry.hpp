#ifndef PCMS_GEOMETRY_GEOMETRY_HPP
#define PCMS_GEOMETRY_GEOMETRY_HPP

#include "pcms/utility/coordinate_system.hpp"
#include "pcms/utility/coordinate_view.hpp"
#include "pcms/field/out_of_bounds_policy.hpp"
#include "pcms/geometry/bound_source.hpp"
#include "pcms/geometry/supports.hpp"
#include "pcms/utility/memory_spaces.hpp"
#include "pcms/utility/types.hpp"
#include "pcms/utility/uniform_grid.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <string>

namespace pcms
{

class CoordinateMap; // pcms/field/coordinate_map.hpp
/// The topology spaces are defined on (opaque; internal to PCMS).
class Discretization;
class FieldLayout; // pcms/field/field_layout.hpp

/// @brief Options of the containing-cell search a geometry builds.
struct SearchOptions
{
  /// grid divisions per axis; 0 derives them from the element count
  int divisions = 0;
  Real tolerance = 1e-12;
};

/// @brief Options of Geometry::Neighbourhood.
struct NeighbourhoodOptions
{
  Real radius = 0.5;
  int min_supports = 10;
  /// upper bound of the supports of the mesh-adjacency search; 0 means the
  /// kind's own bound, 3 * min_supports, which is the only one supported yet
  /// (the kinds throw otherwise)
  int max_supports = 0;
  bool adapt_radius = true;
};

/// @brief Jacobian of the geometry map at bound points.
///
/// jacobian(r, i, j) = d x_j / d xi_i, determinant(r) = det. When
/// per_element is true, r is an element id; otherwise r indexes the bound
/// source's FoundPoints().
///
/// Implementer API: not covered by the public-API stability promise.
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
/// Lifetime: bound sources and the spaces created on a geometry hold it by
/// std::shared_ptr, so it outlives them; what the geometry refers to (e.g. a
/// mesh) must outlive the geometry. A bound source is immutable and
/// stays valid after later binds on the same geometry; evaluators hold their
/// bound source. Bind uses shared_from_this(), so a geometry must be owned by
/// a std::shared_ptr (create it with the kind's factory:
/// CreateUniformGridGeometry, CreatePointCloudGeometry or
/// backend::omega_h::CreateGeometry). Compatibility of spaces is decided by the
/// discretization (SameEntities): the Lagrange spaces created on a mesh
/// geometry share its Discretization object, so the test is pointer equality
/// for them.
class Geometry : public std::enable_shared_from_this<Geometry>
{
public:
  virtual ~Geometry() noexcept = default;
  Geometry(const Geometry&) = delete;
  Geometry& operator=(const Geometry&) = delete;

  /// @return the topology shared by spaces on this geometry, or null if it has
  ///         none; built-in kinds return one, external kinds normally keep
  ///         the default
  [[nodiscard]] virtual std::shared_ptr<const Discretization>
  GetDiscretization() const noexcept
  {
    return nullptr;
  }

  /// @return the native coordinate system: Bind takes points in it
  [[nodiscard]] virtual const std::shared_ptr<const CoordinateSystem>&
  GetCoordinateSystem() const noexcept = 0;

  /// @return whether Bind accepts OutOfBoundsMode::NEAREST_BOUNDARY
  [[nodiscard]] virtual bool SupportsNearestBoundary() const noexcept
  {
    return false;
  }

  /// @brief Localizes physical query points.
  /// @param coords query points, in GetCoordinateSystem() or, with `map`, in
  ///        its source coordinate system; may be released after
  /// @param policy what to do with points outside the geometry
  /// @param map map from the query coordinate system to GetCoordinateSystem(),
  ///        or null; the points are located at their images, and evaluators
  ///        created from the result re-express their values through it
  /// @return bound source for the evaluator sources on this geometry; its
  ///         LocatedPointValues() are the query points (or their images)
  /// @throws pcms_error if the query coordinate system is not
  ///         GetCoordinateSystem() (without a map) or the map does not take it
  ///         to GetCoordinateSystem(), or the policy is NEAREST_BOUNDARY and
  ///         SupportsNearestBoundary() is false, or the policy is ERROR and a
  ///         point is outside
  [[nodiscard]] std::shared_ptr<const BoundSource> Bind(
    CoordinateView<DeviceMemorySpace> coords, OutOfBoundsPolicy policy = {},
    std::shared_ptr<const CoordinateMap> map = nullptr) const;

  /// @brief Bind for points whose owners are known, e.g. points generated in
  ///        the elements of this geometry.
  /// @param coords query points, as for the coordinates overload
  /// @param owners the owner of each query point; used, instead of the
  ///        search, by the kinds that can (a cell hint on a mesh; vertex
  ///        and cell hints for the supports of a reconstruction geometry), and
  ///        ignored if it is for another geometry, not one entity per point,
  ///        or given with a map (it describes the unmapped points). The hint
  ///        is trusted, see KnownOwners: hinted points count as found.
  /// @throws pcms_error as the coordinates overload, or if a used hint names
  ///         an entity the geometry does not have
  /// @param policy what to do with points outside the geometry
  /// @param map map from the query coordinate system to GetCoordinateSystem(),
  ///        or null, as for the coordinates overload
  [[nodiscard]] std::shared_ptr<const BoundSource> Bind(
    CoordinateView<DeviceMemorySpace> coords, const KnownOwners& owners,
    OutOfBoundsPolicy policy = {},
    std::shared_ptr<const CoordinateMap> map = nullptr) const;

  /// @brief Localizes new query points in place of `previous`, a bound source
  ///        of this geometry: same policy and coordinate map.
  ///
  /// When nothing else holds `previous` (its shared_ptr and every Kokkos view
  /// it owns are uniquely held) its storage is reused, so a loop that
  /// releases its evaluators before each call reallocates none of the bound
  /// source's arrays in steady state when the point count does not grow (the
  /// kind's search may still allocate temporaries); otherwise this is an
  /// ordinary Bind. Binds with a coordinate map are not recycled in this
  /// version: they are bound anew with the map. An mdspan taken from a bound
  /// source is valid while a reference to that source is held, as always.
  /// @param previous bound source of this geometry; consumed
  /// @param coords query points, in the coordinate system Bind took them in
  /// @return the bound source of `coords`: `previous`, reused, or a new one
  /// @throws pcms_error if `previous` is null or belongs to another geometry,
  ///         or as Bind
  [[nodiscard]] std::shared_ptr<const BoundSource> Rebind(
    std::shared_ptr<const BoundSource> previous,
    CoordinateView<DeviceMemorySpace> coords) const;

  /// @brief Support sets of query points among this geometry's sites, for
  ///        reconstruction spaces.
  /// @throws pcms_error unless the kind overrides it, or if the coordinate
  ///         system does not have the identity metric (the distances are
  ///         Euclidean), or `options.max_supports` is neither 0 nor
  ///         3 * `options.min_supports` (other bounds are not honoured yet)
  /// @param owners the owner of each query point, as for Bind: the kinds
  ///        with an adjacency search start it there instead of locating the
  ///        points (a vertex hint starts from the elements around the vertex)
  [[nodiscard]] virtual SupportResults Neighbourhood(
    CoordinateView<DeviceMemorySpace> coords,
    const NeighbourhoodOptions& options, const KnownOwners& owners = {}) const;

  /// @brief Jacobian at the bound points.
  /// @param bound a bound source of this geometry
  /// @return an empty view unless the kind overrides it
  /// @throws pcms_error from the kinds that override it if `bound` belongs to
  ///         another geometry
  [[nodiscard]] virtual DifferentialView Differential(
    const BoundSource& bound) const;

protected:
  Geometry() = default;

  /// @param request points to locate, already checked to be in
  ///        GetCoordinateSystem() (mapped there if the bind had a map), and the
  ///        query points the bound source keeps
  /// @return the bound source of the kind
  [[nodiscard]] virtual std::shared_ptr<BoundSource> BindImpl(
    const BindRequest& request) const = 0;

private:
  // creates the passkey of BoundSource::Rebind
  //
  // Implementer API: not covered by the public-API stability promise.
  struct RebindKey : BoundSource::RebindKey
  {
    RebindKey() = default;
  };
};

/// The geometry of a 2D uniform grid: an affine map of the unit cells, located
/// in closed form.
/// @param grid the grid
/// @param system the grid's coordinate system, or null for Cartesian
std::shared_ptr<Geometry> CreateUniformGridGeometry(
  UniformGrid<2> grid, std::shared_ptr<const CoordinateSystem> system);

/// The geometry of a 3D uniform grid.
/// @param grid the grid
/// @param system the grid's coordinate system, or null for Cartesian
std::shared_ptr<Geometry> CreateUniformGridGeometry(
  UniformGrid<3> grid, std::shared_ptr<const CoordinateSystem> system);

/// A point-cloud geometry whose sites are `coords` (copied to the device).
/// @param coords the sites
/// @param options the support search of the geometry
/// @throws pcms_error if the coordinate system does not have the identity
///         metric (MLS distances are Euclidean)
std::shared_ptr<Geometry> CreatePointCloudGeometry(
  CoordinateView<HostMemorySpace> coords, NeighbourhoodOptions options = {});

/// A point-cloud geometry whose sites are the DOF holders of `sites` (e.g. the
/// entities of one dimension of a mesh, through an Omega_h backend layout,
/// which the adjacency search then serves).
/// @param sites the layout of the sites
/// @param options the support search of the geometry
/// @throws pcms_error if `sites` is null or its coordinate system does not
///         have the identity metric
std::shared_ptr<Geometry> CreatePointCloudGeometry(
  std::shared_ptr<const FieldLayout> sites, NeighbourhoodOptions options = {});

namespace detail
{

/// @brief The checks every Neighbourhood makes of its arguments.
/// @param geometry geometry whose Neighbourhood is called
/// @param coords the query points
/// @param options the search options
/// @param caller name of the calling function, for the message
/// @throws pcms_error if `coords` is not in the geometry's coordinate system,
///         that system does not have the identity metric, or
///         `options.max_supports` is neither 0 nor 3 * `options.min_supports`
void CheckNeighbourhoodArguments(
  const Geometry& geometry, const CoordinateView<DeviceMemorySpace>& coords,
  const NeighbourhoodOptions& options, const std::string& caller);

/// @brief Checks that every id of a hint is one of `num_entities` entities.
/// @param ids entity ids, e.g. KnownOwners::entities
/// @param num_entities number of entities the ids index
/// @param caller name of the checking function, for the message
/// @param entity what the ids index, e.g. "element", for the message
/// @throws pcms_error naming the range of the ids if one is outside
///         [0, num_entities)
void CheckEntityIds(Kokkos::View<const LO*, DeviceMemorySpace> ids,
                    LO num_entities, const std::string& caller,
                    const std::string& entity);

} // namespace detail

} // namespace pcms

#endif // PCMS_GEOMETRY_GEOMETRY_HPP
