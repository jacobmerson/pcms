#ifndef PCMS_FIELD_EMBEDDING_HPP
#define PCMS_FIELD_EMBEDDING_HPP

#include "pcms/field/coordinate_map.hpp"
#include "pcms/field/field.hpp"
#include "pcms/field/function_space.hpp"
#include "pcms/geometry/geometry.hpp"
#include <memory>

namespace pcms
{

/// @brief The coordinate map of a position Function: x(xi) = position(xi),
///        from the chart of the position's (parameter) geometry to the chart
///        of its ValueBasis::Coordinates basis.
///
/// Bind evaluates the position and its Gradient at the parameter points (no
/// inverse): the mapped points and the Jacobian J(p, i, j) = d x_j / d xi_i.
/// Values evaluated at the mapped points flow against the map
/// (BoundCoordinateMap::ValueTransform).
/// @param position a Function with a ValueBasis::Coordinates basis, of a space
///        whose evaluator supports Gradient; the parameter geometry and the
///        chart have at most 3 dimensions
/// @throws pcms_error if the position is null, its values are not
///         coordinates, its space has no Gradient or a dimension is above 3
[[nodiscard]] std::shared_ptr<const CoordinateMap> CreateCoordinateMap(
  std::shared_ptr<const Function<Real>> position);

/// @brief Options of CreateEmbeddedGeometry.
struct EmbeddingOptions
{
  /// the search: `divisions` search bins per axis (0 derives it from the
  /// cell count), `tolerance` the relative slack of the point-in-cell test
  SearchOptions search;
};

/// @brief A geometry whose cells are those of a structured parameter geometry
///        and whose chart is the chart of a position Function over it.
///
/// Bind of physical points inverts the position: the images of the cells
/// are binned by their bounding boxes, and Gauss-Newton (Newton for a square
/// Jacobian) finds the parameter coordinates in the candidate cells of a
/// point's bin. Periodic axes of the chart are wrapped. Bind also accepts
/// points in the parameter chart (GetParameterSystem), located directly (no
/// search). The BoundPoints carry the parameter coordinates, so the
/// evaluators of spaces on the parameter geometry evaluate unchanged
/// (CreateEmbeddedSpace). The geometric factors at the points are J, the
/// Gradient of the position, and the metric it induces, g = J^T G J (G the
/// chart's metric at the image); their volume factor is the physical volume
/// per unit of parameter volume. The geometry does not support
/// OutOfBoundsMode::NEAREST_BOUNDARY, and a Rebind searches anew.
/// @param parameter_geometry a uniform-grid geometry (the cells are boxes in
///        the parameter space), whose chart is a parameter chart
///        (csys::CreateParameterChart) or at least not the position's chart
/// @param position a Function on `parameter_geometry` with a
///        ValueBasis::Coordinates basis, whose space evaluates Gradient
/// @param options the options of the search
/// @throws pcms_error if the geometry is not a uniform grid, the position is
///         not a Function of coordinates on it with a Gradient, or the
///         parameter chart is the position's chart
[[nodiscard]] std::shared_ptr<const Geometry> CreateEmbeddedGeometry(
  std::shared_ptr<const Geometry> parameter_geometry,
  std::shared_ptr<const Function<Real>> position,
  EmbeddingOptions options = {});

/// @brief The space `space`, on the parameter geometry, as a space on the
///        embedded geometry built from it: same layout and values, evaluated
///        at BoundPoints of the embedded geometry.
///
/// Vectors and covectors whose basis is the parameter chart are evaluated in
/// the components of the embedded geometry's chart (the Pushforward of the
/// position's Jacobian: vectors by J, covectors by J^-T, which needs as many
/// parameters as coordinates). Only Value is supported, and the fields are
/// plain DOF holders data (the data of a SimpleFieldData), which is what
/// uniform-grid spaces use.
/// @throws pcms_error if `embedded` is not an embedded geometry of
///         `space`'s geometry
[[nodiscard]] std::shared_ptr<const FunctionSpace> CreateEmbeddedSpace(
  std::shared_ptr<const Geometry> embedded,
  std::shared_ptr<const FunctionSpace> space);

} // namespace pcms

#endif // PCMS_FIELD_EMBEDDING_HPP
