#ifndef PCMS_FIELD_EMBEDDING_HPP
#define PCMS_FIELD_EMBEDDING_HPP

#include "pcms/field/coordinate_map.hpp"
#include "pcms/field/field.hpp"
#include "pcms/field/function_space.hpp"
#include "pcms/geometry/geometry.hpp"
#include "pcms/utility/coordinate_system.hpp"
#include <memory>
#include <optional>

namespace pcms
{

/// @brief The embedding of a parameter space given by a position Function:
///        the map from the parameter system to `target`, x(xi) = position(xi).
///
/// Bind evaluates the position and its Gradient at the parameter points (no
/// inverse). The source coordinate system of the map is a parameter system:
/// its metric is the induced one, g = J^T G J with J the Jacobian of the
/// position and G the metric of `target` at the image, and its volume factor
/// is the physical volume element of the image per unit of parameter volume.
/// The bound map re-expresses vectors (components transform with J) and
/// covectors (J^-1) of the parameter system in `target`; both need J square.
/// @param position Function of a space on the parameter geometry whose
///        evaluator supports Gradient: a vector (values::Vector) of the
///        geometry's coordinate system, whose components are the coordinates
///        of the image. (The gradient of a field with several components is
///        shaped [component][derivative] only for a vector, so the parameter
///        space and the target have the same dimension; the Jacobian code
///        handles fewer parameters than target coordinates.)
/// @param target the embedding coordinate system
/// @throws pcms_error if the position is null, is not a vector, its space has
///         no Gradient or the dimension of `target` is not the parameter
///         dimension
[[nodiscard]] std::shared_ptr<const CoordinateMap> CreateEmbedding(
  std::shared_ptr<const Function<Real>> position,
  std::shared_ptr<const CoordinateSystem> target);

/// @brief A translation symmetry of the domain of an embedded geometry: the
///        domain is invariant under adding `length` to coordinate `axis` of
///        the embedding's target system, and its cells cover one `length`
///        (e.g. a 1/zperiod toroidal wedge: axis phi, length 2 pi / zperiod).
struct TranslationSymmetry
{
  int axis = 0;
  Real length = 0;
};

/// @brief A geometry whose cells are those of a structured parameter geometry
///        and whose coordinate system is the embedding's target.
///
/// Bind of physical points inverts the embedding: the images of the cells
/// are binned by their bounding boxes, and Gauss-Newton (Newton for a square
/// Jacobian) finds the parameter coordinates in the candidate cells of a
/// point's bin. Periodic axes of the target system are wrapped. Bind also
/// accepts points in the parameter system, located directly (no search). The
/// BoundPoints carry the parameter coordinates, so the evaluators of spaces
/// on the parameter geometry evaluate unchanged (CreateEmbeddedSpace);
/// BoundPoints::Differential is the Jacobian of the embedding. The geometry
/// does not support OutOfBoundsMode::NEAREST_BOUNDARY, and a Rebind searches
/// anew.
/// @param parameter_geometry a uniform-grid geometry (the cells are boxes in
///        the parameter space)
/// @param embedding an embedding (CreateEmbedding) whose position is a
///        function on `parameter_geometry`
/// @param symmetry a symmetry of the domain: a query is reduced by it before
///        the search (as by a period of the target system), and the
///        BoundPoints keep the reduced coordinates as their located points.
///        The reduction replaces the period of the axis, so a symmetry must
///        divide it; none by default
/// @param options `divisions` is the number of search bins per axis (0
///        derives it from the cell count); `tolerance` the relative slack of
///        the point-in-cell test
/// @throws pcms_error if the geometry is not a uniform grid, the embedding
///         is not one of this geometry, or the symmetry has an axis outside
///         the target system or a non-positive length
[[nodiscard]] std::shared_ptr<const Geometry> CreateEmbeddedGeometry(
  std::shared_ptr<const Geometry> parameter_geometry,
  std::shared_ptr<const CoordinateMap> embedding, SearchOptions options = {},
  std::optional<TranslationSymmetry> symmetry = std::nullopt);

/// @brief The space `space`, on the parameter geometry, as a space on the
///        embedded geometry built from it: same layout and values, evaluated
///        at BoundPoints of the embedded geometry.
/// Only Value is supported, and the fields are plain DOF holders data (the
/// data of a SimpleFieldData), which is what uniform-grid spaces use.
/// @throws pcms_error if `embedded` is not an embedded geometry of
///         `space`'s geometry
[[nodiscard]] std::shared_ptr<const FunctionSpace> CreateEmbeddedSpace(
  std::shared_ptr<const FunctionSpace> space,
  std::shared_ptr<const Geometry> embedded);

} // namespace pcms

#endif // PCMS_FIELD_EMBEDDING_HPP
