#ifndef PCMS_FIELD_EMBEDDING_HPP
#define PCMS_FIELD_EMBEDDING_HPP

#include "pcms/field/coordinate_map.hpp"
#include "pcms/field/field.hpp"
#include "pcms/field/function_space.hpp"
#include "pcms/geometry/geometry.hpp"
#include "pcms/utility/coordinate_system.hpp"
#include <memory>

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

} // namespace pcms

#endif // PCMS_FIELD_EMBEDDING_HPP
