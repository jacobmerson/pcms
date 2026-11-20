#ifndef PCMS_GEOMETRY_PRODUCT_GEOMETRY_HPP
#define PCMS_GEOMETRY_PRODUCT_GEOMETRY_HPP

#include "pcms/geometry/geometry.hpp"
#include <memory>

namespace pcms
{

/// @brief The product of two geometries: points (a, b) with a in `a` and b in
///        `b`, in the Cartesian system of the sum of their dimensions, the
///        coordinates of `a` first (nest products for more factors).
///
/// Bind splits the query coordinates by factor dimension and binds each
/// factor; its BoundPoints keep the factor BoundPoints (ProductBinding in
/// the library's private header) for the spaces of the product, which
/// evaluate factor by factor. A point is found when it is found in both
/// factors. CreateQuadrature is the tensor product of the factors' rules, and
/// the differential is block-diagonal. The product holds no partitioning: it
/// is serial, its factors are not distributed.
/// @throws pcms_error if a geometry is null, a factor's system is not
///         Cartesian, or the dimensions add up to more than 3
[[nodiscard]] std::shared_ptr<const Geometry> CreateProductGeometry(
  std::shared_ptr<const Geometry> a, std::shared_ptr<const Geometry> b);

} // namespace pcms

#endif // PCMS_GEOMETRY_PRODUCT_GEOMETRY_HPP
