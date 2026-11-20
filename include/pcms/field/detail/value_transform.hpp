// Part of pcms/field/value_transform.hpp, which includes it after the
// declarations it needs.
#include "pcms/field/value_transform.hpp"
#ifndef PCMS_FIELD_DETAIL_VALUE_TRANSFORM_HPP
#define PCMS_FIELD_DETAIL_VALUE_TRANSFORM_HPP

#include "pcms/utility/arrays.hpp"
#include "pcms/utility/coordinate_view.hpp"
#include "pcms/utility/memory_spaces.hpp"
#include "pcms/utility/types.hpp"
#include <Kokkos_Core.hpp>

namespace pcms::detail
{

/// A Jacobian at points, J(p, i, j) = d x_j / d xi_i for a map xi -> x; a
/// stride of 0 along the points means the same matrix at every point.
using JacobianView =
  View<3, const Real, DeviceMemorySpace, Kokkos::layout_stride>;

/// @brief The value transform of a map whose values flow against it: values
///        in the components of `from` (the map's target, where the points
///        were located) are expressed in those of `to` (the map's source,
///        the chart of the query points). Covectors by J^T, with no inverse
///        (out_i = sum_j J(p, i, j) in_j); vectors by J^-1 (the solution of
///        in_j = sum_i J(p, i, j) out_i), which needs J square.
/// @param jacobian J(p, i, j) = d from_j / d to_i, [points][dim to][dim from];
///        it must outlive the transform
/// @param from the map's target chart and the mapped points in it
/// @param to the map's source chart and the query points in it
/// @throws pcms_error if the extents of `jacobian` do not match the charts
[[nodiscard]] ValueTransform Pullback(
  JacobianView jacobian, const CoordinateView<DeviceMemorySpace>& from,
  const CoordinateView<DeviceMemorySpace>& to);

/// @brief The value transform along a map: components on the coordinates of
///        `from` (the parameter chart of a geometry) expressed in those of
///        `to` (its chart). Vectors by J (out_j = sum_i J(p, i, j) in_i, no
///        inverse), covectors by J^-T (the solution of
///        in_i = sum_j J(p, i, j) out_j), which needs J square.
/// @param jacobian J(p, i, j) = d to_j / d from_i, [points][dim from][dim to];
///        it must outlive the transform
/// @param from the parameter chart and the parameter points
/// @param to the chart and the points in it
/// @throws pcms_error if the extents of `jacobian` do not match the charts
[[nodiscard]] ValueTransform Pushforward(
  JacobianView jacobian, const CoordinateView<DeviceMemorySpace>& from,
  const CoordinateView<DeviceMemorySpace>& to);

/// @brief Natural to physical components of an orthogonal chart, or back: a
///        vector component is multiplied by the scale factor sqrt(g_ii) of
///        its axis, a covector component divided by it.
/// @param points the chart and the points the values are at
/// @param to the scaling of the output
[[nodiscard]] ValueTransform Scale(
  const CoordinateView<DeviceMemorySpace>& points, ComponentScaling to);

} // namespace pcms::detail

#endif // PCMS_FIELD_DETAIL_VALUE_TRANSFORM_HPP
