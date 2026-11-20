#ifndef PCMS_FIELD_DETAIL_POINT_EVALUATOR_HPP
#define PCMS_FIELD_DETAIL_POINT_EVALUATOR_HPP

#include <Kokkos_Core.hpp>
#include "pcms/utility/arrays.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/field/value_transform.hpp"
#include "pcms/field/value_view.hpp"
#include "pcms/geometry/bound_points.hpp"
#include "pcms/geometry/geometry.hpp"
#include "pcms/utility/coordinate_system.hpp"
#include "pcms/utility/detail/memory_spaces.hpp"
#include "pcms/utility/memory_spaces.hpp"
#include "pcms/utility/types.hpp"
#include <type_traits>

namespace pcms::detail
{

/// @brief Sets every column of the listed rows of `values` to `value`.
/// @param values output view
/// @param rows indices of the rows to set
/// @param value value to write
template <typename T, typename LayoutPolicy>
void FillRows(Rank2View<T, DeviceMemorySpace, LayoutPolicy> values,
              Kokkos::View<const LO*, DeviceMemorySpace> rows, T value)
{
  const auto num_cols = static_cast<int>(values.extent(1));
  const auto fill = KOKKOS_LAMBDA(const LO k)
  {
    for (int c = 0; c < num_cols; ++c) {
      values(rows(k), c) = value;
    }
  };
  detail::ForEach("PointEvaluator::FillRows", static_cast<LO>(rows.extent(0)),
                  fill);
}

/// @brief Turns the partial derivatives along the coordinates of an
///        orthogonal chart into the physical components its gradient basis
///        promises (detail::Scale: the partial along axis i divided by the
///        scale factor sqrt(g_ii) at the located points). Does nothing where
///        physical and natural components coincide or the components stay
///        natural.
/// @param values gradient components [points][field components * dimension],
///        the derivative index last, tagged with the gradient basis
/// @param bound the bound points the partials were taken at
/// @throws pcms_error if the values must be scaled and are not Real values of
///         the default layout
template <typename T, typename LayoutPolicy>
void ScalePartialsToPhysical(
  const ValueView<T, DeviceMemorySpace, LayoutPolicy>& values,
  const BoundPoints& bound)
{
  const auto& chart = bound.GetGeometry()->GetCoordinateSystem();
  const ValueBasis& physical = values.GetBasis();
  if (physical.scaling != ComponentScaling::Physical || !chart->HasMetric() ||
      chart->IsConstantMetric()) {
    return;
  }
  if constexpr (std::is_same_v<T, Real> &&
                std::is_same_v<LayoutPolicy, default_layout_for_memory_space_t<
                                               DeviceMemorySpace>>) {
    const ValueBasis natural{physical.system, ComponentScaling::Natural,
                             physical.variance, physical.parity};
    const auto scale =
      Scale(CoordinateView<DeviceMemorySpace>(
              chart, MakeConstRank2View(bound.LocatedPointValues())),
            ComponentScaling::Physical);
    scale.Apply(
      ValueView<const Real, DeviceMemorySpace>(natural, values.GetValues()),
      values);
  } else {
    throw pcms_error("PointEvaluator::Evaluate: physical gradient components "
                     "in a curvilinear chart need Real values");
  }
}

} // namespace pcms::detail

#endif // PCMS_FIELD_DETAIL_POINT_EVALUATOR_HPP
