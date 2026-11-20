#ifndef PCMS_FIELD_DETAIL_POINT_EVALUATOR_HPP
#define PCMS_FIELD_DETAIL_POINT_EVALUATOR_HPP

#include <Kokkos_Core.hpp>
#include "pcms/utility/arrays.hpp"
#include "pcms/utility/coordinate_system.hpp"
#include "pcms/utility/detail/memory_spaces.hpp"
#include "pcms/utility/memory_spaces.hpp"
#include "pcms/utility/types.hpp"

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
///        orthogonal system into physical components: divides the partial
///        along axis i by the scale factor sqrt(g_ii). Does nothing where the
///        metric is constant or the basis is not orthogonal (the components
///        stay natural).
/// @param values gradient components [points][field components * dimension],
///        the derivative index last; the rows of `found` are scaled
/// @param system coordinate system of the partials
/// @param located [points][dimension] coordinates the partials were taken at
/// @param found indices of the rows to scale
template <typename T, typename LayoutPolicy>
void ScalePartialsToPhysical(
  Rank2View<T, DeviceMemorySpace, LayoutPolicy> values,
  const CoordinateSystem& system,
  Kokkos::View<const Real**, DeviceMemorySpace> located,
  Kokkos::View<const LO*, DeviceMemorySpace> found)
{
  if (system.IsConstantMetric() || !system.IsOrthogonal()) {
    return;
  }
  const auto metric = system.Bind(MakeConstRank2View(located))->Metric();
  const int dim = system.Dimension();
  const auto num_cols = static_cast<int>(values.extent(1));
  const auto scale = KOKKOS_LAMBDA(const LO k)
  {
    const LO p = found(k);
    for (int c = 0; c < num_cols; ++c) {
      const int i = c % dim;
      values(p, c) /= static_cast<T>(Kokkos::sqrt(metric(p, i, i)));
    }
  };
  detail::ForEach("PointEvaluator::ScalePartialsToPhysical",
                  static_cast<LO>(found.extent(0)), scale);
}

} // namespace pcms::detail

#endif // PCMS_FIELD_DETAIL_POINT_EVALUATOR_HPP
