#ifndef PCMS_FIELD_BOUND_SOURCE_HPP
#define PCMS_FIELD_BOUND_SOURCE_HPP

#include "pcms/field/coordinate_view.hpp"
#include "pcms/field/evaluation_request.h"
#include "pcms/field/field_layout.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/types.h"
#include <Kokkos_Core.hpp>
#include <memory>

namespace pcms
{

class PointEvaluatorFactory;

/// @brief Result of binding a source to a set of query points.
///
/// Holds the query points, which of them the source evaluates, and, in
/// subclasses, the source's per-point search results. It is shared by every
/// evaluator created from it; evaluators that share a bound source must not be
/// evaluated concurrently.
class BoundSource
{
public:
  /// @return number of bound query points
  [[nodiscard]] LO NumPoints() const noexcept { return num_points_; }

  /// @return out-of-bounds policy of the evaluators created from this source
  [[nodiscard]] const OutOfBoundsPolicy& Policy() const noexcept
  {
    return policy_;
  }

  /// @return layout whose DOF holders are the bound points, or null if the
  ///         source was bound to coordinates
  [[nodiscard]] const FieldLayout* QueryLayout() const noexcept
  {
    return query_layout_.get();
  }

  /// @return the bound query points (a copy owned by this object)
  [[nodiscard]] CoordinateView<DeviceMemorySpace> Points() const;

  /// @return the coordinates of the bound query points, one row per point
  [[nodiscard]] Kokkos::View<const Real**, DeviceMemorySpace> PointValues()
    const noexcept
  {
    return points_;
  }

  /// @return indices of the query points the source evaluates, ascending
  [[nodiscard]] Kokkos::View<const LO*, DeviceMemorySpace> FoundPoints()
    const noexcept
  {
    return found_;
  }

  /// @return indices of the query points the source does not evaluate,
  ///         ascending; their rows are filled in OutOfBoundsMode::FILL
  [[nodiscard]] Kokkos::View<const LO*, DeviceMemorySpace> MissingPoints()
    const noexcept
  {
    return missing_;
  }

  virtual ~BoundSource() noexcept = default;

  BoundSource(const BoundSource&) = delete;
  BoundSource& operator=(const BoundSource&) = delete;

protected:
  /// @brief Copies the request's points; every point counts as found until
  ///        SetFound is called.
  /// @param request bound query points and out-of-bounds policy
  explicit BoundSource(const EvaluationRequest& request);

  /// @brief Records which query points the source evaluates.
  /// @param found per query point, whether the source evaluates it
  /// @throws pcms_error in OutOfBoundsMode::ERROR if a point is not found
  void SetFound(Kokkos::View<const bool*, DeviceMemorySpace> found);

  /// @brief Takes the found and missing points of a bound source over the
  ///        same number of points.
  /// @param same_points bound source whose points correspond to these
  void SetFound(const BoundSource& same_points);

private:
  // set by PointEvaluatorFactory::Bind, checked by CreatePointEvaluator
  friend class PointEvaluatorFactory;
  const PointEvaluatorFactory* bound_by_ = nullptr;

  LO num_points_;
  OutOfBoundsPolicy policy_;
  std::shared_ptr<const FieldLayout> query_layout_;
  std::shared_ptr<const CoordinateSystem> system_;
  Kokkos::View<Real**, DeviceMemorySpace> points_;
  Kokkos::View<LO*, DeviceMemorySpace> found_;
  Kokkos::View<LO*, DeviceMemorySpace> missing_;
};

} // namespace pcms

#endif // PCMS_FIELD_BOUND_SOURCE_HPP
