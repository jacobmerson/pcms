#ifndef PCMS_EVALUATION_REQUEST_H
#define PCMS_EVALUATION_REQUEST_H

#include "pcms/field/coordinate_view.hpp"
#include "field_layout.h"
#include "out_of_bounds_policy.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/memory_spaces.h"

#include <memory>

namespace pcms
{

class FunctionSpace;

/// @brief Query points, out-of-bounds policy and point provenance passed to
///        PointEvaluatorFactory::BindImpl.
struct EvaluationRequest
{
  // Query coordinates to evaluate at. These are always consumed at evaluator
  // construction time.
  CoordinateView<DeviceMemorySpace> coords;
  // Optional provenance for the query sites. When present, construction-time
  // logic may use the layout and its discretization to select optimized
  // localization paths. Concrete PointEvaluator implementations are not
  // required to retain this layout after construction.
  std::shared_ptr<const FieldLayout> query_layout;
  // Construction-time policy that is baked into the created PointEvaluator.
  OutOfBoundsPolicy policy = {};

  /// @param coords_in query points
  /// @param query_layout_in layout whose DOF holders are the query points, or
  ///        null
  /// @param policy_in out-of-bounds policy
  explicit EvaluationRequest(CoordinateView<DeviceMemorySpace> coords_in,
                             std::shared_ptr<const FieldLayout> query_layout_in,
                             OutOfBoundsPolicy policy_in = {}) noexcept
    : coords(coords_in),
      query_layout(std::move(query_layout_in)),
      policy(policy_in)
  {
  }

  /// @brief Request for the DOF-holder coordinates of a layout.
  /// @param layout layout whose DOF holders are the query points
  /// @param policy_in out-of-bounds policy
  /// @throws pcms_error if `layout` is null
  explicit EvaluationRequest(std::shared_ptr<const FieldLayout> layout,
                             OutOfBoundsPolicy policy_in = {})
    : EvaluationRequest(DOFHolderPoints(layout), layout, policy_in)
  {
  }

  [[deprecated("bind the source with PointEvaluatorFactory::Bind(coords, "
               "policy)")]]
  static EvaluationRequest FromCoordinates(
    CoordinateView<DeviceMemorySpace> coords, OutOfBoundsPolicy policy = {});

  [[deprecated("bind the source to a space with "
               "PointEvaluatorFactory::Bind(space, policy)")]]
  static EvaluationRequest FromLayout(std::shared_ptr<const FieldLayout> layout,
                                      OutOfBoundsPolicy policy = {});

  [[deprecated("bind the source with PointEvaluatorFactory::Bind(space, "
               "policy)")]]
  static EvaluationRequest FromFunctionSpace(
    const FunctionSpace& function_space, OutOfBoundsPolicy policy = {});

  [[nodiscard]] const FieldLayout* GetQueryLayout() const noexcept
  {
    return query_layout.get();
  }

  [[nodiscard]] const Discretization* GetQueryDiscretization() const noexcept
  {
    auto* layout = GetQueryLayout();
    if (layout == nullptr) {
      return nullptr;
    }
    auto disc = layout->GetDiscretization();
    return disc.get();
  }

private:
  // The layout's owned DOF holders: the rows its fields' setters write.
  static CoordinateView<DeviceMemorySpace> DOFHolderPoints(
    const std::shared_ptr<const FieldLayout>& layout)
  {
    if (layout == nullptr) {
      throw pcms_error("EvaluationRequest: layout must not be null");
    }
    return layout->GetDOFHolderCoordinates();
  }
};

} // namespace pcms

#endif // PCMS_EVALUATION_REQUEST_H
