// Part of pcms/field/periodic_axis.hpp, which includes it after the
// declarations it needs.
#include "pcms/field/periodic_axis.hpp"
#ifndef PCMS_FIELD_DETAIL_PERIODIC_AXIS_HPP
#define PCMS_FIELD_DETAIL_PERIODIC_AXIS_HPP

namespace pcms::detail
{

/// @brief The arguments of CreateTwistShiftTransform
struct TwistShiftArguments
{
  int shift_axis = 0;
  std::shared_ptr<const Function<Real>> shift;
  std::vector<int> input_axes;
};

/// @return the arguments `map` was created with, or null if it is not a
///         twist-shift map
[[nodiscard]] const TwistShiftArguments* GetTwistShiftArguments(
  const PeriodicTransform& map) noexcept;

} // namespace pcms::detail

#endif // PCMS_FIELD_DETAIL_PERIODIC_AXIS_HPP
