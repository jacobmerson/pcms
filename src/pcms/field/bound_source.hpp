#ifndef PCMS_FIELD_BOUND_SOURCE_HPP
#define PCMS_FIELD_BOUND_SOURCE_HPP

#include "pcms/field/evaluation_request.h"
#include "pcms/field/field_layout.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/types.h"
#include <Kokkos_Core.hpp>
#include <memory>

namespace pcms
{

class PointEvaluatorFactory;

/// @brief Result of binding a source to a set of query points.
///
/// Holds the source's per-point search results, shared by every evaluator
/// created from it. Evaluators that share a bound source must not be evaluated
/// concurrently.
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

  virtual ~BoundSource() noexcept = default;

  BoundSource(const BoundSource&) = delete;
  BoundSource& operator=(const BoundSource&) = delete;

protected:
  /// @param request bound query points and out-of-bounds policy
  explicit BoundSource(const EvaluationRequest& request)
    : num_points_(static_cast<LO>(request.coords.GetValues().extent(0))),
      policy_(request.policy),
      query_layout_(request.query_layout)
  {
  }

private:
  // set by PointEvaluatorFactory::Bind, checked by CreatePointEvaluator
  friend class PointEvaluatorFactory;
  const PointEvaluatorFactory* bound_by_ = nullptr;

  LO num_points_;
  OutOfBoundsPolicy policy_;
  std::shared_ptr<const FieldLayout> query_layout_;
};

namespace detail
{

/// Bound source of a backend that searches when its evaluators are created:
/// it keeps the request, with its own copy of raw query points.
class DeferredBoundSource final : public BoundSource
{
public:
  explicit DeferredBoundSource(const EvaluationRequest& request);

  /// The request as bound, valid for this object's lifetime.
  [[nodiscard]] const EvaluationRequest& Request() const noexcept
  {
    return request_;
  }

private:
  Kokkos::View<Real**, DeviceMemorySpace> points_;
  EvaluationRequest request_;
};

/// @brief Casts a bound source to the type a backend creates.
/// @tparam Bound bound-source type of the backend
/// @param bound bound source to cast
/// @param context name used in the error message
/// @return `bound` as a `Bound`
/// @throws pcms_error if `bound` is not a `Bound`
template <typename Bound>
const Bound& BoundSourceAs(const BoundSource& bound, const char* context)
{
  const auto* typed = dynamic_cast<const Bound*>(&bound);
  if (typed == nullptr) {
    throw pcms_error(std::string(context) +
                     ": the bound source was not created by this backend");
  }
  return *typed;
}

} // namespace detail

} // namespace pcms

#endif // PCMS_FIELD_BOUND_SOURCE_HPP
