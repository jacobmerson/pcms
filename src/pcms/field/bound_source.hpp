#ifndef PCMS_FIELD_BOUND_SOURCE_HPP
#define PCMS_FIELD_BOUND_SOURCE_HPP

#include "pcms/field/evaluation_request.h"
#include "pcms/field/field_layout.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/utility/types.h"
#include <memory>
#include <string>
#include <utility>

namespace pcms
{

class PointEvaluatorFactory;

// The result of binding a source to one query point set: everything the source
// computes per point (its point search), owned and complete. It does not
// depend on the value type, the derivative order or the field, so every
// evaluator created from it at that point set shares the search. Backends
// subclass it; callers treat it as opaque.
//
// Evaluators sharing a bound source may share mutable per-point state, so they
// must not be evaluated concurrently.
class BoundSource
{
public:
  [[nodiscard]] LO NumPoints() const noexcept { return num_points_; }

  [[nodiscard]] const OutOfBoundsPolicy& Policy() const noexcept
  {
    return policy_;
  }

  /// Layout whose DOF holders are the bound points; null when bound to raw
  /// points.
  [[nodiscard]] const FieldLayout* QueryLayout() const noexcept
  {
    return query_layout_.get();
  }

  virtual ~BoundSource() noexcept = default;

  BoundSource(const BoundSource&) = delete;
  BoundSource& operator=(const BoundSource&) = delete;

protected:
  explicit BoundSource(const EvaluationRequest& request)
    : num_points_(static_cast<LO>(request.coords.GetValues().extent(0))),
      policy_(request.policy),
      query_layout_(request.query_layout)
  {
  }

private:
  // PointEvaluatorFactory::Bind records which factory bound this object, so
  // CreatePointEvaluator can reject one bound by another factory.
  friend class PointEvaluatorFactory;
  const PointEvaluatorFactory* bound_by_ = nullptr;

  LO num_points_;
  OutOfBoundsPolicy policy_;
  std::shared_ptr<const FieldLayout> query_layout_;
};

namespace detail
{

/// Bound source of a backend whose search result is one copyable value, such
/// as a struct of Kokkos views that evaluators copy.
template <typename Hint>
class HintBoundSource final : public BoundSource
{
public:
  HintBoundSource(const EvaluationRequest& request, Hint hint)
    : BoundSource(request), hint_(std::move(hint))
  {
  }

  [[nodiscard]] const Hint& GetHint() const noexcept { return hint_; }

private:
  Hint hint_;
};

/// Downcasts a bound source to the type the backend binds; throws if it is
/// another backend's.
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
