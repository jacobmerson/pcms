#ifndef PCMS_FIELD_INTERPOLATOR_H
#define PCMS_FIELD_INTERPOLATOR_H

#include "pcms/field/field.h"
#include "pcms/field/field_data.h"
#include "pcms/field/function_space.h"
#include "pcms/field/point_evaluator_factory.hpp"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/profile.h"
#include "pcms/utility/types.h"
#include <Kokkos_Core.hpp>
#include <memory>
#include "pcms/transfer/transfer_operator.hpp"

namespace pcms
{

namespace detail
{

/// Whether two layouts hold their DOFs at the same coordinates, in the same
/// order.
bool SameDOFHolderSites(const FieldLayout& a, const FieldLayout& b);

} // namespace detail

// Interpolator<T> separates the expensive localization step from the cheap
// repeated evaluation step, making it efficient to use in a coupling loop.
//
// Construct once per source×target function-space pair (localizes target DOF
// coordinates into the source mesh), then call Apply repeatedly for different
// field states at zero additional localization cost. Any Field sharing the
// same target FunctionSpace can be passed to Apply.
//
// Usage:
//   Interpolator<Real> interp(src_space, tgt_space);
//   interp.Apply(src_field, tgt_field);   // cheap; called in coupling loop
//   interp.Apply(src_field_next, tgt_field_next); // reuses cached localization
//
// Interpolators that evaluate the same source at the same target, e.g. a
// field and its gradient, can share one localization:
//   auto bound = src_space->Bind(*tgt_space);
//   Interpolator<Real> values(*src_space, *tgt_space, bound);
//   Interpolator<Real> gradient(*Gradient(src_space), *grad_space, bound);
//
// The target's declared basis must be the basis the source's evaluator writes
// for the source field (OutputBasis), so a scalar source interpolates its
// gradient onto a covector target.
template <typename T>
class Interpolator : public TransferOperator<T>
{
public:
  // Expensive: localizes target DOF coords into source mesh. Called once.
  /// @param source factory evaluated at the target's DOF-holder coordinates
  /// @param target_space space whose DOF holders receive the values
  /// @param policy out-of-bounds policy for the evaluation
  Interpolator(const PointEvaluatorFactory& source,
               const FunctionSpace& target_space, OutOfBoundsPolicy policy = {})
    : Interpolator(source, target_space, source.Bind(target_space, policy))
  {
  }

  /// Cheap: reuses a localization of the target's DOF holders.
  /// @param source factory evaluated at the target's DOF-holder coordinates
  /// @param target_space space whose DOF holders receive the values
  /// @param bound result of binding `source` (or the source it binds through)
  ///        to `target_space`, or to a space with the same DOF-holder sites
  Interpolator(const PointEvaluatorFactory& source,
               const FunctionSpace& target_space,
               std::shared_ptr<const BoundSource> bound)
    : num_points_(static_cast<LO>(
        target_space.GetLayout()->GetDOFHolderCoordinates().GetValues().extent(
          0))),
      n_comp_(target_space.GetLayout()->GetNumComponents()),
      evaluator_(CreateEvaluator(source, target_space, std::move(bound))),
      target_values_("interp_output", num_points_, n_comp_)
  {
  }

  void Apply(const Field<T>& source, Field<T>& target) const override
  {
    PCMS_FUNCTION_TIMER;
    const auto& sd = source.GetData();
    const auto& td = target.GetData();
    const ValueBasis written = evaluator_->OutputBasis(sd.GetValueBasis());
    if (!SameValueBasis(written, td.GetValueBasis())) {
      throw pcms_error(
        "Interpolator: the basis written for the source's stored basis "
        "differs from the target's declared basis");
    }
    evaluator_->Evaluate(source, ValueView<T, DeviceMemorySpace>(
                                   written, MakeRank2View(target_values_)));
    target.SetDOFHolderDataUnchecked(MakeConstRank2View(target_values_));
  }

private:
  static std::unique_ptr<PointEvaluator<T>> CreateEvaluator(
    const PointEvaluatorFactory& source, const FunctionSpace& target_space,
    std::shared_ptr<const BoundSource> bound)
  {
    if (bound == nullptr || bound->QueryLayout() == nullptr ||
        !detail::SameDOFHolderSites(*bound->QueryLayout(),
                                    *target_space.GetLayout())) {
      throw pcms_error("Interpolator: the bound source must be bound to a "
                       "space with the target's DOF-holder sites");
    }
    return source.CreatePointEvaluator<T>(std::move(bound));
  }

  LO num_points_;
  int n_comp_;
  std::unique_ptr<PointEvaluator<T>> evaluator_;
  mutable Kokkos::View<T**, DeviceMemorySpace> target_values_;
};

} // namespace pcms

#endif // PCMS_FIELD_INTERPOLATOR_H
