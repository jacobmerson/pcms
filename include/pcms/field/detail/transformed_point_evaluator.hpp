#ifndef PCMS_FIELD_DETAIL_TRANSFORMED_POINT_EVALUATOR_HPP
#define PCMS_FIELD_DETAIL_TRANSFORMED_POINT_EVALUATOR_HPP

#include "pcms/field/coordinate_map.hpp"
#include "pcms/field/field.hpp"
#include "pcms/field/field_data.hpp"
#include "pcms/field/point_evaluator.hpp"
#include "pcms/field/value_view.hpp"
#include "pcms/geometry/bound_points.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/memory_spaces.hpp"
#include "pcms/utility/detail/profile.hpp"
#include <memory>
#include <type_traits>
#include <utility>

namespace pcms::detail
{

/// Point evaluator at the points of a BoundPoints with a coordinate map: the
/// inner evaluator evaluates at the mapped points, and this one re-expresses
/// the evaluated components through the bound map's ValueTransform (scalars
/// pass through); its transpose applies the transpose of the value transform
/// and then that of the inner evaluator.
/// PointEvaluatorFactory::CreatePointEvaluator creates it for every bound
/// source whose GetBoundMap() is set.
template <typename T>
class TransformedPointEvaluator final : public PointEvaluator<T>
{
public:
  /// @param bound BoundPoints of the query points; GetBoundMap() not null
  /// @param inner evaluator created from `bound` that evaluates at the mapped
  ///        points without re-expressing the values; not null
  TransformedPointEvaluator(std::shared_ptr<const BoundPoints> bound,
                            std::unique_ptr<PointEvaluator<T>> inner)
    : PointEvaluator<T>(bound, inner->Operator(), inner->AcceptedLayout()),
      map_(bound->GetBoundMap()),
      inner_(std::move(inner))
  {
    PCMS_ALWAYS_ASSERT(map_ != nullptr);
  }

  // Public because CUDA forbids extended lambdas in non-public member
  // functions. User should call Evaluate
  // TODO: add passkey or other mechanism to avoid user calling this
  void EvaluateImpl(const Field<T>& field,
                    ValueView<T, DeviceMemorySpace> values) const override
  {
    PCMS_FUNCTION_TIMER;
    const ValueBasis inner_out =
      inner_->OutputBasis(field.GetData().GetValueBasis());
    if (inner_out.Rank() == 0) {
      inner_->Evaluate(field, values);
      return;
    }
    if constexpr (std::is_same_v<T, Real>) {
      const auto out = values.GetValues();
      const int din = inner_out.system->Dimension();
      const int dout = values.GetBasis().system->Dimension();
      // in place, unless the map changes the number of components
      Kokkos::View<Real**, DeviceMemorySpace> scratch;
      if (din != dout) {
        scratch = Kokkos::View<Real**, DeviceMemorySpace>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "map_source_values"),
          out.extent(0), static_cast<std::size_t>(out.extent(1) / dout) * din);
      }
      const ValueView<Real, DeviceMemorySpace> source_values(
        inner_out, din != dout ? MakeRank2View(scratch) : out);
      inner_->Evaluate(field, source_values);
      map_->ValueTransform().Apply(source_values, values);
    } else {
      throw pcms_error("TransformedPointEvaluator::Evaluate: basis "
                       "transformations require T == Real");
    }
  }

  /// @return whether the inner evaluator supports the transpose
  [[nodiscard]] bool SupportsTranspose() const noexcept override
  {
    return inner_->SupportsTranspose();
  }

  // Public because CUDA forbids extended lambdas in non-public member
  // functions. User should call EvaluateTranspose
  void EvaluateTransposeImpl(Rank2View<const T, DeviceMemorySpace> values,
                             Field<T>& functional) const override
  {
    PCMS_FUNCTION_TIMER;
    const ValueBasis inner_out =
      inner_->OutputBasis(functional.GetData().GetValueBasis());
    if (inner_out.Rank() == 0) {
      inner_->EvaluateTranspose(
        ValueView<const T, DeviceMemorySpace>(inner_out, values), functional);
      return;
    }
    if constexpr (std::is_same_v<T, Real>) {
      // E^T T^T: the transpose of the value transform first
      const auto& transform = map_->ValueTransform();
      const ValueBasis outer = transform.OutputBasis(inner_out);
      const int din = inner_out.system->Dimension();
      const int dout = outer.system->Dimension();
      Kokkos::View<Real**, DeviceMemorySpace> source(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "map_source_values"),
        values.extent(0),
        static_cast<std::size_t>(values.extent(1) / dout) * din);
      transform.ApplyTranspose(
        ValueView<const Real, DeviceMemorySpace>(outer, values),
        ValueView<Real, DeviceMemorySpace>(inner_out, MakeRank2View(source)));
      inner_->EvaluateTranspose(ValueView<const Real, DeviceMemorySpace>(
                                  inner_out, MakeConstRank2View(source)),
                                functional);
    } else {
      throw pcms_error("TransformedPointEvaluator::EvaluateTranspose: basis "
                       "transformations require T == Real");
    }
  }

  [[nodiscard]] ValueBasis OutputBasis(const ValueBasis& stored) const override
  {
    const ValueBasis inner_out = inner_->OutputBasis(stored);
    if (inner_out.Rank() == 0) {
      return inner_out;
    }
    return map_->ValueTransform().OutputBasis(inner_out);
  }

private:
  std::shared_ptr<const BoundCoordinateMap> map_;
  std::unique_ptr<PointEvaluator<T>> inner_;
};

} // namespace pcms::detail

#endif // PCMS_FIELD_DETAIL_TRANSFORMED_POINT_EVALUATOR_HPP
