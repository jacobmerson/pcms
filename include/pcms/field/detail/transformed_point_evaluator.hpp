#ifndef PCMS_FIELD_DETAIL_TRANSFORMED_POINT_EVALUATOR_HPP
#define PCMS_FIELD_DETAIL_TRANSFORMED_POINT_EVALUATOR_HPP

#include "pcms/field/field.hpp"
#include "pcms/field/field_data.hpp"
#include "pcms/field/point_evaluator.hpp"
#include "pcms/field/value_transform.hpp"
#include "pcms/field/value_view.hpp"
#include "pcms/geometry/bound_points.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/memory_spaces.hpp"
#include "pcms/utility/detail/arrays.hpp"
#include "pcms/utility/detail/profile.hpp"
#include <array>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace pcms::detail
{

/// The value transforms, in order, that re-express values in a basis.
using ValueTransforms =
  std::function<std::vector<const ValueTransform*>(const ValueBasis&)>;

/// Point evaluator that re-expresses the values of an inner evaluator through
/// the value transforms of their basis (those of a bind with a coordinate
/// map). Values without transforms are written by the inner evaluator
/// directly; others are transformed in place, through scratch storage only
/// where a transform changes the number of components.
/// PointEvaluatorFactory::CreatePointEvaluator creates it.
template <typename T>
class TransformedPointEvaluator final : public PointEvaluator<T>
{
public:
  /// @param bound BoundPoints of the query points
  /// @param inner evaluator created from `bound` that writes the values
  ///        before the transforms; not null
  /// @param transforms the transforms of the values of `inner`
  TransformedPointEvaluator(std::shared_ptr<const BoundPoints> bound,
                            std::unique_ptr<PointEvaluator<T>> inner,
                            ValueTransforms transforms)
    : PointEvaluator<T>(std::move(bound), inner->Operator(),
                        inner->AcceptedLayout()),
      inner_(std::move(inner)),
      transforms_(std::move(transforms))
  {
  }

  // Public because CUDA forbids extended lambdas in non-public member
  // functions. User should call Evaluate
  // TODO: add passkey or other mechanism to avoid user calling this
  void EvaluateImpl(const Field<T>& field,
                    ValueView<T, DeviceMemorySpace> values) const override
  {
    PCMS_FUNCTION_TIMER;
    const auto [bases, transforms] = Chain(field.GetData().GetValueBasis());
    if (transforms.empty()) {
      inner_->Evaluate(field, values);
      return;
    }
    if constexpr (std::is_same_v<T, Real>) {
      const auto out = values.GetValues();
      const int dout = bases.back().system->Dimension();
      Scratch scratch;
      // the output itself for as many components as it has
      const auto storage = [&](const ValueBasis& basis) {
        return basis.system->Dimension() == dout
                 ? out
                 : Storage(scratch, basis, out.extent(0), out.extent(1) / dout);
      };
      inner_->Evaluate(field, ValueView<Real, DeviceMemorySpace>(
                                bases.front(), storage(bases.front())));
      for (std::size_t k = 1; k < bases.size(); ++k) {
        transforms[k - 1]->Apply(
          ValueView<const Real, DeviceMemorySpace>(bases[k - 1],
                                                   storage(bases[k - 1])),
          ValueView<Real, DeviceMemorySpace>(bases[k], storage(bases[k])));
      }
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

  // Public for the same reason as EvaluateImpl.
  void EvaluateTransposeImpl(Rank2View<const T, DeviceMemorySpace> values,
                             Field<T>& functional) const override
  {
    PCMS_FUNCTION_TIMER;
    // the inner evaluator may count the points this evaluator fills as found
    Kokkos::View<T**, DeviceMemorySpace> masked;
    if (const auto filled = this->FilledPoints(); filled.extent(0) > 0) {
      masked = Kokkos::View<T**, DeviceMemorySpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "transpose_masked"),
        values.extent(0), values.extent(1));
      CopyDeviceRank2ViewToDeviceView(masked, values);
      FillRows(MakeRank2View(masked), filled, T{0});
      values = MakeConstRank2View(masked);
    }
    const auto [bases, transforms] =
      Chain(functional.GetData().GetValueBasis());
    if (transforms.empty()) {
      inner_->EvaluateTranspose(
        ValueView<const T, DeviceMemorySpace>(bases.front(), values),
        functional);
      return;
    }
    if constexpr (std::is_same_v<T, Real>) {
      // E^T T_1^T ... T_n^T
      const std::size_t blocks =
        values.extent(1) / bases.back().system->Dimension();
      Scratch scratch;
      Rank2View<const Real, DeviceMemorySpace> in = values;
      for (std::size_t k = bases.size() - 1; k > 0; --k) {
        const auto out =
          Storage(scratch, bases[k - 1], values.extent(0), blocks);
        transforms[k - 1]->ApplyTranspose(
          ValueView<const Real, DeviceMemorySpace>(bases[k], in),
          ValueView<Real, DeviceMemorySpace>(bases[k - 1], out));
        in = out;
      }
      inner_->EvaluateTranspose(
        ValueView<const Real, DeviceMemorySpace>(bases.front(), in),
        functional);
    } else {
      throw pcms_error("TransformedPointEvaluator::EvaluateTranspose: basis "
                       "transformations require T == Real");
    }
  }

  [[nodiscard]] ValueBasis OutputBasis(const ValueBasis& stored) const override
  {
    return Chain(stored).first.back();
  }

private:
  // scratch values by number of components per block (dimension <= 3)
  using Scratch = std::array<Kokkos::View<Real**, DeviceMemorySpace>, 4>;

  // the storage of values in `basis`, [points][blocks * dimension]: the
  // scratch of its dimension, allocated on first use
  static Rank2View<Real, DeviceMemorySpace> Storage(Scratch& scratch,
                                                    const ValueBasis& basis,
                                                    std::size_t points,
                                                    std::size_t blocks)
  {
    const int dim = basis.system->Dimension();
    auto& view = scratch.at(dim);
    if (view.extent(0) != points) {
      view = Kokkos::View<Real**, DeviceMemorySpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "transformed_values"),
        points, blocks * dim);
    }
    return MakeRank2View(view);
  }

  // the inner evaluator's basis for `stored`, then the output basis of each
  // of its transforms; and the transforms
  [[nodiscard]] std::pair<std::vector<ValueBasis>,
                          std::vector<const ValueTransform*>>
  Chain(const ValueBasis& stored) const
  {
    std::vector<ValueBasis> bases{inner_->OutputBasis(stored)};
    auto transforms = transforms_(bases.front());
    for (const auto* transform : transforms) {
      bases.push_back(transform->OutputBasis(bases.back()));
    }
    return {std::move(bases), std::move(transforms)};
  }

  std::unique_ptr<PointEvaluator<T>> inner_;
  ValueTransforms transforms_;
};

} // namespace pcms::detail

#endif // PCMS_FIELD_DETAIL_TRANSFORMED_POINT_EVALUATOR_HPP
