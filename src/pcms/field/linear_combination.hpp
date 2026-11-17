#ifndef PCMS_FIELD_LINEAR_COMBINATION_HPP
#define PCMS_FIELD_LINEAR_COMBINATION_HPP

#include "pcms/field/differential_operator.hpp"
#include "pcms/field/field.h"
#include "pcms/field/field_data.h"
#include "pcms/field/function_space.h"
#include "pcms/field/layout/empty.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/field/point_evaluator_factory.hpp"
#include "pcms/field/value_view.hpp"
#include "pcms/geometry/bound_source.hpp"
#include "pcms/utility/arrays.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/types.h"
#include <Kokkos_Core.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace pcms
{

/// @brief One term of a LinearCombination: `coefficient` times `function`.
///
/// The term refers to `function`, which must outlive the combination and stay
/// in place (not be moved from); values written to it later are seen by the
/// combination.
template <typename T>
struct Term
{
  T coefficient;
  const Function<T>& function;
};

namespace detail
{

/// @brief values(r, :) *= a for the rows r listed in `rows`.
template <typename T, typename LayoutPolicy>
void ScaleRows(Rank2View<T, DeviceMemorySpace, LayoutPolicy> values,
               Kokkos::View<const LO*, DeviceMemorySpace> rows, T a)
{
  const auto num_cols = static_cast<int>(values.extent(1));
  const auto scale = KOKKOS_LAMBDA(const LO k)
  {
    for (int c = 0; c < num_cols; ++c) {
      values(rows(k), c) *= a;
    }
  };
  ForEach("LinearCombination::ScaleRows", static_cast<LO>(rows.extent(0)),
          scale);
}

/// @brief values(r, :) += a * term(r, :) for the rows r listed in `rows`.
template <typename T, typename LayoutPolicy>
void AddScaledRows(Rank2View<T, DeviceMemorySpace, LayoutPolicy> values,
                   Rank2View<const T, DeviceMemorySpace, LayoutPolicy> term,
                   Kokkos::View<const LO*, DeviceMemorySpace> rows, T a)
{
  const auto num_cols = static_cast<int>(values.extent(1));
  const auto axpy = KOKKOS_LAMBDA(const LO k)
  {
    const LO r = rows(k);
    for (int c = 0; c < num_cols; ++c) {
      values(r, c) += a * term(r, c);
    }
  };
  ForEach("LinearCombination::AddScaledRows", static_cast<LO>(rows.extent(0)),
          axpy);
}

/// @brief The data of a LinearCombination's function: the coefficients and
///        the term functions. Its values live in the terms (no DOFAccess).
template <typename T>
class LinearCombinationData final : public FieldData<T>
{
public:
  LinearCombinationData(ValueBasis basis, std::vector<Term<T>> terms)
    : basis_(std::move(basis)), terms_(std::move(terms))
  {
  }

  FieldValueType GetValueType() const override
  {
    return ValueTypeOfRank(basis_.Rank());
  }
  const ValueBasis& GetValueBasis() const override { return basis_; }

  [[nodiscard]] const std::vector<Term<T>>& Terms() const noexcept
  {
    return terms_;
  }

private:
  ValueBasis basis_;
  std::vector<Term<T>> terms_;
};

/// @brief Evaluates a LinearCombination: each term's evaluator at the same
///        bound source, combined over the found rows.
template <typename T>
class LinearCombinationEvaluator final : public PointEvaluator<T>
{
public:
  using Base = PointEvaluator<T>;
  using Values = ValueView<T, DeviceMemorySpace>;

  /// @param terms one evaluator per term, in the order of the terms
  LinearCombinationEvaluator(std::shared_ptr<const BoundSource> bound,
                             DifferentialOperator op,
                             std::shared_ptr<const FieldLayout> layout,
                             std::vector<std::unique_ptr<Base>> terms)
    : Base(std::move(bound), op, std::move(layout)), terms_(std::move(terms))
  {
  }

  /// @return the basis the first term's evaluator writes; the terms share
  ///         their stored basis and so the written one
  [[nodiscard]] ValueBasis OutputBasis(const ValueBasis& stored) const override
  {
    return terms_.front()->OutputBasis(stored);
  }

protected:
  void EvaluateImpl(const Field<T>& field, Values values) const override
  {
    const auto* data =
      dynamic_cast<const LinearCombinationData<T>*>(&field.GetData());
    PCMS_ALWAYS_ASSERT(data != nullptr); // the layout check passed
    const auto& terms = data->Terms();
    PCMS_ALWAYS_ASSERT(terms.size() == terms_.size());
    const auto rows = this->Bound().FoundPoints();
    const auto out = values.GetValues();
    terms_.front()->Evaluate(terms.front().function, values);
    if (terms.front().coefficient != T{1}) {
      ScaleRows(out, rows, terms.front().coefficient);
    }
    if (terms_.size() == 1) {
      return;
    }
    if (scratch_.extent(0) != out.extent(0) ||
        scratch_.extent(1) != out.extent(1)) {
      scratch_ = Kokkos::View<T**, DeviceMemorySpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "LinearCombination::scratch"),
        out.extent(0), out.extent(1));
    }
    const auto term_values = MakeRank2View(scratch_);
    for (size_t i = 1; i < terms_.size(); ++i) {
      terms_[i]->Evaluate(terms[i].function,
                          Values(values.GetBasis(), term_values));
      AddScaledRows(out, MakeConstRank2View(scratch_), rows,
                    terms[i].coefficient);
    }
  }

private:
  std::vector<std::unique_ptr<Base>> terms_;
  // the values of one term at a time, reused across Evaluate calls
  mutable Kokkos::View<T**, DeviceMemorySpace> scratch_;
};

/// @brief Evaluator source of a LinearCombination: it creates the terms'
///        evaluators at the bound source it is given and combines them.
template <typename T>
class LinearCombinationSource final : public PointEvaluatorFactory
{
public:
  /// @param spaces the terms' spaces, in the order of the terms; on one
  ///        geometry
  /// @param layout the layout of the combination's function
  LinearCombinationSource(
    std::vector<std::shared_ptr<const FunctionSpace>> spaces,
    std::shared_ptr<const FieldLayout> layout)
    : spaces_(std::move(spaces)), layout_(std::move(layout))
  {
  }

  [[nodiscard]] std::shared_ptr<const Geometry> GetGeometry()
    const noexcept override
  {
    return spaces_.front()->GetGeometry();
  }

  /// @return whether every term's space supports `op`
  [[nodiscard]] bool Supports(DifferentialOperator op) const noexcept override
  {
    for (const auto& space : spaces_) {
      if (!space->Supports(op)) {
        return false;
      }
    }
    return true;
  }

protected:
  PointEvaluatorVariant CreatePointEvaluatorImpl(
    Type value_type, std::shared_ptr<const BoundSource> bound,
    DifferentialOperator op) const override
  {
    if (value_type != TypeEnumFromType<T>()) {
      throw pcms_error("CreatePointEvaluator: a LinearCombination evaluates "
                       "fields of its own value type only");
    }
    std::vector<std::unique_ptr<PointEvaluator<T>>> terms;
    terms.reserve(spaces_.size());
    for (const auto& space : spaces_) {
      terms.push_back(CreateUntransformedPointEvaluator<T>(*space, bound, op));
    }
    return std::unique_ptr<PointEvaluator<T>>(
      std::make_unique<LinearCombinationEvaluator<T>>(
        std::move(bound), op, layout_, std::move(terms)));
  }

private:
  std::vector<std::shared_ptr<const FunctionSpace>> spaces_;
  std::shared_ptr<const FieldLayout> layout_;
};

} // namespace detail

/// @brief The function sum_i terms[i].coefficient * terms[i].function, for
///        evaluation.
///
/// The combination is a Function on a space of its own, on the terms'
/// geometry; it evaluates wherever its terms do (QueryPoints, an
/// Interpolator from its space, CreatePointEvaluator). An operator is
/// supported when every term's space supports it, and is evaluated as the
/// coefficient-weighted sum of the terms' results at the same bound
/// source, so the points are located once for all terms. The combination
/// keeps what its terms share: the geometry, the value basis (so the
/// basis its evaluators write) and the number of components. Its values live
/// in the terms: it has no DOFAccess, and the terms must outlive it and not
/// be moved from (see Term).
///
/// Not covered, deliberately: terms on different geometries (transfer them
/// onto one space first, then combine there), products of fields, functions
/// of a field's value, and coordinate-dependent factors (e.g. phi-dependent
/// ones); these stay user code.
///
/// @param name the combination's name
/// @param terms at least one term
/// @return the combination, a Function whose data refers to the terms
/// @throws pcms_error if `terms` is empty, or the terms' spaces are not on
///         one geometry (the same Geometry object), or the terms' value
///         bases (semantics) or numbers of components differ
template <typename T>
[[nodiscard]] Function<T> LinearCombination(std::string name,
                                            std::vector<Term<T>> terms)
{
  static_assert(is_supported_field_type_v<T>,
                "T is not a supported field type");
  if (terms.empty()) {
    throw pcms_error("LinearCombination: at least one term is required");
  }
  const Function<T>& first = terms.front().function;
  const auto geometry = first.GetSpace().GetGeometry();
  if (geometry == nullptr) {
    throw pcms_error("LinearCombination: the terms' spaces do not evaluate "
                     "at points (they have no geometry)");
  }
  const ValueBasis& basis = first.GetData().GetValueBasis();
  const int num_components = first.GetLayout().GetNumComponents();
  // TODO: check the units once the field metadata carries them
  std::vector<std::shared_ptr<const FunctionSpace>> spaces;
  spaces.reserve(terms.size());
  for (size_t i = 0; i < terms.size(); ++i) {
    const Function<T>& f = terms[i].function;
    const std::string which = "LinearCombination: term " + std::to_string(i) +
                              " ('" + f.GetName() + "') ";
    if (f.GetSpace().GetGeometry().get() != geometry.get()) {
      throw pcms_error(which +
                       "is on another geometry than term 0; terms must share "
                       "one Geometry object (transfer onto one space first)");
    }
    if (f.GetLayout().GetNumComponents() != num_components) {
      throw pcms_error(
        which + "has " + std::to_string(f.GetLayout().GetNumComponents()) +
        " components but term 0 has " + std::to_string(num_components));
    }
    if (!SameValueBasis(f.GetData().GetValueBasis(), basis)) {
      throw pcms_error(which +
                       "has another value basis (type, variance, coordinate "
                       "system or scaling) than term 0");
    }
    spaces.push_back(f.GetSpacePtr());
  }
  auto layout = std::make_shared<EmptyFieldLayout>(
    geometry->GetCoordinateSystem(), num_components);
  auto space = std::make_shared<FunctionSpace>(
    layout,
    [](Type, ValueBasis) -> FieldDataVariant {
      throw pcms_error("LinearCombination: the space of a combination has no "
                       "fields of its own; call LinearCombination");
    },
    std::make_shared<detail::LinearCombinationSource<T>>(std::move(spaces),
                                                         layout));
  return space->template CreateFunction<T>(
    std::move(name), std::make_unique<detail::LinearCombinationData<T>>(
                       basis, std::move(terms)));
}

} // namespace pcms

#endif // PCMS_FIELD_LINEAR_COMBINATION_HPP
