#ifndef PCMS_FIELD_POINT_EVALUATOR_HPP
#define PCMS_FIELD_POINT_EVALUATOR_HPP

#include "pcms/field/differential_operator.hpp"
#include "pcms/field/field_layout.hpp"
#include "pcms/geometry/bound_points.hpp"
#include "pcms/geometry/geometry.hpp"
#include "pcms/field/value_view.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/arrays.hpp"
#include "pcms/utility/memory_spaces.hpp"
#include "pcms/utility/types.hpp"
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include "pcms/field/detail/point_evaluator.hpp"
#include <memory>
#include <variant>

namespace pcms
{

template <typename T>
class Field;

/// @brief The evaluation at the points as a sparse matrix in CSR form: row p
///        holds the entries row_offsets(p) .. row_offsets(p + 1) - 1, whose
///        columns are the global ids of the DOF holders (the gids of the
///        layout) and whose values are the weights of their DOFs.
template <typename T>
struct PointMatrix
{
  Kokkos::View<const LO*, DeviceMemorySpace> row_offsets; // [points + 1]
  Kokkos::View<const GO*, DeviceMemorySpace> columns;     // [entries]
  Kokkos::View<const T*, DeviceMemorySpace> values;       // [entries]
};

/// @brief Evaluates fields at the points of a BoundPoints.
///
/// Created by PointEvaluatorFactory::CreatePointEvaluator. Evaluate writes
/// OutputBasis(stored basis of the field): the basis `OutputBasis` gives for
/// the evaluator's operator. The output has one row per query point and the
/// field's number of components times the coordinate dimension per added
/// index as columns. EvaluateImpl writes the rows of the
/// BoundPoints's FoundPoints(); in OutOfBoundsMode::FILL, Evaluate then sets
/// the rows of MissingPoints() to the fill value.
/// @tparam T field value type
/// @tparam LayoutPolicy layout of the output view
template <typename T,
          typename LayoutPolicy =
            detail::default_layout_for_memory_space_t<DeviceMemorySpace>>
class PointEvaluator
{
public:
  /// Writes the field's values at the query points into `values`.
  /// @param field a field of a compatible space; today that is the space
  ///        this evaluator was created from (a field of its layout), or any
  ///        field for an evaluator that accepts every layout
  /// @param values output, tagged with OutputBasis of the field's basis
  /// @throws std::invalid_argument if `values` is not shaped
  ///         [NumPoints()][number of output components]
  /// @throws pcms_error if `values` does not claim the basis this evaluator
  ///         writes for the field, or if the evaluator accepts the fields of
  ///         one layout and `field` is not one of them
  void Evaluate(const Field<T>& field,
                ValueView<T, DeviceMemorySpace, LayoutPolicy> values) const
  {
    if (accepted_layout_ != nullptr &&
        &field.GetLayout() != accepted_layout_.get()) {
      throw pcms_error("PointEvaluator::Evaluate: the field is not a field of "
                       "the function space this evaluator was created from");
    }
    CheckOutputShape(field, values);
    detail::CheckWrittenValueBasis(
      "PointEvaluator::Evaluate", values.GetBasis(),
      OutputBasis(field.GetData().GetValueBasis()));
    EvaluateImpl(field, values);
    if (op_ == DifferentialOperator::Gradient) {
      // the kinds write partials along the coordinates; the basis promises
      // physical components where the system's basis is orthogonal
      detail::ScalePartialsToPhysical(
        values.GetValues(), *bound_->GetGeometry()->GetCoordinateSystem(),
        bound_->LocatedPointValues(), bound_->FoundPoints());
    }
    const auto filled = FilledPoints();
    if (filled.extent(0) > 0) {
      detail::FillRows(values.GetValues(), filled,
                       static_cast<T>(bound_->Policy().fill_value));
    }
  }

  /// @return whether EvaluateTranspose is implemented for this evaluator
  [[nodiscard]] virtual bool SupportsTranspose() const noexcept
  {
    return false;
  }

  /// @brief Adds the transpose of the evaluation to a functional:
  ///        functional += E^T values, where E is the linear map from the DOFs
  ///        of a field of the evaluator's layout to its values at the points
  ///        (the rows of the FoundPoints(); the rows of missing points are
  ///        zero), so that <E x, y> = <x, E^T y>.
  ///
  /// Only for DifferentialOperator::Value. The functional is a plain Field of
  /// the evaluator's layout (not a Function: it is dual to the coefficients),
  /// accumulated into, never zeroed. The kind sums the contributions to DOF
  /// holders that other ranks own into their owners, so that the result is
  /// the same as on one rank (collective), and leaves the functional's data
  /// consistent. The weights of a quadrature are applied to `values` by the
  /// caller: b = E^T (W f).
  /// @param values one row per query point, one column per component of the
  ///        layout, tagged with the basis of the functional's data
  /// @param functional the field to accumulate into
  /// @throws pcms_error if SupportsTranspose() is false, the operator is not
  ///         Value, `functional` is not of the layout this evaluator accepts
  ///         or `values` claims another basis than the functional's
  /// @throws std::invalid_argument if `values` is not shaped [NumPoints()]
  ///         [components]
  void EvaluateTranspose(
    ValueView<const T, DeviceMemorySpace, LayoutPolicy> values,
    Field<T>& functional) const
  {
    if (!SupportsTranspose() || op_ != DifferentialOperator::Value) {
      throw pcms_error("PointEvaluator::EvaluateTranspose: not supported by "
                       "this evaluator");
    }
    if (accepted_layout_ != nullptr &&
        &functional.GetLayout() != accepted_layout_.get()) {
      throw pcms_error("PointEvaluator::EvaluateTranspose: the functional is "
                       "not a field of the layout of this evaluator");
    }
    detail::CheckWrittenValueBasis("PointEvaluator::EvaluateTranspose",
                                   values.GetBasis(),
                                   functional.GetData().GetValueBasis());
    if (values.extent(0) != static_cast<size_t>(NumPoints()) ||
        values.extent(1) !=
          static_cast<size_t>(functional.GetLayout().GetNumComponents())) {
      throw std::invalid_argument(
        "PointEvaluator::EvaluateTranspose: values must be shaped [query "
        "points][components of the layout]");
    }
    EvaluateTransposeImpl(values.GetValues(), functional);
  }

  /// @return whether GetMatrix is implemented for this evaluator
  [[nodiscard]] virtual bool SupportsMatrix() const noexcept { return false; }

  /// @brief The evaluation as a sparse matrix E: the row of a query point
  ///        holds the weights of the DOF holders its value is a combination
  ///        of; the same for every component, so that component c of the
  ///        values is E times component c of the DOFs.
  ///
  /// For DifferentialOperator::Value; an evaluator of the Gradient operator
  /// that supports the matrix has the matrix of the partials, whose row
  /// p * d + j is the partial j of point p (d the output columns of the
  /// operator), which the product spaces combine. The rows of missing points
  /// are empty or zero; a row may hold explicit zero entries.
  /// @throws pcms_error if SupportsMatrix() is false
  [[nodiscard]] PointMatrix<T> GetMatrix() const
  {
    if (!SupportsMatrix()) {
      throw pcms_error("PointEvaluator::GetMatrix: not supported by this "
                       "evaluator");
    }
    return GetMatrixImpl();
  }

  /// @return number of query points
  [[nodiscard]] LO NumPoints() const noexcept { return bound_->NumPoints(); }

  /// @param stored value basis of the evaluated field
  /// @return value basis this evaluator writes for `stored`
  /// @throws pcms_error if the operator is not supported for `stored`
  [[nodiscard]] virtual ValueBasis OutputBasis(const ValueBasis& stored) const
  {
    switch (op_) {
      case DifferentialOperator::Value: return stored;
      case DifferentialOperator::Gradient:
        // the partials are taken where the points were located, in the
        // geometry's system; a bound map re-expresses them afterwards
        return detail::GradientBasis(
          stored, bound_->GetGeometry()->GetCoordinateSystem());
      default:
        throw pcms_error(std::string("PointEvaluator::OutputBasis: ") +
                         std::string(ToString(op_)) + " is not implemented");
    }
  }

  /// @return the operator this evaluator applies
  [[nodiscard]] DifferentialOperator Operator() const noexcept { return op_; }

  /// @return the layout whose fields Evaluate accepts, or null for any
  [[nodiscard]] const std::shared_ptr<const FieldLayout>& AcceptedLayout()
    const noexcept
  {
    return accepted_layout_;
  }

  /// @return indices of the query points whose rows Evaluate sets to the fill
  ///         value; empty unless the policy is OutOfBoundsMode::FILL
  [[nodiscard]] Kokkos::View<const LO*, DeviceMemorySpace> FilledPoints()
    const noexcept
  {
    if (bound_->Policy().mode != OutOfBoundsMode::FILL) {
      return {};
    }
    return bound_->MissingPoints();
  }

  virtual ~PointEvaluator() noexcept = default;

protected:
  /// @param bound BoundPoints of the query points; not null
  /// @param op the operator this evaluator applies
  /// @param accepted_layout the layout whose fields EvaluateImpl can read, or
  ///        null to accept any field (layout-agnostic evaluators)
  explicit PointEvaluator(
    std::shared_ptr<const BoundPoints> bound,
    DifferentialOperator op = DifferentialOperator::Value,
    std::shared_ptr<const FieldLayout> accepted_layout = nullptr)
    : bound_(std::move(bound)),
      op_(op),
      accepted_layout_(std::move(accepted_layout))
  {
  }

  /// @return the BoundPoints of the query points
  [[nodiscard]] const BoundPoints& Bound() const noexcept { return *bound_; }

  /// @brief Writes the field's values at the BoundPoints's FoundPoints().
  /// @param field field to evaluate; its output shape and basis are checked
  /// @param values output view; rows of other points may be left unset
  virtual void EvaluateImpl(
    const Field<T>& field,
    ValueView<T, DeviceMemorySpace, LayoutPolicy> values) const = 0;

  /// @brief Adds E^T values to the functional; the layout of the functional,
  ///        the basis and the shape of `values` are checked.
  /// @throws pcms_error unless the evaluator overrides it
  virtual void EvaluateTransposeImpl(
    Rank2View<const T, DeviceMemorySpace, LayoutPolicy>, Field<T>&) const
  {
    throw pcms_error("PointEvaluator::EvaluateTranspose: not implemented by "
                     "this evaluator");
  }

  /// @throws pcms_error unless the evaluator overrides it
  virtual PointMatrix<T> GetMatrixImpl() const
  {
    throw pcms_error("PointEvaluator::GetMatrix: not implemented by this "
                     "evaluator");
  }

private:
  /// @brief Checks that `values` has one row per query point and one column
  ///        per component of the basis OutputBasis gives for the field.
  /// @throws std::invalid_argument otherwise
  void CheckOutputShape(
    const Field<T>& field,
    const ValueView<T, DeviceMemorySpace, LayoutPolicy>& values) const
  {
    const auto num_points = static_cast<size_t>(NumPoints());
    // the field's components times the dimension per index OutputBasis adds
    // for the evaluator's operator
    const ValueBasis stored = field.GetData().GetValueBasis();
    const ValueBasis written = OutputBasis(stored);
    auto num_components =
      static_cast<size_t>(field.GetLayout().GetNumComponents());
    for (int i = stored.Rank(); i < written.Rank(); ++i) {
      num_components *= static_cast<size_t>(written.system->Dimension());
    }
    if (values.extent(0) != num_points || values.extent(1) != num_components) {
      throw std::invalid_argument(
        "PointEvaluator::Evaluate: the output view is shaped [" +
        std::to_string(values.extent(0)) + "][" +
        std::to_string(values.extent(1)) + "] but must be [" +
        std::to_string(num_points) + "][" + std::to_string(num_components) +
        "] (query points x output components)");
    }
  }

  std::shared_ptr<const BoundPoints> bound_;
  DifferentialOperator op_;
  // layout of the fields Evaluate accepts; null accepts any field
  std::shared_ptr<const FieldLayout> accepted_layout_;
};

// Variant types using default layout for device memory space
using PointEvaluatorVariant =
  std::variant<std::unique_ptr<PointEvaluator<int8_t>>,
               std::unique_ptr<PointEvaluator<int32_t>>,
               std::unique_ptr<PointEvaluator<int64_t>>,
               std::unique_ptr<PointEvaluator<float>>,
               std::unique_ptr<PointEvaluator<double>>>;

} // namespace pcms

#endif // PCMS_FIELD_POINT_EVALUATOR_HPP
