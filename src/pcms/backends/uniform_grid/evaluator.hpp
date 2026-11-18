#ifndef PCMS_UNIFORM_GRID_EVALUATOR_FACTORY_H
#define PCMS_UNIFORM_GRID_EVALUATOR_FACTORY_H

#include "pcms/backends/uniform_grid/geometry.hpp"
#include "pcms/backends/uniform_grid/layout.hpp"
#include "pcms/utility/types.hpp"
#include <Kokkos_Core.hpp>
#include "pcms/field/point_evaluator_factory.hpp"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/field/field_data.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/profile.h"

#include <memory>
#include <utility>
#include <Omega_h_vector.hpp>

namespace pcms
{

/// @brief Evaluates order-0 (cell value) or order-1 (multilinear) fields on a
///        uniform grid.
/// @tparam Dim grid dimension
/// @tparam LayoutPolicy layout of the output view
template <unsigned Dim = 2,
          typename LayoutPolicy =
            detail::default_layout_for_memory_space_t<DeviceMemorySpace>>
class UniformGridPointEvaluator : public PointEvaluator<Real, LayoutPolicy>
{
public:
  /// @param layout uniform-grid layout of the evaluated fields
  /// @param binding cells of the query points
  UniformGridPointEvaluator(
    std::shared_ptr<const UniformGridFieldLayout<Dim>> layout,
    std::shared_ptr<const UniformGridBinding<Dim>> binding)
    : PointEvaluator<Real, LayoutPolicy>(binding, DifferentialOperator::Value,
                                         layout),
      layout_(std::move(layout)),
      grid_(layout_->GetGrid()),
      binding_(std::move(binding))
  {
  }

  // Public because CUDA forbids extended lambdas in non-public member
  // functions. User call Evaluate.
  // TODO: add passkey or other mechanism to protect this
  void EvaluateImpl(
    const Field<Real>& field,
    ValueView<Real, DeviceMemorySpace, LayoutPolicy> out) const override
  {
    PCMS_FUNCTION_TIMER;
    const auto values = out.GetValues();
    const auto dof_data = field.GetDOFHolderData().GetValues();
    LO num_points = this->NumPoints();
    int n_comp = layout_->GetNumComponents();

    PCMS_ALWAYS_ASSERT(dof_data.size() ==
                       static_cast<size_t>(layout_->GetNumDofHolder() *
                                           layout_->GetNumComponents()));

    // Every point has a (closest) cell, so all rows are computed; Evaluate
    // overwrites the rows of points outside the grid in FILL mode.
    auto cell_indices = binding_->CellIndices();

    if (layout_->GetOrder() == 0) {
      ForEach(
        "evaluate_order0_device", num_points, KOKKOS_LAMBDA(const LO i) {
          for (int c = 0; c < n_comp; ++c) {
            values(i, c) = dof_data(cell_indices(i), c);
          }
        });
      return;
    }

    // Order-1: multilinear interpolation. Vertices are numbered with the first
    // axis fastest (UniformGridFieldLayout), while GetDimensionedIndex()
    // returns the cell indices in the opposite order of the coordinates, so
    // axis d is entry Dim - 1 - d.
    // Local copy to avoid capturing a reference in device lambdas in some
    // compilers
    auto grid = grid_;
    // the binding's parametric coordinates: the point's position in its
    // cell, per axis in [0, 1] (outside it for a NEAREST_BOUNDARY point)
    const auto parametric = binding_->Parametric();
    ForEach(
      "evaluate_order1_device", num_points, KOKKOS_LAMBDA(const LO i) {
        const LO cell = cell_indices(i);
        const auto cell_index = grid.GetDimensionedIndex(cell);
        LO base = 0;
        LO stride = 1;
        Real t[Dim];
        for (unsigned d = 0; d < Dim; ++d) {
          t[d] = parametric(i, d);
          base += cell_index[Dim - 1 - d] * stride;
          stride *= grid.divisions[d] + 1;
        }
        for (int c = 0; c < n_comp; ++c) {
          // Bit d of corner selects the upper vertex along axis d.
          Real value = 0;
          for (unsigned corner = 0; corner < (1u << Dim); ++corner) {
            Real weight = 1;
            LO vertex = base;
            LO axis_stride = 1;
            for (unsigned d = 0; d < Dim; ++d) {
              const bool upper = (corner >> d) & 1u;
              weight *= upper ? t[d] : 1 - t[d];
              vertex += upper ? axis_stride : 0;
              axis_stride *= grid.divisions[d] + 1;
            }
            value += weight * dof_data(vertex, c);
          }
          values(i, c) = value;
        }
      });
  }

private:
  std::shared_ptr<const UniformGridFieldLayout<Dim>> layout_;
  UniformGrid<Dim> grid_;
  std::shared_ptr<const UniformGridBinding<Dim>> binding_;
};

/// @brief Point evaluation of fields on a structured uniform grid.
/// @tparam Dim grid dimension
template <unsigned Dim = 2>
class UniformGridEvaluatorFactory
  : public TypedPointEvaluatorFactory<Real, UniformGridBinding<Dim>>
{
  using Bound = UniformGridBinding<Dim>;

public:
  /// @param geometry geometry of the layout's grid
  /// @param layout uniform-grid layout of the evaluated fields
  UniformGridEvaluatorFactory(
    std::shared_ptr<const UniformGridGeometry<Dim>> geometry,
    std::shared_ptr<const UniformGridFieldLayout<Dim>> layout)
    : TypedPointEvaluatorFactory<Real, Bound>(std::move(geometry)),
      layout_(std::move(layout))
  {
  }

protected:
  std::unique_ptr<PointEvaluator<Real>> CreateEvaluator(
    std::shared_ptr<const Bound> bound,
    DifferentialOperator /*op*/) const override
  {
    return std::make_unique<UniformGridPointEvaluator<Dim>>(layout_,
                                                            std::move(bound));
  }

private:
  std::shared_ptr<const UniformGridFieldLayout<Dim>> layout_;
};

using UniformGridEvaluatorFactory2D = UniformGridEvaluatorFactory<2>;

} // namespace pcms

#endif // PCMS_UNIFORM_GRID_EVALUATOR_FACTORY_H
