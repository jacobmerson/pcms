#ifndef PCMS_UNIFORM_GRID_SPLINE_EVALUATOR_FACTORY_H
#define PCMS_UNIFORM_GRID_SPLINE_EVALUATOR_FACTORY_H

#include "pcms/field/evaluator/spline_interpolator.hpp"
#include "pcms/field/evaluator/uniform_grid.h"
#include "pcms/field/field_data.h"
#include "pcms/field/point_evaluator_factory.hpp"
#include "pcms/field/layout/uniform_grid.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/types.h"

#include <memory>

namespace pcms
{

namespace detail
{

struct InitCoordsFunctor
{
  Kokkos::View<Real*, DeviceMemorySpace> coords;
  Real bot_left;
  Real delta;

  KOKKOS_FUNCTION void operator()(LO i) const
  {
    coords(i) = bot_left + delta * i;
  }
};

} // namespace detail

/// @brief Evaluates the bicubic spline of a field's grid values, or its
///        gradient (df/dx, df/dy) for derivative order 1.
template <typename LayoutPolicy =
            detail::default_layout_for_memory_space_t<DeviceMemorySpace>>
class UniformGridSplinePointEvaluator2D
  : public PointEvaluator<Real, LayoutPolicy>
{
public:
  /// @param layout order-1 single-component uniform-grid layout
  /// @param binding cells of the query points
  /// @param derivative_order 0 for values, 1 for the gradient
  UniformGridSplinePointEvaluator2D(
    std::shared_ptr<const UniformGridFieldLayout<2>> layout,
    std::shared_ptr<const UniformGridBinding<2>> binding,
    int derivative_order = 0)
    : PointEvaluator<Real, LayoutPolicy>(binding),
      layout_(std::move(layout)),
      grid_(layout_->GetGrid()),
      binding_(std::move(binding)),
      derivative_order_(derivative_order),
      x_coords_("uniform_grid_spline_x", grid_.divisions[0] + 1),
      y_coords_("uniform_grid_spline_y", grid_.divisions[1] + 1)
  {
    Real dx = grid_.edge_length[0] / grid_.divisions[0];
    Real dy = grid_.edge_length[1] / grid_.divisions[1];
    Real bot_left_x = grid_.bot_left[0];
    Real bot_left_y = grid_.bot_left[1];
    LO nx = grid_.divisions[0];
    LO ny = grid_.divisions[1];

    Kokkos::parallel_for(
      "init_x_coords",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, nx + 1),
      detail::InitCoordsFunctor{x_coords_, bot_left_x, dx});

    Kokkos::parallel_for(
      "init_y_coords",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, ny + 1),
      detail::InitCoordsFunctor{y_coords_, bot_left_y, dy});
  }

  [[nodiscard]] ValueBasis OutputBasis(const ValueBasis& stored) const override
  {
    if (derivative_order_ == 0) {
      return stored;
    }
    return FirstDerivativeBasis(stored, layout_->GetCoordinateSystem());
  }

  // Public because CUDA forbids extended lambdas in non-public member
  // functions. User call Evaluate.
  // TODO: add passkey or other mechanism to protect this
  void EvaluateImpl(
    const Field<Real>& field,
    ValueView<Real, DeviceMemorySpace, LayoutPolicy> out) const override
  {
    const auto values = out.GetValues();
    if (field.GetLayout().GetNumComponents() != 1) {
      throw pcms_error("UniformGridSplinePointEvaluator2D: only scalar "
                       "(num_components==1) fields are supported");
    }
    // One column per selected spline output: f, or (df/dx, df/dy).
    const int num_outputs = derivative_order_ == 0 ? 1 : 2;

    const auto dof_data = field.GetDOFHolderData().GetValues();
    LO nx = grid_.divisions[0] + 1;
    LO ny = grid_.divisions[1] + 1;
    PCMS_ALWAYS_ASSERT(static_cast<LO>(dof_data.size()) == nx * ny);

    Kokkos::View<Real*, DeviceMemorySpace> spline_values(
      "uniform_grid_spline_values", static_cast<size_t>(nx * ny));
    Kokkos::parallel_for(
      "copy_spline_values_device",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
      KOKKOS_LAMBDA(const LO iy, const LO ix) {
        spline_values(ix * ny + iy) = dof_data(iy * nx + ix, 0);
      });

    const auto found = binding_->FoundPoints();
    const auto num_in_bounds = static_cast<LO>(found.extent(0));
    if (num_in_bounds == 0) {
      return;
    }
    Kokkos::View<Real*, DeviceMemorySpace> eval_x("uniform_grid_spline_eval_x",
                                                  num_in_bounds);
    Kokkos::View<Real*, DeviceMemorySpace> eval_y("uniform_grid_spline_eval_y",
                                                  num_in_bounds);
    const auto points = binding_->Points().GetValues();
    Kokkos::parallel_for(
      "gather_found_points",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, num_in_bounds),
      KOKKOS_LAMBDA(const LO k) {
        eval_x(k) = points(found(k), 0);
        eval_y(k) = points(found(k), 1);
      });

    Kokkos::View<Real**, DeviceMemorySpace> spline_output(
      "uniform_grid_spline_output", num_in_bounds, num_outputs);
    Kokkos::View<Real*, DeviceMemorySpace> empty_bc(
      "uniform_grid_spline_empty_bc", 0);

    CompactBiCubicSplineInterpolator<Real, DeviceMemorySpace> interpolator(
      Rank1View<Real, DeviceMemorySpace>(x_coords_.data(), x_coords_.extent(0)),
      Rank1View<Real, DeviceMemorySpace>(y_coords_.data(), y_coords_.extent(0)),
      Rank1View<Real, DeviceMemorySpace>(spline_values.data(),
                                         spline_values.extent(0)),
      BoundaryCondition::NOT_A_KNOT,
      Rank1View<Real, DeviceMemorySpace>(empty_bc.data(), empty_bc.extent(0)),
      BoundaryCondition::NOT_A_KNOT,
      Rank1View<Real, DeviceMemorySpace>(empty_bc.data(), empty_bc.extent(0)),
      BoundaryCondition::NOT_A_KNOT,
      Rank1View<Real, DeviceMemorySpace>(empty_bc.data(), empty_bc.extent(0)),
      BoundaryCondition::NOT_A_KNOT,
      Rank1View<Real, DeviceMemorySpace>(empty_bc.data(), empty_bc.extent(0)));
    // The selector picks f, df/dx, df/dy, d2f/dx2, d2f/dy2, d2f/dxdy.
    Kokkos::View<LO*, DeviceMemorySpace> select("uniform_grid_spline_select",
                                                6);
    const int order = derivative_order_;
    Kokkos::parallel_for(
      "uniform_grid_spline_select",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, 6),
      KOKKOS_LAMBDA(LO k) {
        select(k) = order == 0 ? (k == 0) : (k == 1 || k == 2);
      });
    interpolator.evaluate(
      select,
      Rank1View<Real, DeviceMemorySpace>(eval_x.data(), eval_x.extent(0)),
      Rank1View<Real, DeviceMemorySpace>(eval_y.data(), eval_y.extent(0)),
      Rank2View<Real, DeviceMemorySpace>(spline_output.data(),
                                         spline_output.extent(0), num_outputs));

    Kokkos::parallel_for(
      "copy_in_bounds_values",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, num_in_bounds),
      KOKKOS_LAMBDA(LO k) {
        for (int c = 0; c < num_outputs; ++c) {
          values(found(k), c) = spline_output(k, c);
        }
      });
  }

private:
  std::shared_ptr<const UniformGridFieldLayout<2>> layout_;
  UniformGrid<2> grid_;
  std::shared_ptr<const UniformGridBinding<2>> binding_;
  int derivative_order_;
  Kokkos::View<Real*, DeviceMemorySpace> x_coords_;
  Kokkos::View<Real*, DeviceMemorySpace> y_coords_;
};

/// @brief Point evaluation of the bicubic spline of fields on a 2D uniform
///        grid, and of its gradient.
class UniformGridSplineEvaluatorFactory2D
  : public TypedPointEvaluatorFactory<Real, UniformGridBinding<2>>
{
  using Bound = UniformGridBinding<2>;

public:
  /// @param layout order-1 single-component uniform-grid layout
  explicit UniformGridSplineEvaluatorFactory2D(
    std::shared_ptr<const UniformGridFieldLayout<2>> layout)
    : TypedPointEvaluatorFactory<Real, Bound>(layout->GetCoordinateSystem()),
      layout_(std::move(layout)),
      grid_(layout_->GetGrid())
  {
  }

protected:
  std::shared_ptr<Bound> BindPoints(
    const EvaluationRequest& request) const override
  {
    if (layout_->GetOrder() != 1) {
      throw pcms_error(
        "UniformGridSplineEvaluatorFactory2D: spline evaluation requires an "
        "order-1 uniform-grid layout");
    }
    if (layout_->GetNumComponents() != 1) {
      throw pcms_error(
        "UniformGridSplineEvaluatorFactory2D: spline evaluation only supports "
        "single-component fields");
    }

    return std::make_shared<Bound>(request, grid_);
  }

  std::unique_ptr<PointEvaluator<Real>> CreateEvaluator(
    std::shared_ptr<const Bound> bound, int derivative_order) const override
  {
    if (derivative_order > 1) {
      throw pcms_error(
        "UniformGridSplineEvaluatorFactory2D: derivative order " +
        std::to_string(derivative_order) + " is not supported");
    }
    return std::make_unique<UniformGridSplinePointEvaluator2D<>>(
      layout_, std::move(bound), derivative_order);
  }

private:
  std::shared_ptr<const UniformGridFieldLayout<2>> layout_;
  UniformGrid<2> grid_;
};

} // namespace pcms

#endif // PCMS_UNIFORM_GRID_SPLINE_EVALUATOR_FACTORY_H
