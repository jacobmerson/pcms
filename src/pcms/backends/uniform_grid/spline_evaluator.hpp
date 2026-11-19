#ifndef PCMS_BACKENDS_UNIFORM_GRID_SPLINE_EVALUATOR_HPP
#define PCMS_BACKENDS_UNIFORM_GRID_SPLINE_EVALUATOR_HPP

#include "pcms/backends/uniform_grid/spline_interpolator.hpp"
#include "pcms/backends/uniform_grid/uniform_grid_geometry.hpp"
#include "pcms/field/field_data.hpp"
#include "pcms/field/point_evaluator_factory.hpp"
#include "pcms/backends/uniform_grid/layout.hpp"
#include "pcms/field/point_evaluator.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/types.hpp"

#include <memory>

namespace pcms::uniform_grid
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

/// @brief Copies the coordinates of the listed points into one view per axis.
/// @param points coordinates, one row per point
/// @param rows indices of the points to copy
/// @param x x coordinates of the listed points
/// @param y y coordinates of the listed points
inline void GatherPoints(Kokkos::View<const Real**, DeviceMemorySpace> points,
                         Kokkos::View<const LO*, DeviceMemorySpace> rows,
                         Kokkos::View<Real*, DeviceMemorySpace> x,
                         Kokkos::View<Real*, DeviceMemorySpace> y)
{
  Kokkos::parallel_for(
    "gather_spline_points",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(
      0, static_cast<LO>(rows.extent(0))),
    KOKKOS_LAMBDA(const LO k) {
      x(k) = points(rows(k), 0);
      y(k) = points(rows(k), 1);
    });
}

/// @brief Copies the coordinates of the listed points of a 1D grid.
/// @param points coordinates, one row per point
/// @param rows indices of the points to copy
/// @param x coordinates of the listed points
inline void GatherPoints1D(Kokkos::View<const Real**, DeviceMemorySpace> points,
                           Kokkos::View<const LO*, DeviceMemorySpace> rows,
                           Kokkos::View<Real*, DeviceMemorySpace> x)
{
  Kokkos::parallel_for(
    "gather_spline_points_1d",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(
      0, static_cast<LO>(rows.extent(0))),
    KOKKOS_LAMBDA(const LO k) { x(k) = points(rows(k), 0); });
}

} // namespace detail

/// @brief Evaluates the bicubic spline of a field's grid values, or its
///        gradient (df/dx, df/dy) for DifferentialOperator::Gradient.
template <typename LayoutPolicy =
            pcms::detail::default_layout_for_memory_space_t<DeviceMemorySpace>>
class UniformGridSplinePointEvaluator2D
  : public PointEvaluator<Real, LayoutPolicy>
{
public:
  /// @param layout order-1 single-component uniform-grid layout
  /// @param binding cells of the query points
  /// @param op Value or Gradient
  UniformGridSplinePointEvaluator2D(
    std::shared_ptr<const UniformGridFieldLayout<2>> layout,
    std::shared_ptr<const pcms::detail::UniformGridBinding<2>> binding,
    DifferentialOperator op = DifferentialOperator::Value)
    : PointEvaluator<Real, LayoutPolicy>(binding, op, layout),
      grid_(layout->GetGrid()),
      x_coords_("uniform_grid_spline_x", grid_.divisions[0] + 1),
      y_coords_("uniform_grid_spline_y", grid_.divisions[1] + 1),
      found_(binding->FoundPoints()),
      eval_x_("uniform_grid_spline_eval_x", found_.extent(0)),
      eval_y_("uniform_grid_spline_eval_y", found_.extent(0)),
      spline_output_("uniform_grid_spline_output", found_.extent(0),
                     op == DifferentialOperator::Value ? 1 : 2),
      select_("uniform_grid_spline_select", 6),
      empty_bc_("uniform_grid_spline_empty_bc", 0)
  {
    Real dx = grid_.edge_length[0] / grid_.divisions[0];
    Real dy = grid_.edge_length[1] / grid_.divisions[1];
    Kokkos::parallel_for(
      "init_x_coords",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(
        0, grid_.divisions[0] + 1),
      detail::InitCoordsFunctor{x_coords_, grid_.bot_left[0], dx});
    Kokkos::parallel_for(
      "init_y_coords",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(
        0, grid_.divisions[1] + 1),
      detail::InitCoordsFunctor{y_coords_, grid_.bot_left[1], dy});
    detail::GatherPoints(binding->LocatedPointValues(), found_, eval_x_,
                         eval_y_);
    // The selector picks f, df/dx, df/dy, d2f/dx2, d2f/dy2, d2f/dxdy.
    auto select = Kokkos::create_mirror_view(select_);
    for (int k = 0; k < 6; ++k) {
      select(k) =
        op == DifferentialOperator::Value ? (k == 0) : (k == 1 || k == 2);
    }
    Kokkos::deep_copy(select_, select);
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

    const auto found = found_;
    const auto num_in_bounds = static_cast<LO>(found.extent(0));
    if (num_in_bounds == 0) {
      return;
    }
    const auto empty_bc =
      Rank1View<Real, DeviceMemorySpace>(empty_bc_.data(), 0);
    CompactBiCubicSplineInterpolator<Real, DeviceMemorySpace> interpolator(
      Rank1View<Real, DeviceMemorySpace>(x_coords_.data(), x_coords_.extent(0)),
      Rank1View<Real, DeviceMemorySpace>(y_coords_.data(), y_coords_.extent(0)),
      Rank1View<Real, DeviceMemorySpace>(spline_values.data(),
                                         spline_values.extent(0)),
      BoundaryCondition::NOT_A_KNOT, empty_bc, BoundaryCondition::NOT_A_KNOT,
      empty_bc, BoundaryCondition::NOT_A_KNOT, empty_bc,
      BoundaryCondition::NOT_A_KNOT, empty_bc);
    // one column per selected spline output: f, or (df/dx, df/dy)
    const auto spline_output = spline_output_;
    const auto num_outputs = static_cast<int>(spline_output.extent(1));
    interpolator.evaluate(
      select_,
      Rank1View<Real, DeviceMemorySpace>(eval_x_.data(), eval_x_.extent(0)),
      Rank1View<Real, DeviceMemorySpace>(eval_y_.data(), eval_y_.extent(0)),
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
  UniformGrid<2> grid_;
  Kokkos::View<Real*, DeviceMemorySpace> x_coords_;
  Kokkos::View<Real*, DeviceMemorySpace> y_coords_;
  // per bound point: found points' coordinates and spline output
  Kokkos::View<const LO*, DeviceMemorySpace> found_;
  Kokkos::View<Real*, DeviceMemorySpace> eval_x_;
  Kokkos::View<Real*, DeviceMemorySpace> eval_y_;
  Kokkos::View<Real**, DeviceMemorySpace> spline_output_;
  Kokkos::View<LO*, DeviceMemorySpace> select_;
  Kokkos::View<Real*, DeviceMemorySpace> empty_bc_;
};

/// @brief Evaluates the cubic spline of a field's grid values on a 1D uniform
///        grid, or its derivative df/dx for DifferentialOperator::Gradient.
///
/// The spline has not-a-knot boundary conditions at both ends, as the bicubic
/// spline of a 2D grid has; it reproduces any cubic exactly, up to the ends of
/// the grid, when the grid has at least four vertices.
template <typename LayoutPolicy =
            pcms::detail::default_layout_for_memory_space_t<DeviceMemorySpace>>
class UniformGridSplinePointEvaluator1D
  : public PointEvaluator<Real, LayoutPolicy>
{
public:
  /// @param layout order-1 single-component uniform-grid layout
  /// @param binding cells of the query points
  /// @param op Value or Gradient
  UniformGridSplinePointEvaluator1D(
    std::shared_ptr<const UniformGridFieldLayout<1>> layout,
    std::shared_ptr<const pcms::detail::UniformGridBinding<1>> binding,
    DifferentialOperator op = DifferentialOperator::Value)
    : PointEvaluator<Real, LayoutPolicy>(binding, op, layout),
      grid_(layout->GetGrid()),
      x_coords_("uniform_grid_spline_x", grid_.divisions[0] + 1),
      found_(binding->FoundPoints()),
      eval_x_("uniform_grid_spline_eval_x", found_.extent(0)),
      spline_output_("uniform_grid_spline_output", found_.extent(0), 1),
      select_("uniform_grid_spline_select", 3)
  {
    Real dx = grid_.edge_length[0] / grid_.divisions[0];
    Kokkos::parallel_for(
      "init_x_coords",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(
        0, grid_.divisions[0] + 1),
      detail::InitCoordsFunctor{x_coords_, grid_.bot_left[0], dx});
    detail::GatherPoints1D(binding->LocatedPointValues(), found_, eval_x_);
    // The selector picks f, df/dx, d2f/dx2.
    auto select = Kokkos::create_mirror_view(select_);
    for (int k = 0; k < 3; ++k) {
      select(k) = op == DifferentialOperator::Value ? (k == 0) : (k == 1);
    }
    Kokkos::deep_copy(select_, select);
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
      throw pcms_error("UniformGridSplinePointEvaluator1D: only scalar "
                       "(num_components==1) fields are supported");
    }
    const auto dof_data = field.GetDOFHolderData().GetValues();
    const LO nx = grid_.divisions[0] + 1;
    PCMS_ALWAYS_ASSERT(static_cast<LO>(dof_data.size()) == nx);

    Kokkos::View<Real*, DeviceMemorySpace> spline_values(
      "uniform_grid_spline_values", static_cast<size_t>(nx));
    Kokkos::parallel_for(
      "copy_spline_values_device",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, nx),
      KOKKOS_LAMBDA(const LO ix) { spline_values(ix) = dof_data(ix, 0); });

    const auto found = found_;
    const auto num_in_bounds = static_cast<LO>(found.extent(0));
    if (num_in_bounds == 0) {
      return;
    }
    ExplicitCubicSplineInterpolator<Real, DeviceMemorySpace> interpolator(
      Rank1View<Real, DeviceMemorySpace>(x_coords_.data(), x_coords_.extent(0)),
      Rank1View<Real, DeviceMemorySpace>(spline_values.data(),
                                         spline_values.extent(0)),
      BoundaryCondition::NOT_A_KNOT, 0, BoundaryCondition::NOT_A_KNOT, 0);
    const auto spline_output = spline_output_;
    interpolator.evaluate(
      select_,
      Rank1View<Real, DeviceMemorySpace>(eval_x_.data(), eval_x_.extent(0)),
      Rank2View<Real, DeviceMemorySpace>(spline_output.data(),
                                         spline_output.extent(0), 1));

    Kokkos::parallel_for(
      "copy_in_bounds_values",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, num_in_bounds),
      KOKKOS_LAMBDA(LO k) { values(found(k), 0) = spline_output(k, 0); });
  }

private:
  UniformGrid<1> grid_;
  Kokkos::View<Real*, DeviceMemorySpace> x_coords_;
  // per bound point: found points' coordinates and spline output
  Kokkos::View<const LO*, DeviceMemorySpace> found_;
  Kokkos::View<Real*, DeviceMemorySpace> eval_x_;
  Kokkos::View<Real**, DeviceMemorySpace> spline_output_;
  Kokkos::View<LO*, DeviceMemorySpace> select_;
};

/// @brief Point evaluation of the cubic (1D) or bicubic (2D) spline of fields
///        on a uniform grid, and of its gradient.
/// @tparam Dim grid dimension, 1 or 2
template <unsigned Dim>
class UniformGridSplineEvaluatorFactory
  : public pcms::detail::TypedPointEvaluatorFactory<
      Real, pcms::detail::UniformGridBinding<Dim>>
{
  static_assert(Dim == 1 || Dim == 2, "splines are on 1D and 2D grids");
  using Bound = pcms::detail::UniformGridBinding<Dim>;

public:
  /// @param geometry geometry of the layout's grid
  /// @param layout order-1 single-component uniform-grid layout
  /// @throws pcms_error if the layout is not order 1 with one component
  UniformGridSplineEvaluatorFactory(
    std::shared_ptr<const UniformGridGeometry<Dim>> geometry,
    std::shared_ptr<const UniformGridFieldLayout<Dim>> layout)
    : pcms::detail::TypedPointEvaluatorFactory<Real, Bound>(
        std::move(geometry), {DifferentialOperator::Gradient}),
      layout_(std::move(layout))
  {
    if (layout_->GetOrder() != 1) {
      throw pcms_error(
        "UniformGridSplineEvaluatorFactory: spline evaluation requires an "
        "order-1 uniform-grid layout");
    }
    if (layout_->GetNumComponents() != 1) {
      throw pcms_error(
        "UniformGridSplineEvaluatorFactory: spline evaluation only supports "
        "single-component fields");
    }
  }

protected:
  std::unique_ptr<PointEvaluator<Real>> CreateEvaluator(
    std::shared_ptr<const Bound> bound, DifferentialOperator op) const override
  {
    if constexpr (Dim == 1) {
      return std::make_unique<UniformGridSplinePointEvaluator1D<>>(
        layout_, std::move(bound), op);
    } else {
      return std::make_unique<UniformGridSplinePointEvaluator2D<>>(
        layout_, std::move(bound), op);
    }
  }

private:
  std::shared_ptr<const UniformGridFieldLayout<Dim>> layout_;
};

using UniformGridSplineEvaluatorFactory1D =
  UniformGridSplineEvaluatorFactory<1>;
using UniformGridSplineEvaluatorFactory2D =
  UniformGridSplineEvaluatorFactory<2>;

} // namespace pcms::uniform_grid

#endif // PCMS_BACKENDS_UNIFORM_GRID_SPLINE_EVALUATOR_HPP
