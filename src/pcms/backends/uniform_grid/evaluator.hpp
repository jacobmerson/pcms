#ifndef PCMS_UNIFORM_GRID_EVALUATOR_FACTORY_H
#define PCMS_UNIFORM_GRID_EVALUATOR_FACTORY_H

#include "pcms/backends/uniform_grid/layout.hpp"
#include "pcms/utility/types.hpp"
#include <Kokkos_Core.hpp>
#include "pcms/field/point_evaluator_factory.hpp"
#include "pcms/field/out_of_bounds_policy.hpp"
#include "pcms/field/point_evaluator.hpp"
#include "pcms/field/field_data.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/profile.hpp"
#include "pcms/backends/uniform_grid/linear_interpolant.hpp"
#include "pcms/backends/uniform_grid/multidimarray.hpp"

#include <memory>
#include <utility>

namespace pcms
{

namespace detail
{

/// @brief Locates points in a uniform grid.
/// @param grid grid to locate the points in
/// @param points query points, one row per point
/// @param nearest whether points outside the grid count as found (in their
///        closest cell)
/// @return per point, whether it is found, and its (closest) cell
template <unsigned Dim>
std::pair<Kokkos::View<bool*, DeviceMemorySpace>,
          Kokkos::View<LO*, DeviceMemorySpace>>
LocateInGrid(const UniformGrid<Dim>& grid,
             Kokkos::View<const Real**, DeviceMemorySpace> points, bool nearest)
{
  const auto n = static_cast<LO>(points.extent(0));
  Kokkos::View<bool*, DeviceMemorySpace> found("uniform_grid_found", n);
  Kokkos::View<LO*, DeviceMemorySpace> cells("uniform_grid_cell_indices", n);
  Kokkos::parallel_for(
    "LocateInGrid",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, n),
    KOKKOS_LAMBDA(const LO i) {
      Omega_h::Vector<Dim> point;
      for (unsigned d = 0; d < Dim; ++d) {
        point[d] = points(i, d);
      }
      found(i) = nearest || grid.IsPointInBounds(point);
      cells(i) = grid.ClosestCellID(point);
    });
  return {found, cells};
}

} // namespace detail

/// @brief Uniform-grid bound source: the cell of each query point.
///
/// Points outside the grid are not found, except with
/// OutOfBoundsMode::NEAREST_BOUNDARY, where they use the closest cell.
/// @tparam Dim grid dimension
template <unsigned Dim>
class UniformGridBinding final : public BoundSource
{
public:
  /// @param request query points and out-of-bounds policy
  /// @param grid grid to locate the points in
  UniformGridBinding(const BindRequest& request, const UniformGrid<Dim>& grid)
    : BoundSource(request)
  {
    auto [found, cells] = detail::LocateInGrid(
      grid, PointValues(), Policy().mode == OutOfBoundsMode::NEAREST_BOUNDARY);
    SetFound(found);
    cell_indices_ = cells;
  }

  /// @return the cell of each query point (the closest cell if outside)
  [[nodiscard]] Kokkos::View<const LO*, DeviceMemorySpace> CellIndices()
    const noexcept
  {
    return cell_indices_;
  }

private:
  Kokkos::View<LO*, DeviceMemorySpace> cell_indices_;
};

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
    : PointEvaluator<Real, LayoutPolicy>(binding),
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
    auto coordinates = binding_->PointValues();

    if (layout_->GetOrder() == 0) {
      Kokkos::parallel_for(
        "evaluate_order0_device",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_points, n_comp}),
        KOKKOS_LAMBDA(const LO i, const int c) {
          values(i, c) = dof_data(cell_indices(i), c);
        });
      return;
    }

    // Order-1: multilinear interpolation
    // Create local copy to avoid capturing reference in device lambdas in some
    // compilers
    auto grid_copy = grid_;
    Kokkos::View<LO**, DeviceMemorySpace> cell_dim_indices("cell_dim_indices",
                                                           num_points, Dim);
    Kokkos::parallel_for(
      "compute_cell_dim_indices",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, num_points),
      KOKKOS_CLASS_LAMBDA(const LO i) {
        auto dim_idx = grid_copy.GetDimensionedIndex(cell_indices(i));
        for (unsigned d = 0; d < Dim; ++d) {
          cell_dim_indices(i, d) = dim_idx[d];
        }
      });

    auto cell_divisions = grid_.divisions;
    IntVecView dimensions_view("dimensions", Dim);
    Kokkos::parallel_for(
      "set_dimensions_view",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, Dim),
      KOKKOS_LAMBDA(const unsigned d) {
        // cell_dim_indices uses GetDimensionedIndex(), which returns indices in
        // the opposite order of the coordinates, so we reverse the order here.
        dimensions_view(d) = cell_divisions[Dim - 1 - d] + 1;
      });

    RealMatView parametric_coords("parametric_coords", num_points, Dim);
    Kokkos::parallel_for(
      "compute_parametric_coords",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, num_points),
      KOKKOS_CLASS_LAMBDA(const LO i) {
        auto cell_bbox = grid_copy.GetCellBBOX(cell_indices(i));
        for (unsigned d = 0; d < Dim; ++d) {
          Real coord = coordinates(i, d);
          Real cell_min = cell_bbox.center[d] - cell_bbox.half_width[d];
          Real cell_max = cell_bbox.center[d] + cell_bbox.half_width[d];
          // cell_dim_indices is filled from GetDimensionedIndex(), which
          // returns indices in the opposite order of the coordinates
          parametric_coords(i, Dim - 1 - d) =
            (coord - cell_min) / (cell_max - cell_min);
        }
      });

    IntMatView cell_indices_interp("cell_indices_interp", num_points, Dim);
    Kokkos::parallel_for(
      "copy_cell_dim_indices_to_interp",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, num_points),
      KOKKOS_LAMBDA(const LO i) {
        for (unsigned d = 0; d < Dim; ++d)
          cell_indices_interp(i, d) = cell_dim_indices(i, d);
      });

    RealMatView values_interp("values_interp",
                              static_cast<size_t>(layout_->GetNumDofHolder()),
                              static_cast<size_t>(n_comp));
    Kokkos::parallel_for(
      "copy_dof_data_to_values_interp",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {0, 0}, {static_cast<LO>(layout_->GetNumDofHolder()), n_comp}),
      KOKKOS_LAMBDA(const LO i, const int c) {
        values_interp(i, c) = dof_data(i, c);
      });

    auto interpolator = RegularGridInterpolator(
      parametric_coords, values_interp, cell_indices_interp, dimensions_view);
    auto interpolated = interpolator.linear_interpolation();

    Kokkos::parallel_for(
      "evaluate_order1_device",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_points, n_comp}),
      KOKKOS_LAMBDA(const LO i, const int c) {
        values(i, c) = interpolated(i, c);
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
  /// @param layout uniform-grid layout of the evaluated fields
  explicit UniformGridEvaluatorFactory(
    std::shared_ptr<const UniformGridFieldLayout<Dim>> layout)
    : TypedPointEvaluatorFactory<Real, Bound>(layout->GetCoordinateSystem()),
      layout_(std::move(layout)),
      grid_(layout_->GetGrid())
  {
  }

protected:
  [[nodiscard]] bool SupportsNearestBoundary() const noexcept override
  {
    return true;
  }

  std::shared_ptr<Bound> BindPoints(const BindRequest& request) const override
  {
    PCMS_FUNCTION_TIMER;
    return std::make_shared<Bound>(request, grid_);
  }

  std::unique_ptr<PointEvaluator<Real>> CreateEvaluator(
    std::shared_ptr<const Bound> bound, int /*derivative_order*/) const override
  {
    return std::make_unique<UniformGridPointEvaluator<Dim>>(layout_,
                                                            std::move(bound));
  }

private:
  std::shared_ptr<const UniformGridFieldLayout<Dim>> layout_;
  UniformGrid<Dim> grid_;
};

using UniformGridEvaluatorFactory2D = UniformGridEvaluatorFactory<2>;

} // namespace pcms

#endif // PCMS_UNIFORM_GRID_EVALUATOR_FACTORY_H
