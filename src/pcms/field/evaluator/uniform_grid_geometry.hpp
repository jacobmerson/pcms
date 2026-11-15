#ifndef PCMS_FIELD_EVALUATOR_UNIFORM_GRID_GEOMETRY_HPP
#define PCMS_FIELD_EVALUATOR_UNIFORM_GRID_GEOMETRY_HPP

#include "pcms/discretization/discretization/uniform_grid.hpp"
#include "pcms/field/coordinate_system.hpp"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/geometry/geometry.hpp"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/profile.h"
#include "pcms/utility/types.h"
#include "pcms/utility/uniform_grid.h"
#include <Kokkos_Core.hpp>
#include <Omega_h_vector.hpp>
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
  /// @param geometry geometry the points are localized on
  /// @param request query points and out-of-bounds policy
  /// @param grid grid to locate the points in
  UniformGridBinding(std::shared_ptr<const Geometry> geometry,
                     const BindRequest& request, const UniformGrid<Dim>& grid)
    : BoundSource(std::move(geometry), request)
  {
    auto [found, cells] =
      detail::LocateInGrid(grid, LocatedPointValues(),
                           Policy().mode == OutOfBoundsMode::NEAREST_BOUNDARY);
    SetFound(found);
    cell_indices_ = cells;
  }

  /// @brief Takes the cells of the points as given; the positions in the
  ///        cells are not used yet.
  /// @param geometry geometry the points are localized on
  /// @param points cell of each point and its position in the cell; no points
  ///        gives an empty bound source
  /// @param policy out-of-bounds policy of the evaluators
  /// @param grid grid whose cells the ids index
  /// @throws pcms_error if an id is not a cell of `grid`
  UniformGridBinding(std::shared_ptr<const Geometry> geometry,
                     const ParametricPoints& points, OutOfBoundsPolicy policy,
                     const UniformGrid<Dim>& grid)
    : BoundSource(std::move(geometry), points, policy),
      cell_indices_(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                       "uniform_grid_cell_indices"),
                    points.ids.extent(0))
  {
    if (points.NumPoints() == 0) {
      return;
    }
    detail::CheckParametricIds(points, grid.GetNumCells(), "UniformGridBinding",
                               "cell");
    Kokkos::deep_copy(cell_indices_, points.ids);
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

/// @brief Geometry of a uniform grid: an affine map of the unit cells, located
///        in closed form.
/// @tparam Dim grid dimension
template <unsigned Dim>
class UniformGridGeometry final : public Geometry
{
public:
  /// @brief Creates the geometry owned by a std::shared_ptr, as Bind needs.
  /// @param grid the grid
  /// @param system coordinate system of the grid coordinates
  [[nodiscard]] static std::shared_ptr<UniformGridGeometry> Create(
    UniformGrid<Dim> grid, std::shared_ptr<const CoordinateSystem> system)
  {
    return std::make_shared<UniformGridGeometry>(grid, std::move(system));
  }

  /// @brief Create it with Create or std::make_shared: Bind uses
  ///        shared_from_this().
  /// @param grid the grid
  /// @param system coordinate system of the grid coordinates
  UniformGridGeometry(UniformGrid<Dim> grid,
                      std::shared_ptr<const CoordinateSystem> system)
    : grid_(grid),
      discretization_(std::make_shared<UniformGridDiscretization<Dim>>(grid)),
      system_(ResolveCoordinateSystem(std::move(system), Dim))
  {
  }

  [[nodiscard]] std::shared_ptr<const Discretization> GetDiscretization()
    const noexcept override
  {
    return discretization_;
  }

  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>&
  GetCoordinateSystem() const noexcept override
  {
    return system_;
  }

  [[nodiscard]] bool SupportsNearestBoundary() const noexcept override
  {
    return true;
  }

  /// @return the grid
  [[nodiscard]] const UniformGrid<Dim>& GetGrid() const noexcept
  {
    return grid_;
  }

  /// @brief The cell Jacobian diag(edge_length / divisions), shared by every
  ///        cell: one row (row 0), per_element = true.
  /// @throws pcms_error if `bound` belongs to another geometry
  [[nodiscard]] DifferentialView Differential(
    const BoundSource& bound) const override
  {
    if (bound.GetGeometry().get() != this) {
      throw pcms_error(
        "Differential: the bound source belongs to another geometry");
    }
    Kokkos::View<Real***, DeviceMemorySpace> jacobian("uniform_grid_jacobian",
                                                      1, Dim, Dim);
    Kokkos::View<Real*, DeviceMemorySpace> determinant(
      "uniform_grid_jacobian_det", 1);
    auto jacobian_host = Kokkos::create_mirror_view(jacobian);
    auto determinant_host = Kokkos::create_mirror_view(determinant);
    Real det = 1;
    for (unsigned d = 0; d < Dim; ++d) {
      const Real h = grid_.edge_length[d] / grid_.divisions[d];
      jacobian_host(0, d, d) = h;
      det *= h;
    }
    determinant_host(0) = det;
    Kokkos::deep_copy(jacobian, jacobian_host);
    Kokkos::deep_copy(determinant, determinant_host);
    return DifferentialView{jacobian, determinant, /*per_element=*/true};
  }

protected:
  [[nodiscard]] std::shared_ptr<BoundSource> BindImpl(
    const BindRequest& request) const override
  {
    PCMS_FUNCTION_TIMER;
    return std::make_shared<UniformGridBinding<Dim>>(shared_from_this(),
                                                     request, grid_);
  }

  [[nodiscard]] std::shared_ptr<BoundSource> BindParametricImpl(
    const ParametricPoints& points, OutOfBoundsPolicy policy) const override
  {
    return std::make_shared<UniformGridBinding<Dim>>(shared_from_this(), points,
                                                     policy, grid_);
  }

private:
  UniformGrid<Dim> grid_;
  std::shared_ptr<const Discretization> discretization_;
  std::shared_ptr<const CoordinateSystem> system_;
};

} // namespace pcms

#endif // PCMS_FIELD_EVALUATOR_UNIFORM_GRID_GEOMETRY_HPP
