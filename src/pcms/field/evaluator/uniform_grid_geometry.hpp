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
#include <string>
#include <utility>

namespace pcms
{

namespace detail
{

/// Per point, whether it is in `grid` (always, if `nearest`) and its closest
/// cell.
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

/// Uniform-grid bound source: the cell of each query point. Points outside the
/// grid are not found, except with OutOfBoundsMode::NEAREST_BOUNDARY, where
/// they use the closest cell.
template <unsigned Dim>
class UniformGridBinding final : public BoundSource
{
public:
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

  [[nodiscard]] Kokkos::View<const LO*, DeviceMemorySpace> CellIndices()
    const noexcept
  {
    return cell_indices_;
  }

private:
  Kokkos::View<LO*, DeviceMemorySpace> cell_indices_;
};

/// Geometry of a uniform grid: an affine map of the unit cells, located in
/// closed form (no search to build).
template <unsigned Dim>
class UniformGridGeometry final : public Geometry
{
public:
  [[nodiscard]] static std::shared_ptr<UniformGridGeometry> Create(
    UniformGrid<Dim> grid, std::shared_ptr<const CoordinateSystem> system)
  {
    return std::make_shared<UniformGridGeometry>(grid, std::move(system));
  }

  /// Use Create or std::make_shared: Bind uses shared_from_this().
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

  [[nodiscard]] const UniformGrid<Dim>& GetGrid() const noexcept
  {
    return grid_;
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
    const ParametricPoints& /*points*/,
    OutOfBoundsPolicy /*policy*/) const override
  {
    throw pcms_error("UniformGridGeometry: parametric binds are not supported");
  }

private:
  UniformGrid<Dim> grid_;
  std::shared_ptr<const Discretization> discretization_;
  std::shared_ptr<const CoordinateSystem> system_;
};

} // namespace pcms

#endif // PCMS_FIELD_EVALUATOR_UNIFORM_GRID_GEOMETRY_HPP
