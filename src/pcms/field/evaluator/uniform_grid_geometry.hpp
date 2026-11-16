#ifndef PCMS_FIELD_EVALUATOR_UNIFORM_GRID_GEOMETRY_HPP
#define PCMS_FIELD_EVALUATOR_UNIFORM_GRID_GEOMETRY_HPP

#include "pcms/discretization/discretization/uniform_grid.hpp"
#include "pcms/field/coordinate_system.hpp"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/geometry/geometry.hpp"
#include "pcms/localization/uniform_grid_search.hpp"
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

/// Uniform-grid bound source: the cell of each query point and its position in
/// the cell. Points outside the grid are not found, except with
/// OutOfBoundsMode::NEAREST_BOUNDARY, where they use the closest cell and
/// their position relative to it.
template <unsigned Dim>
class UniformGridBinding final : public BoundSource
{
public:
  UniformGridBinding(std::shared_ptr<const Geometry> geometry,
                     const BindRequest& request, const UniformGrid<Dim>& grid)
    : BoundSource(std::move(geometry), request)
  {
    using Result = typename UniformGridPointSearch<Dim>::Result;
    const auto results =
      UniformGridPointSearch<Dim>(grid)(LocatedPointValues());
    const auto n = static_cast<LO>(results.extent(0));
    const bool nearest = Policy().mode == OutOfBoundsMode::NEAREST_BOUNDARY;
    Kokkos::View<bool*, DeviceMemorySpace> found("uniform_grid_found", n);
    cell_indices_ =
      Kokkos::View<LO*, DeviceMemorySpace>("uniform_grid_cell_indices", n);
    parametric_ = Kokkos::View<Real**, DeviceMemorySpace>(
      "uniform_grid_parametric", n, Dim);
    const auto cell_indices = cell_indices_;
    const auto parametric = parametric_;
    Kokkos::parallel_for(
      "UniformGridBinding",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, n),
      KOKKOS_LAMBDA(const LO i) {
        const Result result = results(i);
        found(i) = nearest || !result.IsOutside();
        cell_indices(i) = result.CellId();
        for (unsigned d = 0; d < Dim; ++d) {
          parametric(i, d) = result.parametric[d];
        }
      });
    SetFound(found);
  }

  [[nodiscard]] Kokkos::View<const LO*, DeviceMemorySpace> CellIndices()
    const noexcept
  {
    return cell_indices_;
  }

  /// [point][axis]: the position of each query point in its cell, each
  /// coordinate in [0, 1] for a point inside the grid
  [[nodiscard]] Kokkos::View<const Real**, DeviceMemorySpace> Parametric()
    const noexcept
  {
    return parametric_;
  }

private:
  Kokkos::View<LO*, DeviceMemorySpace> cell_indices_;
  Kokkos::View<Real**, DeviceMemorySpace> parametric_;
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

private:
  UniformGrid<Dim> grid_;
  std::shared_ptr<const Discretization> discretization_;
  std::shared_ptr<const CoordinateSystem> system_;
};

} // namespace pcms

#endif // PCMS_FIELD_EVALUATOR_UNIFORM_GRID_GEOMETRY_HPP
