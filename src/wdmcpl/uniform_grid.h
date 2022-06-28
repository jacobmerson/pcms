#ifndef WDM_COUPLING_UNIFORM_GRID_H
#define WDM_COUPLING_UNIFORM_GRID_H
#include "wdmcpl/bounding_box.h"
#include "Omega_h_vector.hpp"
#include <iostream>
#include <numeric>
#include <Kokkos_Core.hpp>
namespace wdmcpl
{
struct UniformGrid
{
  // Make private?
  static constexpr int dim = 2;
  UniformGrid() = default;
  UniformGrid(std::array<Real, dim> edge_length, std::array<Real, dim> bot_left,
              std::array<LO, dim> divisions)
  {
    auto edge_length_h = Kokkos::create_mirror_view(edge_length_);
    auto bot_left_h = Kokkos::create_mirror_view(bot_left_);
    auto divisions_h = Kokkos::create_mirror_view(divisions_);
    for (int i = 0; i < dim; ++i) {
      edge_length_h(i) = edge_length[i];
      bot_left_h(i) = bot_left[i];
      divisions_h(i) = divisions[i];
    }
    Kokkos::deep_copy(divisions_,divisions_h);
    Kokkos::deep_copy(bot_left_,bot_left_h);
    Kokkos::deep_copy(edge_length_,edge_length_h);
    num_cells_ = std::accumulate(divisions.begin(), divisions.end(), 1,
                                 std::multiplies<LO>{});
  }

private:
  Kokkos::View<Real[dim]> edge_length_;
  Kokkos::View<Real[dim]> bot_left_;
  Kokkos::View<LO[dim]> divisions_;
  LO num_cells_;

public:
  [[nodiscard]] KOKKOS_INLINE_FUNCTION LO GetNumCells() const { return num_cells_; }
  /// return the grid cell ID that the input point is inside or closest to if
  /// the point lies outside
  [[nodiscard]] KOKKOS_INLINE_FUNCTION LO
  ClosestCellID(const Omega_h::Vector<dim>& point) const
  {
    std::array<Real, dim> distance_within_grid{point[0] - bot_left_[0],
                                               point[1] - bot_left_[1]};
    std::array<LO, dim> indexes{-1, -1};
    // note that the indexes refer to row/columns which have the opposite order
    // of the coordinates i.e. x,y
    for (int i = 0; i < dim; ++i) {
      if (distance_within_grid[i] <= 0) {
        indexes[dim - (i + 1)] = 0;
      } else if (distance_within_grid[i] >= edge_length_[i]) {
        indexes[dim - (i + 1)] = divisions_[i] - 1;
      } else {
        indexes[dim - (i + 1)] = static_cast<LO>(std::floor(
          distance_within_grid[i] * divisions_[i] / edge_length_[i]));
      }
    }
    return GetCellIndex(indexes[0], indexes[1]);
  }
  [[nodiscard]] KOKKOS_INLINE_FUNCTION AABBox<dim> GetCellBBOX(LO idx) const
  {
    auto [i, j] = GetTwoDCellIndex(idx);
    std::array<Real, dim> half_width = {edge_length_[0] / (2.0 * divisions_[0]),
                                        edge_length_[1] /
                                          (2.0 * divisions_[1])};
    AABBox<dim> bbox{.center = {(2.0 * j + 1.0) * half_width[0] + bot_left_[0],
                                (2.0 * i + 1.0) * half_width[1] + bot_left_[0]},
                     .half_width = half_width};
    return bbox;
  }
  [[nodiscard]] KOKKOS_INLINE_FUNCTION std::array<LO, 2> GetTwoDCellIndex(
    LO idx) const
  {
    return {idx / divisions_[0], idx % divisions_[0]};
  }
  [[nodiscard]] KOKKOS_INLINE_FUNCTION LO GetCellIndex(LO i, LO j) const
  {
    OMEGA_H_CHECK(i >= 0 && j >= 0 && i < divisions_[1] && j < divisions_[0]);
    return i * divisions_[0] + j;
  }
};
} // namespace wdmcpl

#endif // WDM_COUPLING_UNIFORM_GRID_H
