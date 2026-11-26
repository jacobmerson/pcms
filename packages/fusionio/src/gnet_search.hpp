#ifndef FUSIONIO_SRC_GNET_SEARCH_HPP
#define FUSIONIO_SRC_GNET_SEARCH_HPP

// Private (not installed): the analytic bin search of GNET's (rho, theta,
// zeta, energy, pitch) grid, after gnet_grid.{hpp,cpp} of the PCMS branch
// gnet-gtc-coupling (GnetSearch::analyticSearchGnet and its helpers), without
// Omega_h. The supports of a query point are the markers of the bins around
// it. The poloidal offset of a radial bin (the scan array of the original) is
// its closed form.

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstddef>

namespace fusionio::detail
{

/// The most bins of one coordinate around a query point, and of all five.
constexpr int MaxBinsEachDirection = 200;
constexpr int MaxGnetSupports = 500;

/// The bins of a GNET grid.
struct GnetGrid
{
  std::size_t num_rho;
  std::size_t num_zeta;
  std::size_t num_pitch;
  std::size_t num_energy;
  double zeta_max;   // toroidal extent (rad)
  double energy_max; // eV
};

/// The bins of each coordinate around a query point for MLS with at least
/// `min_num_supports` supports: floor(min_num_supports^(1/5)) + 1 (the
/// original's BinNeighbors::eval).
inline int NumNeighbourBins(std::size_t min_num_supports)
{
  return static_cast<int>(
           std::pow(static_cast<double>(min_num_supports), 0.2)) +
         1;
}

/// The radial bin whose centre is at or below `rho` (normalized, [0, 1]).
KOKKOS_INLINE_FUNCTION
std::size_t SearchRadialBin(std::size_t num_rho, double rho)
{
  const double bin = static_cast<double>(num_rho) * rho - 0.5;
  return static_cast<std::size_t>(bin < 0.0 ? 0.0 : bin);
}

/// The poloidal bin whose centre is at or below `theta`, in radial bin `rho`
/// (4 (rho + 1) poloidal bins).
KOKKOS_INLINE_FUNCTION
std::size_t SearchPoloidalBin(std::size_t rho, double theta)
{
  const std::size_t num_poloidal = 4 * (rho + 1);
  const double dtheta = 2 * 3.14159265358979323846 / num_poloidal;
  const double bin = theta / dtheta - 0.5;
  return static_cast<std::size_t>(bin < 0.0 ? 0.0 : bin);
}

KOKKOS_INLINE_FUNCTION
std::size_t SearchToroidalBin(std::size_t num_zeta, double zeta_max,
                              double zeta)
{
  const double dzeta = zeta_max / static_cast<double>(num_zeta);
  const double bin = zeta / dzeta - 0.5;
  return static_cast<std::size_t>(bin < 0.0 ? 0.0 : bin);
}

KOKKOS_INLINE_FUNCTION
std::size_t SearchEnergyBin(std::size_t num_energy, double energy_max,
                            double energy)
{
  const double bin = num_energy * energy / energy_max - 0.5;
  return static_cast<std::size_t>(bin < 0.0 ? 0.0 : bin);
}

KOKKOS_INLINE_FUNCTION
std::size_t SearchPitchBin(std::size_t num_pitch, double pitch)
{
  const double bin = 0.5 * num_pitch * (pitch + 1) - 0.5;
  return static_cast<std::size_t>(bin < 0.0 ? 0.0 : bin);
}

/// The index of the cell of poloidal bin `theta` of radial bin `rho`.
KOKKOS_INLINE_FUNCTION
std::size_t CellIndex(std::size_t rho, std::size_t theta, std::size_t zeta,
                      std::size_t num_zeta)
{
  return (2 * rho * (rho + 1) + theta) * num_zeta + zeta;
}

/// The index of the marker of a cell, energy bin and pitch bin.
KOKKOS_INLINE_FUNCTION
std::size_t MarkerIndex(std::size_t num_pitch, std::size_t num_energy,
                        std::size_t cell, std::size_t pitch, std::size_t energy)
{
  return cell * num_pitch * num_energy + energy * num_pitch + pitch;
}

/// The `count` bins around bin `bin` of `num_bins`: `bin`, then alternately
/// the next one below and above, wrapping if `periodic`, else moving to the
/// other side at an end.
KOKKOS_INLINE_FUNCTION
void BinNeighbours(std::size_t num_bins, std::size_t bin, int count,
                   bool periodic, int* neighbours)
{
  neighbours[0] = static_cast<int>(bin);
  std::size_t left = bin;
  std::size_t right = bin;
  for (int n = 1; n < count; ++n) {
    if (n % 2 != 1) {
      if (periodic) {
        left = (left - 1 + num_bins) % num_bins;
        neighbours[n] = static_cast<int>(left);
      } else if (left > 0) {
        left -= 1;
        neighbours[n] = static_cast<int>(left);
      } else {
        right += 1;
        neighbours[n] = static_cast<int>(right);
      }
    } else {
      if (periodic) {
        right = (right + 1) % num_bins;
        neighbours[n] = static_cast<int>(right);
      } else if (right < num_bins - 1) {
        right += 1;
        neighbours[n] = static_cast<int>(right);
      } else {
        left -= 1;
        neighbours[n] = static_cast<int>(left);
      }
    }
  }
}

/// The supports of the query point (rho, theta, zeta, energy, pitch): the
/// markers of the `count` bins of each coordinate around it (radial, energy
/// and pitch bins do not wrap; poloidal and toroidal ones do), written to
/// `supports`, count^5 of them, ordered by (radial, poloidal, toroidal)
/// neighbour, then energy, then pitch.
KOKKOS_INLINE_FUNCTION
void GnetSupports(const GnetGrid& grid, int count, const double* query,
                  int* supports)
{
  int radial[MaxBinsEachDirection] = {};
  int toroidal[MaxBinsEachDirection] = {};
  int poloidal[MaxBinsEachDirection] = {};
  int pitch[MaxBinsEachDirection] = {};
  int energy[MaxBinsEachDirection] = {};

  BinNeighbours(grid.num_rho, SearchRadialBin(grid.num_rho, query[0]), count,
                false, radial);
  BinNeighbours(grid.num_zeta,
                SearchToroidalBin(grid.num_zeta, grid.zeta_max, query[2]),
                count, true, toroidal);
  BinNeighbours(grid.num_pitch, SearchPitchBin(grid.num_pitch, query[4]), count,
                false, pitch);
  BinNeighbours(grid.num_energy,
                SearchEnergyBin(grid.num_energy, grid.energy_max, query[3]),
                count, false, energy);

  int n = 0;
  for (int i = 0; i < count; ++i) {
    const std::size_t rho = radial[i];
    BinNeighbours(4 * (rho + 1), SearchPoloidalBin(rho, query[1]), count, true,
                  poloidal);
    for (int j = 0; j < count; ++j) {
      for (int k = 0; k < count; ++k) {
        const std::size_t cell =
          CellIndex(rho, poloidal[j], toroidal[k], grid.num_zeta);
        for (int e = 0; e < count; ++e) {
          for (int p = 0; p < count; ++p) {
            supports[n++] = static_cast<int>(MarkerIndex(
              grid.num_pitch, grid.num_energy, cell, pitch[p], energy[e]));
          }
        }
      }
    }
  }
}

} // namespace fusionio::detail

#endif // FUSIONIO_SRC_GNET_SEARCH_HPP
