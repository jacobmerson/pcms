#ifndef FUSIONIO_SRC_GNET_READER_HPP
#define FUSIONIO_SRC_GNET_READER_HPP

// Private (not installed): a GNET 5D distribution file (HDF5), after
// pcms::gnet::GnetHDF5Reader and calculate_number_density of the PCMS branch
// gnet-gtc-coupling, without Omega_h.

#include <cstddef>
#include <string>
#include <vector>

namespace fusionio::detail
{

/// The distribution of a GNET file on its (rho, theta, zeta, energy, pitch)
/// bins.
///
/// The file holds the bins of rho (radius), zeta (toroidal angle, up to
/// max_zeta_rad), energy (up to max_energy_ev) and pitch; the poloidal bins
/// of radial bin rho are NumPoloidalBins(rho). A spatial cell is
/// (PoloidalOffset(rho) + theta) * num_zeta + zeta, and f5d and volume3d are
/// indexed by it.
struct GnetData
{
  int num_rho = 0;
  int num_zeta = 0;
  int num_energy = 0;
  int num_pitch = 0;
  double max_energy_ev = 0;
  double max_zeta_rad = 0;
  /// [cell][energy][pitch], row-major
  std::vector<double> f5d;
  /// the volume of each cell
  std::vector<double> volume3d;

  /// The poloidal bins of radial bin `rho`: 4 (rho + 1).
  static int NumPoloidalBins(int rho) { return 4 * (rho + 1); }
  /// The poloidal bins of the radial bins before `rho`.
  static int PoloidalOffset(int rho);
  /// The cells.
  [[nodiscard]] std::size_t NumCells() const;
  /// The markers: one per (cell, energy, pitch), in that order.
  [[nodiscard]] std::size_t NumMarkers() const;
  /// f5d / volume3d of every marker, in marker order.
  [[nodiscard]] std::vector<double> NumberDensity() const;
};

/// Reads the file; the layout is
///   /distribution5D/dimensions/{mxenergy, mxrho, mxzeta, mxpitch}  (int)
///   /distribution5D/range/{engmax_eV, zetamax_rad}                  (double)
///   /distribution5D/data/f5d       (cell, energy, pitch)            (double)
///   /distribution5D/data/volume3d  (cell)                           (double)
/// @throws std::runtime_error if the file or a dataset cannot be read or the
///         datasets do not match the dimensions
GnetData ReadGnet(const std::string& path);

} // namespace fusionio::detail

#endif // FUSIONIO_SRC_GNET_READER_HPP
