#ifndef FUSIONIO_GNET_HPP
#define FUSIONIO_GNET_HPP

/// GNET distributions (the FUSIONIO_ENABLE_GNET component): the 5D
/// distribution of GNET's (rho, theta, zeta, energy, pitch) bins as a PCMS
/// function, to be sampled with SampleMarkers into GTC markers.

#include "fusionio/sampling.hpp"
#include <pcms/field/field.hpp>
#include <cstddef>
#include <string>

namespace fusionio
{

struct GnetOptions
{
  /// degree of the polynomial reconstruction
  unsigned degree = 1;
  /// its regularization
  pcms::Real lambda = 1e-4;
  /// whether theta (period 2 pi) and zeta (period the file's zeta extent) are
  /// periodic in the reconstruction: the supports across a seam (the bin
  /// search wraps its bins either way) are used at their images nearest the
  /// point. If not, they are fit at their raw coordinates, as GNET's own
  /// tool does.
  bool periodic = true;
};

/// An open GNET file.
struct GnetDistribution
{
  /// The number density (f5d / volume3d) of the bin centres reconstructed by
  /// moving least squares, with the bins of each coordinate around a point as
  /// supports, on a point cloud in (rho, theta, zeta, energy, pitch): rho is
  /// normalized to [0, 1], theta and zeta are angles in radians, energy is in
  /// eV and pitch in [-1, 1].
  pcms::Function<pcms::Real> density;
  /// the file's toroidal extent (rad) and maximum energy (eV)
  pcms::Real max_zeta;
  pcms::Real max_energy;
  /// the least and greatest cell volume (volume3d): GNET's rejection
  /// threshold is a multiple of 1 / max_volume
  pcms::Real min_volume;
  pcms::Real max_volume;
};

/// Reads a GNET HDF5 file (the datasets are listed in the README).
/// @throws if the file cannot be read, or the options need more bins around a
///         point than the file has or than the search supports
[[nodiscard]] GnetDistribution OpenGnet(const std::string& path,
                                        const GnetOptions& options = {});

/// The box GTC's markers are in: rho in [0, 1] (drawn as sqrt, uniform in
/// area), theta in [0, 2 pi], zeta in [0, max_zeta], energy in [2000, 200000]
/// eV and pitch in [-1, 1].
[[nodiscard]] Domain GTCDomain(pcms::Real max_zeta = 2.0 *
                                                     3.14159265358979323846 /
                                                     5.0);

/// Writes GTC markers to an HDF5 file as /gtc_n6d, a table of `num_markers`
/// rows (rho, theta, zeta, energy, pitch, value), in as many Append calls as
/// the markers come in. The file is created (overwritten) at construction.
class GTCMarkerWriter
{
public:
  /// @throws if the file cannot be created
  GTCMarkerWriter(const std::string& path, std::size_t num_markers);
  ~GTCMarkerWriter();
  GTCMarkerWriter(const GTCMarkerWriter&) = delete;
  GTCMarkerWriter& operator=(const GTCMarkerWriter&) = delete;

  /// Appends the markers (rows of 6 columns, on the device).
  /// @throws if they do not fit in num_markers or are not 6 columns
  void Append(const MarkerRows& rows);

  /// the markers appended so far
  [[nodiscard]] std::size_t Written() const noexcept { return written_; }

private:
  std::int64_t file_ = -1;
  std::int64_t dataset_ = -1;
  std::size_t num_markers_;
  std::size_t written_ = 0;
};

/// Writes all the markers (a GTCMarkerWriter with one Append).
void WriteGTCMarkers(const std::string& path, const MarkerRows& markers);

} // namespace fusionio

#endif // FUSIONIO_GNET_HPP
