#ifndef FUSIONIO_TEST_GNET_SYNTHETIC_HPP
#define FUSIONIO_TEST_GNET_SYNTHETIC_HPP

// A small synthetic GNET file, for the tests and as the input of the GNET
// tools' reference runs.

#include <numbers>
#include <string>

namespace fusionio::test
{

/// The dimensions of a synthetic file.
struct GnetShape
{
  int num_rho = 4;
  int num_zeta = 3;
  int num_energy = 5;
  int num_pitch = 4;
};

/// The ranges of a synthetic file (those of the GTC grid).
constexpr double SyntheticMaxEnergyEv = 200000.0;
constexpr double SyntheticMaxZetaRad = 2.0 * std::numbers::pi / 5.0;

/// The number density of the synthetic file at a bin centre: smooth and
/// positive. The file holds n * volume for f5d.
double SyntheticNumberDensity(double rho, double theta, double zeta,
                              double energy, double pitch);

/// The volume of the cell of the bin centres (rho, theta): positive and
/// growing with rho.
double SyntheticVolume(double rho, double theta);

/// Writes the file (a GNET file with the bin centres of GNET's grid).
/// @throws std::runtime_error if it cannot be written
void WriteSyntheticGnet(const std::string& path, const GnetShape& shape = {});

} // namespace fusionio::test

#endif // FUSIONIO_TEST_GNET_SYNTHETIC_HPP
