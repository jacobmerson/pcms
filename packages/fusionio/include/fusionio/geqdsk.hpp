#ifndef FUSIONIO_GEQDSK_HPP
#define FUSIONIO_GEQDSK_HPP

/// geqdsk equilibria read and evaluated natively (no Fusion-IO 1.x): the
/// poloidal flux psi(R, Z) as a bicubic spline on the file's uniform (R, Z)
/// grid and the profiles of the file as cubic splines in psi, as PCMS
/// functions.
///
/// The quantities are Fusion-IO's: PoloidalFlux is 2 pi psi (Wb),
/// PoloidalFluxNorm is (psi - psi_axis) / (psi_lcfs - psi_axis), TotalPressure
/// is p(psi) in Pa and MagneticField is
/// B = F(psi) grad(phi) + grad(phi) x grad(psi), as physical (R, phi, Z)
/// components, in tesla. The profiles are constant outside [psi_axis,
/// psi_lcfs] at their end values (1.x's convention). Time slices, parts,
/// species, scale and phase do not exist: CreateFunction throws when an
/// option is set.

#include "fusionio/fusionio.hpp"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace fusionio
{

/// (R, Z) points of a closed or open curve in the poloidal plane.
using Polyline = std::vector<std::array<Real, 2>>;

/// The profiles of a geqdsk file, functions of psi.
enum class GeqdskProfile
{
  /// F = R B_phi, in T m
  PoloidalCurrent,
  /// p, in Pa
  Pressure,
  /// q
  SafetyFactor
};

/// An open geqdsk file. Its Source geometry takes (R, phi, Z) points; psi does
/// not depend on phi.
class GeqdskSource : public Source
{
public:
  /// The plasma boundary (the file's rbbbs, zbbbs); empty if the file has none.
  [[nodiscard]] virtual const Polyline& Boundary() const = 0;

  /// The limiter (the file's rlim, zlim); empty if the file has none.
  [[nodiscard]] virtual const Polyline& Limiter() const = 0;

  /// A profile as a cubic spline of psi on [psi_axis, psi_lcfs] (the file's
  /// uniform grid in the normalized flux), on a one-dimensional geometry
  /// whose coordinate is psi in Wb/rad (the file's units, not 2 pi psi).
  [[nodiscard]] virtual pcms::Function<Real> CreateProfile(
    GeqdskProfile profile) const = 0;
};

/// Reads a geqdsk file.
/// @throws pcms_error if the file cannot be read or is malformed
[[nodiscard]] std::shared_ptr<GeqdskSource> OpenGeqdsk(const std::string& path);

} // namespace fusionio

#endif // FUSIONIO_GEQDSK_HPP
