#ifndef PCMS_FIELD_UNITS_HPP
#define PCMS_FIELD_UNITS_HPP

#include <memory>
#include <string>
#include <string_view>

namespace pcms
{

/// The conventions of a unit string.
enum class UnitSyntax
{
  /// SI and common notations: "T", "m/s", "A.m^-2", "kg/(A*s^2)", "eV".
  Default,
  /// CF/UDUNITS (netCDF attributes): space-separated factors with exponents,
  /// e.g. "m2 s-1" or "J kg-1 K-1", and "degrees_Celsius".
  CF,
  /// The IMAS Data Dictionary: as Default, plus "-" for dimensionless.
  IMAS,
};

/// Physical units, parsed from a string at setup. Transfers check them once
/// per Apply, never per point.
class Units
{
public:
  /// Throws if `text` is not a unit in `syntax`, including IMAS "mixed" and
  /// "as_parent", which name no single unit (leave such units unspecified).
  [[nodiscard]] static Units Parse(std::string_view text,
                                   UnitSyntax syntax = UnitSyntax::Default);

  /// A canonical form that Parse reads back, e.g. "T" or "0.0001T" (gauss).
  [[nodiscard]] std::string ToString() const;

  /// Whether a value in these units converts to `other` by a scale factor or
  /// an offset: tesla and gauss do, tesla and metre do not.
  [[nodiscard]] bool SameDimensions(const Units& other) const;

  /// The same units, scale and offset included: tesla == "kg/(A*s^2)", but
  /// tesla != gauss.
  friend bool operator==(const Units& a, const Units& b);

private:
  struct Impl;
  explicit Units(std::shared_ptr<const Impl> impl);

  std::shared_ptr<const Impl> impl_;
};

} // namespace pcms

#endif // PCMS_FIELD_UNITS_HPP
