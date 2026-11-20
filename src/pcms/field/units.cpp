#include "pcms/field/units.hpp"
#include "pcms/utility/assert.hpp"
#include <units/units.hpp>
#include <cctype>
#include <cmath>
#include <string>

namespace pcms
{

// LLNL/units (github.com/LLNL/units) parses and compares the units. It is a
// private dependency: no PCMS header includes it.
struct Units::Impl
{
  ::units::precise_unit unit;
};

namespace
{

std::string Trim(std::string_view text)
{
  const auto first = text.find_first_not_of(" \t");
  if (first == std::string_view::npos) {
    return {};
  }
  const auto last = text.find_last_not_of(" \t");
  return std::string(text.substr(first, last - first + 1));
}

// LLNL/units reads CF's space-separated factors only when they are joined by
// '.': "m2 s-1" fails and "mol mol-1" parses as 1/mol^2, while "m2.s-1" and
// "mol.mol-1" are right. It also reads "degrees_Celsius" as an angle.
std::string FromCF(std::string text)
{
  if (text == "degrees_Celsius") {
    return "degC";
  }
  std::string joined;
  bool separator = false;
  for (const char c : text) {
    if (std::isspace(static_cast<unsigned char>(c)) != 0) {
      separator = true;
      continue;
    }
    if (separator) {
      joined += '.';
      separator = false;
    }
    joined += c;
  }
  return joined;
}

std::string FromIMAS(std::string text)
{
  if (text == "-") {
    return "1";
  }
  if (text == "mixed" || text == "as_parent") {
    throw pcms_error("Units::Parse: IMAS units \"" + text +
                     "\" name no single unit; leave the units unspecified");
  }
  return text;
}

} // namespace

Units::Units(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}

Units Units::Parse(std::string_view text, UnitSyntax syntax)
{
  std::string normalized = Trim(text);
  if (normalized.empty()) {
    throw pcms_error("Units::Parse: empty unit string");
  }
  switch (syntax) {
    case UnitSyntax::Default: break;
    case UnitSyntax::CF: normalized = FromCF(std::move(normalized)); break;
    case UnitSyntax::IMAS: normalized = FromIMAS(std::move(normalized)); break;
  }
  const auto unit = ::units::unit_from_string(normalized);
  if (!::units::is_valid(unit) || !std::isfinite(unit.multiplier())) {
    throw pcms_error("Units::Parse: \"" + std::string(text) +
                     "\" is not a unit");
  }
  return Units(std::make_shared<const Impl>(Impl{unit}));
}

std::string Units::ToString() const
{
  return ::units::to_string(impl_->unit);
}

bool Units::SameDimensions(const Units& other) const
{
  return impl_->unit.has_same_base(other.impl_->unit);
}

bool operator==(const Units& a, const Units& b)
{
  return a.impl_->unit == b.impl_->unit;
}

} // namespace pcms
