#ifndef PCMS_FIELD_FIELD_METADATA_HPP
#define PCMS_FIELD_FIELD_METADATA_HPP

#include "pcms/field/units.hpp"
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>

namespace pcms
{

/// Physical properties of a field's quantity that a transfer should preserve.
enum class FieldProperty : std::uint8_t
{
  /// A density whose integral is conserved, e.g. particle density.
  Conserved = 1U << 0U,
  DivergenceFree = 1U << 1U,
  NonNegative = 1U << 2U,
};

/// A set of FieldProperty values.
class FieldProperties
{
public:
  constexpr FieldProperties() = default;
  constexpr FieldProperties(std::initializer_list<FieldProperty> properties)
  {
    for (const auto property : properties) {
      bits_ |= static_cast<std::uint8_t>(property);
    }
  }

  [[nodiscard]] constexpr bool Has(FieldProperty property) const noexcept
  {
    return (bits_ & static_cast<std::uint8_t>(property)) != 0;
  }

  friend constexpr bool operator==(FieldProperties, FieldProperties) = default;

private:
  std::uint8_t bits_ = 0;
};

/// What a field's values mean. The coordinate system and value basis are not
/// here: they live on the layout and the field data.
struct FieldMetadata
{
  std::string name;
  /// Unset when unknown: no unit checks apply.
  std::optional<Units> units;
  FieldProperties properties;
};

/// Throws unless values of `source` may be written to `target` as they are:
/// when both have units, they must be the same. PCMS does not convert units.
/// @param operation what moves the values, for the message, e.g. "transfer"
void CheckSameUnits(const FieldMetadata& source, const FieldMetadata& target,
                    const std::string& operation);

} // namespace pcms

#endif // PCMS_FIELD_FIELD_METADATA_HPP
