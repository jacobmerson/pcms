#include "fusionio/fusionio.hpp"
#include <pcms/utility/assert.hpp>
#include <algorithm>
#include <iterator>
#include <string>

namespace fusionio
{

namespace
{

struct QuantityEntry
{
  Quantity quantity;
  // Fusion-IO 1.x's name of the field type (fio_get_field_name)
  std::string_view name;
  // the units 1.x returns the values in, empty where not certain
  std::string_view units;
};

// TODO: constraints per quantity and space guarantees (design-invariants.md
// §5 Fusion-IO)
constexpr QuantityEntry Quantities[] = {
  {Quantity::Alpha, "alpha", ""},
  {Quantity::ScalarPotential, "scalar potential", "V"},
  {Quantity::TotalPressure, "total pressure", "Pa"},
  {Quantity::PoloidalFlux, "poloidal flux", ""},
  {Quantity::PoloidalFluxNorm, "normalized poloidal flux", ""},
  {Quantity::TotalRadiation, "total radiation", ""},
  {Quantity::Resistivity, "resistivity", ""},
  {Quantity::Viscosity, "viscosity", ""},
  {Quantity::ThermalDiffusivity, "thermal diffusivity", ""},
  {Quantity::BootstrapCurrent, "JpdotB", ""},
  {Quantity::BootstrapCurrentDnDpsi, "JpdotB_dndpsi", ""},
  {Quantity::BootstrapCurrentDteDpsi, "JpdotB_dtedpsi", ""},
  {Quantity::BootstrapCurrentDtiDpsi, "JpdotB_dtidpsi", ""},
  {Quantity::BootstrapCurrentL31, "JpdotB_L31", ""},
  {Quantity::BootstrapCurrentL32, "JpdotB_L32", ""},
  {Quantity::BootstrapCurrentL34, "JpdotB_L34", ""},
  {Quantity::BootstrapCurrentAlpha, "JpdotB_alpha", ""},
  {Quantity::Pressure, "pressure", "Pa"},
  {Quantity::Density, "density", "m^-3"},
  {Quantity::Temperature, "temperature", "eV"},
  {Quantity::ElectricField, "electric field", "V/m"},
  {Quantity::VectorPotential, "vector potential", "T*m"},
  {Quantity::MagneticField, "magnetic field", "T"},
  {Quantity::CurrentDensity, "current density", "A/m^2"},
  {Quantity::FluidVelocity, "fluid velocity", "m/s"},
  {Quantity::Velocity, "velocity", "m/s"},
};

const QuantityEntry& Entry(Quantity quantity)
{
  const auto* found =
    std::find_if(std::begin(Quantities), std::end(Quantities),
                 [quantity](const auto& e) { return e.quantity == quantity; });
  if (found == std::end(Quantities)) {
    throw pcms::pcms_error("fusionio: unknown quantity");
  }
  return *found;
}

} // namespace

std::string_view Name(Quantity quantity)
{
  return Entry(quantity).name;
}

pcms::FieldMetadata Metadata(Quantity quantity)
{
  const QuantityEntry& entry = Entry(quantity);
  pcms::FieldMetadata metadata{.name = std::string(entry.name),
                               .extent = pcms::Extent::Intensive};
  if (!entry.units.empty()) {
    metadata.units = pcms::Units::Parse(entry.units);
  }
  return metadata;
}

} // namespace fusionio
