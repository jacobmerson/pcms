#include "fusionio/fusionio.hpp"
#include <pcms/utility/assert.hpp>

namespace fusionio
{

// Fusion-IO 1.x's names of its field types (fio_get_field_name).
std::string_view Name(Quantity quantity)
{
  switch (quantity) {
    case Quantity::Alpha: return "alpha";
    case Quantity::ScalarPotential: return "scalar potential";
    case Quantity::TotalPressure: return "total pressure";
    case Quantity::PoloidalFlux: return "poloidal flux";
    case Quantity::PoloidalFluxNorm: return "normalized poloidal flux";
    case Quantity::TotalRadiation: return "total radiation";
    case Quantity::Resistivity: return "resistivity";
    case Quantity::Viscosity: return "viscosity";
    case Quantity::ThermalDiffusivity: return "thermal diffusivity";
    case Quantity::BootstrapCurrent: return "JpdotB";
    case Quantity::BootstrapCurrentDnDpsi: return "JpdotB_dndpsi";
    case Quantity::BootstrapCurrentDteDpsi: return "JpdotB_dtedpsi";
    case Quantity::BootstrapCurrentDtiDpsi: return "JpdotB_dtidpsi";
    case Quantity::BootstrapCurrentL31: return "JpdotB_L31";
    case Quantity::BootstrapCurrentL32: return "JpdotB_L32";
    case Quantity::BootstrapCurrentL34: return "JpdotB_L34";
    case Quantity::BootstrapCurrentAlpha: return "JpdotB_alpha";
    case Quantity::Pressure: return "pressure";
    case Quantity::Density: return "density";
    case Quantity::Temperature: return "temperature";
    case Quantity::ElectricField: return "electric field";
    case Quantity::VectorPotential: return "vector potential";
    case Quantity::MagneticField: return "magnetic field";
    case Quantity::CurrentDensity: return "current density";
    case Quantity::FluidVelocity: return "fluid velocity";
    case Quantity::Velocity: return "velocity";
  }
  throw pcms::pcms_error("fusionio: unknown quantity");
}

} // namespace fusionio
