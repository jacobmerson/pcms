#ifndef FUSIONIO_FUSIONIO_HPP
#define FUSIONIO_FUSIONIO_HPP

/// Fusion-IO 2.0: fields of fusion simulation codes as PCMS functions.
///
/// A Source is an open file (or dump) of one code. Its quantities are created
/// as pcms::Function objects, which are evaluated, differentiated, mapped and
/// transferred with PCMS. They share the source's geometry, so points bound
/// once (pcms::QueryPoints on GetGeometry()) serve every quantity. Points are
/// (R, phi, Z) cylindrical coordinates
/// (pcms CylindricalRThetaZ: right-handed, phi is theta), values are in SI
/// units, and vector values are physical cylindrical components. Meshes are
/// serial for now: every rank opens the whole source.

#include <pcms/field/field.h>
#include <pcms/geometry/geometry.hpp>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fusionio
{

using Real = pcms::Real;

/// File formats read through Fusion-IO 1.x.
enum class Format
{
  M3DC1,
  GEQDSK,
  GATO,
  MARS,
  GPEC
};

/// Quantities a source can provide (Fusion-IO 1.x's field types).
enum class Quantity
{
  // scalars
  Alpha,
  ScalarPotential,
  TotalPressure,
  PoloidalFlux,
  PoloidalFluxNorm,
  TotalRadiation,
  Resistivity,
  Viscosity,
  ThermalDiffusivity,
  BootstrapCurrent,
  BootstrapCurrentDnDpsi,
  BootstrapCurrentDteDpsi,
  BootstrapCurrentDtiDpsi,
  BootstrapCurrentL31,
  BootstrapCurrentL32,
  BootstrapCurrentL34,
  BootstrapCurrentAlpha,
  // species-dependent scalars (see Options::species)
  Pressure,
  Density,
  Temperature,
  // vectors
  ElectricField,
  VectorPotential,
  MagneticField,
  CurrentDensity,
  FluidVelocity,
  // species-dependent vectors
  Velocity
};

/// Fusion-IO's name for a quantity, e.g. "magnetic field".
std::string_view Name(Quantity quantity);

/// Which part of a field to evaluate.
enum class Part
{
  Total,
  Perturbed,
  Equilibrium
};

enum class Species
{
  MainIon,
  Electron
};

/// Scalar time series a source may provide.
enum class Series
{
  MagneticAxisPsi,
  LcfsPsi,
  MagneticAxisR,
  MagneticAxisZ
};

/// Options for CreateFunction. Unset members keep the source's default;
/// setting one a source does not support throws.
struct Options
{
  /// Index into Times(); -1 selects the equilibrium.
  std::optional<int> timeslice;
  std::optional<Part> part;
  /// For species-dependent quantities.
  std::optional<Species> species;
  /// Scale of the perturbation of linear runs.
  std::optional<Real> linear_scale;
  /// Toroidal phase shift in radians: the field at phi is the source's field
  /// at phi - phase.
  std::optional<Real> phase;
};

/// A quantity a source provides and its number of components.
struct QuantityInfo
{
  Quantity quantity;
  int num_components;
};

/// An open source. Functions created from it keep it open.
class Source
{
public:
  virtual ~Source() = default;

  /// The geometry of the source's functions: bind query points on it once
  /// and evaluate any of its functions there.
  [[nodiscard]] virtual std::shared_ptr<const pcms::Geometry> GetGeometry()
    const = 0;

  /// The quantities CreateFunction accepts.
  [[nodiscard]] virtual std::vector<QuantityInfo> Quantities() const = 0;

  /// Times of the time slices; empty if the source has none.
  [[nodiscard]] virtual std::vector<Real> Times() const = 0;

  /// Toroidal period in radians (2 pi for a full torus).
  [[nodiscard]] virtual Real Period() const = 0;

  /// Value of a time series at `time`, or nullopt if the source has none.
  [[nodiscard]] virtual std::optional<Real> EvaluateSeries(Series series,
                                                           Real time) const = 0;

  /// The quantity as a function of (R, phi, Z) on GetGeometry(); it holds no
  /// values in PCMS, so it can be evaluated and used as a transfer source but
  /// not written. Its space supports pcms::DifferentialOperator::Gradient
  /// when the source provides the quantity's derivatives (gradients of scalar
  /// quantities only, so far).
  /// Throws if the source does not provide the quantity or an option.
  [[nodiscard]] virtual pcms::Function<Real> CreateFunction(
    Quantity quantity, const Options& options = {}) const = 0;
};

/// How the functions of a source opened with Open are evaluated.
enum class Evaluation
{
  /// by Fusion-IO 1.x, point by point on the host (every format)
  FusionIO,
  /// by Kokkos kernels on the device, from the file's mesh and coefficients
  /// as Fusion-IO 1.x reads them: M3D-C1 sources with a 2D mesh, whose
  /// fields are axisymmetric or of one toroidal mode (linear runs). The
  /// scalar quantities stored as one field (with gradients and Hessians) and
  /// the magnetic field are evaluated at (R, phi, Z) points, as with
  /// FusionIO. CreateFunction throws for the quantities and options it does
  /// not evaluate natively.
  Native
};

/// Opens a file through Fusion-IO 1.x, e.g. an M3D-C1 C1.h5 (with its
/// time-slice files next to it) or a geqdsk file. Only sources with
/// cylindrical geometry are supported.
/// @throws if `evaluation` is Native and the source cannot be evaluated
///         natively
std::shared_ptr<Source> Open(Format format, const std::string& path,
                             Evaluation evaluation = Evaluation::FusionIO);

/// Opens a NIMROD dump for the fields named by NIMROD's single-character ids
/// in `fields` (e.g. "bvn"). NIMROD's evaluator holds one dump per process,
/// so this throws while another NIMROD source is open. Only toroidal cases
/// are supported, and only the Part option. Not thread safe.
///
/// @param case_dir directory holding the case's nimrod.in (default: the
/// current directory); it must be writable. `dump` is relative to the
/// current directory.
/// @throws if Fusion-IO was built without NIMROD
std::shared_ptr<Source> OpenNimrod(const std::string& dump,
                                   const std::string& fields,
                                   const std::string& case_dir = "");

} // namespace fusionio

#endif // FUSIONIO_FUSIONIO_HPP
