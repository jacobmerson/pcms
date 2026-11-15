#include "fusion_io_space.hpp"
#include "fusion_io_source.hpp"
#include "host_evaluation.hpp"
#include "pcms/utility/profile.h"
#include <algorithm>
#include <numbers>
#include <string_view>

namespace fio::detail
{

// Private implementation built on PCMS's public types.
using namespace pcms;

namespace
{

struct QuantityEntry
{
  Quantity quantity;
  field_type type;
  std::string_view name;
};

// Fusion-IO 1.x's field types and names (fio_get_field_name), in Quantity
// order.
constexpr QuantityEntry QuantityTable[] = {
  {Quantity::Alpha, FIO_ALPHA, "alpha"},
  {Quantity::ScalarPotential, FIO_SCALAR_POTENTIAL, "scalar potential"},
  {Quantity::TotalPressure, FIO_TOTAL_PRESSURE, "total pressure"},
  {Quantity::PoloidalFlux, FIO_POLOIDAL_FLUX, "poloidal flux"},
  {Quantity::PoloidalFluxNorm, FIO_POLOIDAL_FLUX_NORM,
   "normalized poloidal flux"},
  {Quantity::TotalRadiation, FIO_TOTAL_RADIATION, "total radiation"},
  {Quantity::Resistivity, FIO_RESISTIVITY, "resistivity"},
  {Quantity::Viscosity, FIO_VISCOSITY, "viscosity"},
  {Quantity::ThermalDiffusivity, FIO_THERMAL_DIFFUSIVITY,
   "thermal diffusivity"},
  {Quantity::BootstrapCurrent, FIO_JBS, "JpdotB"},
  {Quantity::BootstrapCurrentDnDpsi, FIO_JBS_dndpsi, "JpdotB_dndpsi"},
  {Quantity::BootstrapCurrentDteDpsi, FIO_JBS_dtedpsi, "JpdotB_dtedpsi"},
  {Quantity::BootstrapCurrentDtiDpsi, FIO_JBS_dtidpsi, "JpdotB_dtidpsi"},
  {Quantity::BootstrapCurrentL31, FIO_JBS_L31, "JpdotB_L31"},
  {Quantity::BootstrapCurrentL32, FIO_JBS_L32, "JpdotB_L32"},
  {Quantity::BootstrapCurrentL34, FIO_JBS_L34, "JpdotB_L34"},
  {Quantity::BootstrapCurrentAlpha, FIO_JBS_alpha, "JpdotB_alpha"},
  {Quantity::Pressure, FIO_PRESSURE, "pressure"},
  {Quantity::Density, FIO_DENSITY, "density"},
  {Quantity::Temperature, FIO_TEMPERATURE, "temperature"},
  {Quantity::ElectricField, FIO_ELECTRIC_FIELD, "electric field"},
  {Quantity::VectorPotential, FIO_VECTOR_POTENTIAL, "vector potential"},
  {Quantity::MagneticField, FIO_MAGNETIC_FIELD, "magnetic field"},
  {Quantity::CurrentDensity, FIO_CURRENT_DENSITY, "current density"},
  {Quantity::FluidVelocity, FIO_FLUID_VELOCITY, "fluid velocity"},
  {Quantity::Velocity, FIO_VELOCITY, "velocity"},
};

int FusionIOPart(Part part)
{
  switch (part) {
    case Part::Total: return FIO_TOTAL;
    case Part::Perturbed: return FIO_PERTURBED_ONLY;
    case Part::Equilibrium: return FIO_EQUILIBRIUM_ONLY;
  }
  throw pcms_error("fio: unknown part");
}

int FusionIOSpecies(Species species)
{
  switch (species) {
    case Species::MainIon: return FIO_MAIN_ION;
    case Species::Electron: return FIO_ELECTRON;
  }
  throw pcms_error("fio: unknown species");
}

series_type FusionIOSeries(Series series)
{
  switch (series) {
    case Series::MagneticAxisPsi: return FIO_MAGAXIS_PSI;
    case Series::LcfsPsi: return FIO_LCFS_PSI;
    case Series::MagneticAxisR: return FIO_MAGAXIS_R;
    case Series::MagneticAxisZ: return FIO_MAGAXIS_Z;
  }
  throw pcms_error("fio: unknown series");
}

const QuantityEntry& EntryOf(Quantity quantity)
{
  for (const auto& entry : QuantityTable) {
    if (entry.quantity == quantity) {
      return entry;
    }
  }
  throw pcms_error("fio: unknown quantity");
}

// A quantity of a source, read with its options.
struct FusionIOQuantity
{
  std::shared_ptr<FusionIOSource> source;
  FusionIOFieldPtr field;
};

using FusionIOFieldData = ExternalFieldData<FusionIOQuantity>;

// The query points in Fusion-IO's layout and the per-point search hints. The
// hints let Fusion-IO start each point's search where it last found it; every
// field and evaluator created from this bound source shares them, and the
// next binding of the source to as many points starts from them.
//
// With OutOfBoundsMode::FILL the points outside the source's domain are found
// here by evaluating the source's domain probe at every point; the fields of
// a source share its domain, so they hold for every quantity. Otherwise every
// point counts as found, and evaluating outside the domain throws.
class FusionIOBoundSource final : public HostBoundSource
{
public:
  FusionIOBoundSource(const FusionIOSource& source, const BindRequest& request)
    : HostBoundSource(request),
      hint_size(source.Get().sizeof_search_hint()),
      hints(source.SearchHints(points.extent(0)))
  {
    if (request.policy.mode == OutOfBoundsMode::FILL) {
      FindPointsInDomain(source.DomainProbe());
    }
  }

  [[nodiscard]] fio_hint Hint(size_t i) const
  {
    return hint_size > 0 ? &(*hints)[i * hint_size] : nullptr;
  }

  size_t hint_size;
  std::shared_ptr<std::vector<char>> hints;

private:
  void FindPointsInDomain(fio_field& probe)
  {
    Kokkos::View<bool*, HostMemorySpace> inside(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fusion_io_inside"),
      points.extent(0));
    std::vector<Real> values(static_cast<size_t>(probe.dimension()));
    for (size_t i = 0; i < points.extent(0); ++i) {
      const int status = probe.eval(&points(i, 0), values.data(), Hint(i));
      inside(i) = status != FIO_OUT_OF_BOUNDS;
      if (status != FIO_SUCCESS && status != FIO_OUT_OF_BOUNDS) {
        throw pcms_error("FusionIOFunctionSpace: locating query point " +
                         std::to_string(i) + " failed (Fusion-IO error " +
                         std::to_string(status) + ")");
      }
    }
    SetFound(inside);
  }
};

// Evaluates a quantity, or with derivative order 1 its gradient, which
// Fusion-IO provides for scalar quantities of some formats.
// Fusion-IO evaluation is not thread safe, don't call Evaluate concurrently
class FusionIOEvaluator final : public PointEvaluator<Real>
{
public:
  FusionIOEvaluator(std::shared_ptr<FusionIOSource> source,
                    std::shared_ptr<const FusionIOBoundSource> bound,
                    int derivative_order)
    : PointEvaluator<Real>(bound, derivative_order),
      source_(std::move(source)),
      bound_(std::move(bound))
  {
  }

protected:
  void EvaluateImpl(const Field<Real>& field,
                    ValueView<Real, DeviceMemorySpace> values) const override
  {
    PCMS_FUNCTION_TIMER;
    const auto* data = dynamic_cast<const FusionIOFieldData*>(&field.GetData());
    if (data == nullptr || data->quantity.source != source_) {
      throw pcms_error(
        "FusionIOFunctionSpace: field is not a quantity of this source");
    }
    const auto out = values.GetValues();
    const auto& points = bound_->points;
    const auto host_out = HostRowsLike(out);
    fio_field& quantity = *data->quantity.field;
    const bool gradient = DerivativeOrder() == 1;
    for (size_t k = 0; k < bound_->found.extent(0); ++k) {
      const auto i = static_cast<size_t>(bound_->found(k));
      Real* row = &host_out(i, 0);
      const int status =
        gradient ? quantity.eval_deriv(&points(i, 0), row, bound_->Hint(i))
                 : quantity.eval(&points(i, 0), row, bound_->Hint(i));
      if (status == FIO_UNSUPPORTED && gradient) {
        throw pcms_error("FusionIOFunctionSpace: Fusion-IO does not provide "
                         "derivatives of this quantity for this format");
      }
      if (status == FIO_OUT_OF_BOUNDS) {
        throw pcms_error(
          "FusionIOFunctionSpace: query point " + std::to_string(i) +
          (bound_->Policy().mode == OutOfBoundsMode::FILL
             ? " is outside this quantity's domain but inside the source's"
             : " is outside the source's domain"));
      }
      if (status != FIO_SUCCESS) {
        throw pcms_error("FusionIOFunctionSpace: evaluation failed at query "
                         "point " +
                         std::to_string(i) + " (Fusion-IO error " +
                         std::to_string(status) + ")");
      }
      if (gradient) {
        // Fusion-IO gives the coordinate partial along phi; the physical
        // component is (1/R) d/dphi.
        row[1] /= points(i, 0);
      }
    }
    CopyToOutput(host_out, out);
  }

private:
  std::shared_ptr<FusionIOSource> source_;
  std::shared_ptr<const FusionIOBoundSource> bound_;
};

// Binds a source's points and creates the evaluators of a space.
class FusionIOEvaluatorFactory final
  : public TypedPointEvaluatorFactory<Real, FusionIOBoundSource>
{
public:
  FusionIOEvaluatorFactory(std::shared_ptr<FusionIOSource> source,
                           int num_components)
    : TypedPointEvaluatorFactory(csys::CylindricalRThetaZ::Create(), 1),
      source_(std::move(source)),
      num_components_(num_components)
  {
  }

protected:
  std::shared_ptr<FusionIOBoundSource> BindPoints(
    const BindRequest& request) const override
  {
    return std::make_shared<FusionIOBoundSource>(*source_, request);
  }

  std::unique_ptr<PointEvaluator<Real>> CreateEvaluator(
    std::shared_ptr<const FusionIOBoundSource> bound,
    int derivative_order) const override
  {
    CheckGradientComponents("FusionIOFunctionSpace", derivative_order,
                            num_components_);
    return std::make_unique<FusionIOEvaluator>(source_, std::move(bound),
                                               derivative_order);
  }

private:
  std::shared_ptr<FusionIOSource> source_;
  int num_components_;
};

} // namespace

field_type FieldTypeOf(Quantity quantity)
{
  return EntryOf(quantity).type;
}

int NumComponents(Quantity quantity)
{
  // Fusion-IO's vector field types start at FIO_ELECTRIC_FIELD (1001).
  return FieldTypeOf(quantity) >= FIO_ELECTRIC_FIELD ? 3 : 1;
}

FusionIOSource::FusionIOSource(FusionIOSourcePtr source)
  : source_(std::move(source))
{
  // Points are passed to Fusion-IO as (R, phi, Z), which only holds for
  // cylindrical sources.
  int geometry = -1;
  if (source_->get_int_parameter(FIO_GEOMETRY, &geometry) != FIO_SUCCESS ||
      geometry != FIO_CYLINDRICAL) {
    throw pcms_error("fio::Open: only sources with cylindrical geometry are "
                     "supported");
  }
  fio_source::fio_field_list fields;
  source_->get_available_fields(&fields);
  for (const auto& entry : QuantityTable) {
    if (std::ranges::find(fields, entry.type) != fields.end()) {
      quantities_.push_back(entry.quantity);
    }
  }
}

bool FusionIOSource::Provides(Quantity quantity) const
{
  return std::ranges::find(quantities_, quantity) != quantities_.end();
}

FusionIOFieldPtr FusionIOSource::GetField(Quantity quantity,
                                          const fio_option_list& options) const
{
  fio_field* field = nullptr;
  const int status =
    source_->get_field(FieldTypeOf(quantity), &field, &options);
  FusionIOFieldPtr owned(field);
  if (status != FIO_SUCCESS) {
    throw pcms_error("fio: could not read \"" + std::string(Name(quantity)) +
                     "\" (Fusion-IO error " + std::to_string(status) + ")");
  }
  return owned;
}

fio_field& FusionIOSource::DomainProbe() const
{
  if (probe_ != nullptr) {
    return *probe_;
  }
  // Sources list quantities their file may lack, so try the ones every MHD
  // source has first (the poloidal flux is a cheap scalar), then the rest.
  std::vector<Quantity> candidates = {Quantity::PoloidalFlux,
                                      Quantity::MagneticField};
  candidates.insert(candidates.end(), quantities_.begin(), quantities_.end());
  // The equilibrium part is the cheapest to evaluate, and for M3D-C1 it is
  // on the mesh already loaded rather than a time slice's.
  fio_option_list options;
  source_->get_field_options(&options);
  options.set_option(FIO_PART, FIO_EQUILIBRIUM_ONLY);
  for (const Quantity quantity : candidates) {
    if (!Provides(quantity)) {
      continue;
    }
    fio_field* field = nullptr;
    if (source_->get_field(FieldTypeOf(quantity), &field, &options) ==
        FIO_SUCCESS) {
      probe_.reset(field);
      return *probe_;
    }
    delete field;
  }
  throw pcms_error("fio: none of the source's quantities can be read to "
                   "locate points");
}

std::shared_ptr<std::vector<char>> FusionIOSource::SearchHints(
  size_t num_points) const
{
  // As in Fusion-IO's korc_example, the hints are one zero-initialized block
  // of sizeof_search_hint() bytes per point. For M3D-C1 a hint is an element
  // to try first, so zero is a valid starting guess.
  const size_t size =
    static_cast<size_t>(source_->sizeof_search_hint()) * num_points;
  if (hints_ == nullptr || hints_->size() != size) {
    hints_ = std::make_shared<std::vector<char>>(size, 0);
  }
  return hints_;
}

std::shared_ptr<FusionIOSource> OpenFusionIOSource(Format format,
                                                   const std::string& filename)
{
  int type = 0;
  switch (format) {
    case Format::M3DC1: type = FIO_M3DC1_SOURCE; break;
    case Format::GEQDSK: type = FIO_GEQDSK_SOURCE; break;
    case Format::GATO: type = FIO_GATO_SOURCE; break;
    case Format::MARS: type = FIO_MARS_SOURCE; break;
    case Format::GPEC: type = FIO_GPEC_SOURCE; break;
  }
  fio_source* opened = nullptr;
  const int status = fio_open_source(&opened, type, filename.c_str());
  // On failure fio_open_source has already deleted the source.
  if (status != FIO_SUCCESS) {
    throw pcms_error("fio::Open: could not open '" + filename +
                     "' (Fusion-IO error " + std::to_string(status) + ")");
  }
  return std::make_shared<FusionIOSource>(FusionIOSourcePtr(opened));
}

std::shared_ptr<FusionIOFunctionSpace> FusionIOFunctionSpace::Create(
  std::shared_ptr<FusionIOSource> source, int num_components)
{
  return std::shared_ptr<FusionIOFunctionSpace>(
    new FusionIOFunctionSpace(std::move(source), num_components));
}

FusionIOFunctionSpace::FusionIOFunctionSpace(
  std::shared_ptr<FusionIOSource> source, int num_components)
  : FunctionSpace(
      std::make_shared<FusionIOEvaluatorFactory>(source, num_components)),
    source_(std::move(source)),
    layout_(std::make_shared<EmptyFieldLayout>(
      csys::CylindricalRThetaZ::Create(), num_components))
{
  if (source_ == nullptr) {
    throw pcms_error("FusionIOFunctionSpace: source must not be null");
  }
}

Function<Real> FusionIOFunctionSpace::CreateFunction(
  std::string name, Quantity quantity, const Options& options) const
{
  if (!source_->Provides(quantity)) {
    std::vector<std::string> available;
    for (const Quantity known : source_->Quantities()) {
      available.emplace_back(Name(known));
    }
    ThrowUnknownQuantity("FusionIOFunctionSpace", Name(quantity), available);
  }
  CheckComponents("FusionIOFunctionSpace", Name(quantity),
                  NumComponents(quantity), layout_->GetNumComponents());
  fio_option_list fio_options;
  source_->Get().get_field_options(&fio_options);
  if (options.timeslice) {
    fio_options.set_option(FIO_TIMESLICE, *options.timeslice);
  }
  if (options.part) {
    fio_options.set_option(FIO_PART, FusionIOPart(*options.part));
  }
  if (options.species) {
    fio_options.set_option(FIO_SPECIES, FusionIOSpecies(*options.species));
  }
  if (options.linear_scale) {
    fio_options.set_option(FIO_LINEAR_SCALE, *options.linear_scale);
  }
  if (options.phase) {
    fio_options.set_option(FIO_PHASE, *options.phase);
  }
  return FunctionSpace::CreateFunction<Real>(
    std::move(name),
    std::make_unique<FusionIOFieldData>(
      FusionIOQuantity{source_, source_->GetField(quantity, fio_options)},
      QuantityBasis(NumComponents(quantity) == 3)));
}

FieldVariant FusionIOFunctionSpace::CreateFieldImpl(Type /*storage_type*/,
                                                    ValueBasis /*basis*/) const
{
  throw pcms_error("FusionIOFunctionSpace: a value type does not name a "
                   "quantity; use CreateFunction(name, quantity)");
}

FieldVariant FusionIOFunctionSpace::CreateFieldImpl(FieldDataVariant data) const
{
  return WrapField<Real>(
    layout_, std::get<std::unique_ptr<FieldData<Real>>>(std::move(data)));
}

namespace
{

/// A Fusion-IO 1.x source as a fio::Source.
class FusionIOFileSource final : public Source
{
public:
  explicit FusionIOFileSource(std::shared_ptr<FusionIOSource> source)
    : source_(std::move(source))
  {
  }

  [[nodiscard]] std::vector<QuantityInfo> Quantities() const override
  {
    std::vector<QuantityInfo> quantities;
    for (const Quantity quantity : source_->Quantities()) {
      quantities.push_back({quantity, NumComponents(quantity)});
    }
    return quantities;
  }

  [[nodiscard]] std::vector<Real> Times() const override
  {
    fio_source& source = source_->Get();
    int num_timeslices = 0;
    if (source.get_int_parameter(FIO_NUM_TIMESLICES, &num_timeslices) !=
        FIO_SUCCESS) {
      return {};
    }
    std::vector<Real> times(num_timeslices);
    for (int i = 0; i < num_timeslices; ++i) {
      if (source.get_slice_time(i, &times[i]) != FIO_SUCCESS) {
        throw pcms_error("fio: could not read the time of time slice " +
                         std::to_string(i));
      }
    }
    return times;
  }

  [[nodiscard]] Real Period() const override
  {
    Real period = 2 * std::numbers::pi;
    source_->Get().get_real_parameter(FIO_PERIOD, &period);
    return period;
  }

  [[nodiscard]] std::optional<Real> EvaluateSeries(Series series,
                                                   Real time) const override
  {
    fio_series* opened = nullptr;
    if (source_->Get().get_series(FusionIOSeries(series), &opened) !=
          FIO_SUCCESS ||
        opened == nullptr) {
      return std::nullopt;
    }
    const std::unique_ptr<fio_series> owned(opened);
    Real value = 0;
    if (owned->eval(time, &value) != FIO_SUCCESS) {
      throw pcms_error("fio: could not evaluate a time series at t = " +
                       std::to_string(time));
    }
    return value;
  }

  [[nodiscard]] Function<Real> CreateFunction(
    Quantity quantity, const Options& options) const override
  {
    return FusionIOFunctionSpace::Create(source_, NumComponents(quantity))
      ->CreateFunction(std::string(Name(quantity)), quantity, options);
  }

private:
  std::shared_ptr<FusionIOSource> source_;
};

} // namespace

} // namespace fio::detail

namespace fio
{

std::string_view Name(Quantity quantity)
{
  return detail::EntryOf(quantity).name;
}

std::shared_ptr<Source> Open(Format format, const std::string& path)
{
  return std::make_shared<detail::FusionIOFileSource>(
    detail::OpenFusionIOSource(format, path));
}

} // namespace fio
