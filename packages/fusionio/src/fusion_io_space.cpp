#include "fusion_io_space.hpp"
#include "fusion_io_source.hpp"
#include "host_evaluation.hpp"
#include "m3dc1_native.hpp"
#include "pcms/discretization/discretization/empty.hpp"
#include "pcms/field/layout/empty.h"
#include "pcms/utility/profile.h"
#include <algorithm>
#include <map>
#include <numbers>
#include <string_view>
#include <utility>

namespace fusionio::detail
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
  throw pcms_error("fusionio: unknown part");
}

int FusionIOSpecies(Species species)
{
  switch (species) {
    case Species::MainIon: return FIO_MAIN_ION;
    case Species::Electron: return FIO_ELECTRON;
  }
  throw pcms_error("fusionio: unknown species");
}

series_type FusionIOSeries(Series series)
{
  switch (series) {
    case Series::MagneticAxisPsi: return FIO_MAGAXIS_PSI;
    case Series::LcfsPsi: return FIO_LCFS_PSI;
    case Series::MagneticAxisR: return FIO_MAGAXIS_R;
    case Series::MagneticAxisZ: return FIO_MAGAXIS_Z;
  }
  throw pcms_error("fusionio: unknown series");
}

const QuantityEntry& EntryOf(Quantity quantity)
{
  for (const auto& entry : QuantityTable) {
    if (entry.quantity == quantity) {
      return entry;
    }
  }
  throw pcms_error("fusionio: unknown quantity");
}

// A quantity of a source, read with its options.
struct FusionIOQuantity
{
  std::shared_ptr<FusionIOSource> source;
  FusionIOFieldPtr field;
};

using FusionIOFieldData = ExternalFieldData<FusionIOQuantity>;

// The query points of a Fusion-IO source located in its domain, with each
// point's Fusion-IO search hint (sizeof_search_hint() bytes per point; for
// M3D-C1 the element to try first). Every evaluator created from the binding
// shares it.
//
// The hints are state of the binding, sized to its capacity, and survive a
// Geometry::Rebind, so that points that move start their search where they
// were (the warm start lockstep tracers need); points beyond the previous
// capacity start from zero, a valid first guess. Fusion-IO's evaluation
// writes back the hint it used, hence the mutable hints. For M3D-C1 sources,
// whose points are located at bind, that write-back stores the element the
// bind found for every field on the probe's mesh, so evaluations do not
// change the binding observably; other formats have no hints.
//
// Not thread safe: neither is Fusion-IO 1.x.
class FusionIOBinding final : public HostBoundSource
{
public:
  FusionIOBinding(std::shared_ptr<const Geometry> geometry,
                  const BindRequest& request,
                  std::shared_ptr<FusionIOSource> source)
    : HostBoundSource(std::move(geometry), request),
      source_(std::move(source)),
      hint_size_(static_cast<size_t>(source_->Get().sizeof_search_hint())),
      hints_(hint_size_ * static_cast<size_t>(NumPoints()), 0)
  {
    Locate();
  }

  /// The Fusion-IO search hint of query point i; null if the source has none.
  [[nodiscard]] fio_hint Hint(size_t i) const
  {
    return hint_size_ > 0 ? &hints_[i * hint_size_] : nullptr;
  }

protected:
  void RebindImpl() override
  {
    RefreshHostViews();
    const size_t size = hint_size_ * static_cast<size_t>(NumPoints());
    if (hints_.size() < size) {
      hints_.resize(size, 0);
    }
    Locate();
  }

private:
  // Locates the points: M3D-C1 sources search the domain probe's mesh, which
  // finds the points and elements the probe's evaluation would without
  // evaluating it, and leaves each point's element in its hint; other formats
  // evaluate the probe under FILL. The fields of a source share its domain,
  // so the result holds for every quantity.
  void Locate()
  {
    const auto n = static_cast<size_t>(NumPoints());
    if (m3dc1_mesh* mesh = source_->DomainMesh()) {
      // The probe's phase is 0, so its phi is the point's.
      Kokkos::View<bool*, HostMemorySpace> inside(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fusionio_inside"), n);
      for (size_t i = 0; i < n; ++i) {
        int* element = static_cast<int*>(Hint(i));
        *element = mesh->in_element(points(i, 0), points(i, 1), points(i, 2),
                                    nullptr, nullptr, nullptr, *element);
        inside(i) = *element >= 0;
      }
      SetFound(inside);
      return;
    }
    if (Policy().mode != OutOfBoundsMode::FILL) {
      return;
    }
    fio_field& probe = source_->DomainProbe();
    std::vector<Real> values(static_cast<size_t>(probe.dimension()));
    Kokkos::View<bool*, HostMemorySpace> inside(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fusionio_inside"), n);
    for (size_t i = 0; i < n; ++i) {
      const int status = probe.eval(&points(i, 0), values.data(), Hint(i));
      inside(i) = status != FIO_OUT_OF_BOUNDS;
      if (status != FIO_SUCCESS && status != FIO_OUT_OF_BOUNDS) {
        throw pcms_error("FusionIOGeometry: locating query point " +
                         std::to_string(i) + " failed (Fusion-IO error " +
                         std::to_string(status) + ")");
      }
    }
    SetFound(inside);
  }

  std::shared_ptr<FusionIOSource> source_;
  size_t hint_size_;
  // hint_size_ bytes per point, for the binding's capacity
  mutable std::vector<char> hints_;
};

// Evaluates a quantity, or its gradient, which Fusion-IO provides for scalar
// quantities of some formats. Fusion-IO evaluation is not thread safe, don't
// call Evaluate concurrently.
class FusionIOEvaluator final : public PointEvaluator<Real>
{
public:
  FusionIOEvaluator(std::shared_ptr<const FusionIOBinding> bound,
                    DifferentialOperator op,
                    std::shared_ptr<const FieldLayout> layout)
    : PointEvaluator<Real>(bound, op, std::move(layout)),
      bound_(std::move(bound))
  {
  }

protected:
  void EvaluateImpl(const Field<Real>& field,
                    ValueView<Real, DeviceMemorySpace> values) const override
  {
    PCMS_FUNCTION_TIMER;
    const auto* data = dynamic_cast<const FusionIOFieldData*>(&field.GetData());
    if (data == nullptr) {
      throw pcms_error(
        "FusionIOGeometry: the field is not a quantity of a Fusion-IO source");
    }
    const auto out = values.GetValues();
    const auto& points = bound_->points;
    const auto host_out = HostRowsLike(out);
    fio_field& quantity = *data->quantity.field;
    const bool gradient = Operator() == DifferentialOperator::Gradient;
    for (size_t k = 0; k < bound_->found.extent(0); ++k) {
      const auto i = static_cast<size_t>(bound_->found(k));
      Real* row = &host_out(i, 0);
      const int status =
        gradient ? quantity.eval_deriv(&points(i, 0), row, bound_->Hint(i))
                 : quantity.eval(&points(i, 0), row, bound_->Hint(i));
      if (status == FIO_OUT_OF_BOUNDS) {
        throw pcms_error(
          "FusionIOGeometry: query point " + std::to_string(i) +
          (bound_->Policy().mode == OutOfBoundsMode::FILL
             ? " is outside this quantity's domain but inside the source's"
             : " is outside the source's domain"));
      }
      if (status != FIO_SUCCESS) {
        throw pcms_error("FusionIOGeometry: evaluation failed at query "
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
  std::shared_ptr<const FusionIOBinding> bound_;
};

// Creates the evaluators of the quantities of one representation.
class FusionIOEvaluatorFactory final
  : public TypedPointEvaluatorFactory<Real, FusionIOBinding>
{
public:
  FusionIOEvaluatorFactory(std::shared_ptr<const FusionIOGeometry> geometry,
                           std::shared_ptr<const FieldLayout> layout,
                           bool derivatives)
    : TypedPointEvaluatorFactory(
        geometry, derivatives
                    ? std::initializer_list<
                        DifferentialOperator>{DifferentialOperator::Gradient}
                    : std::initializer_list<DifferentialOperator>{}),
      layout_(std::move(layout))
  {
  }

protected:
  std::unique_ptr<PointEvaluator<Real>> CreateEvaluator(
    std::shared_ptr<const FusionIOBinding> bound,
    DifferentialOperator op) const override
  {
    CheckGradientComponents("FusionIOGeometry", op,
                            layout_->GetNumComponents());
    return std::make_unique<FusionIOEvaluator>(std::move(bound), op, layout_);
  }

private:
  std::shared_ptr<const FieldLayout> layout_;
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

bool ProvidesDerivatives(Format format, Quantity quantity)
{
  switch (format) {
    case Format::M3DC1:
      return NumComponents(quantity) == 1 ||
             quantity == Quantity::VectorPotential ||
             quantity == Quantity::MagneticField;
    case Format::GEQDSK: return quantity == Quantity::PoloidalFlux;
    case Format::GATO:
    case Format::MARS:
    case Format::GPEC: return false;
  }
  return false;
}

FusionIOSource::FusionIOSource(Format format, FusionIOSourcePtr source)
  : format_(format), source_(std::move(source))
{
  // Points are passed to Fusion-IO as (R, phi, Z), which only holds for
  // cylindrical sources.
  int geometry = -1;
  if (source_->get_int_parameter(FIO_GEOMETRY, &geometry) != FIO_SUCCESS ||
      geometry != FIO_CYLINDRICAL) {
    throw pcms_error(
      "fusionio::Open: only sources with cylindrical geometry are "
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
    throw pcms_error("fusionio: could not read \"" +
                     std::string(Name(quantity)) + "\" (Fusion-IO error " +
                     std::to_string(status) + ")");
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
  throw pcms_error("fusionio: none of the source's quantities can be read to "
                   "locate points");
}

m3dc1_mesh* FusionIOSource::DomainMesh() const
{
  if (domain_mesh_) {
    return *domain_mesh_;
  }
  domain_mesh_ = nullptr;
  auto* m3dc1 = dynamic_cast<m3dc1_source*>(source_.get());
  if (m3dc1 == nullptr) {
    return nullptr;
  }
  // DomainProbe() of an M3D-C1 source is the equilibrium part of the
  // poloidal flux, which m3dc1_scalar_field::load reads from time slice -1,
  // or from the default slice if the file does not subtract the
  // equilibrium.
  fio_option_list options;
  source_->get_field_options(&options);
  int timeslice = 0;
  options.get_option(FIO_TIMESLICE, &timeslice);
  if (m3dc1->eqsubtract == 1) {
    timeslice = -1;
  }
  m3dc1_field* psi = m3dc1->file.load_field("psi", timeslice);
  if (psi != nullptr && dynamic_cast<m3dc1_stell_field*>(psi) == nullptr) {
    domain_mesh_ = psi->mesh;
  }
  return *domain_mesh_;
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
    throw pcms_error("fusionio::Open: could not open '" + filename +
                     "' (Fusion-IO error " + std::to_string(status) + ")");
  }
  return std::make_shared<FusionIOSource>(format, FusionIOSourcePtr(opened));
}

FusionIOGeometry::FusionIOGeometry(std::shared_ptr<FusionIOSource> source)
  : source_(std::move(source)),
    discretization_(std::make_shared<EmptyDiscretization>()),
    system_(csys::CylindricalRThetaZ::Create())
{
  if (source_ == nullptr) {
    throw pcms_error("FusionIOGeometry: source must not be null");
  }
}

std::shared_ptr<BoundSource> FusionIOGeometry::BindImpl(
  const BindRequest& request) const
{
  return std::make_shared<FusionIOBinding>(shared_from_this(), request,
                                           source_);
}

std::shared_ptr<FunctionSpace> CreateFusionIOSpace(
  std::shared_ptr<const FusionIOGeometry> geometry, int num_components,
  bool derivatives)
{
  auto layout = std::make_shared<EmptyFieldLayout>(
    geometry->GetCoordinateSystem(), num_components);
  auto evaluator = std::make_shared<FusionIOEvaluatorFactory>(
    std::move(geometry), layout, derivatives);
  return std::make_shared<FunctionSpace>(std::move(layout), CreateNoFieldData,
                                         std::move(evaluator));
}

namespace
{

/// A Fusion-IO 1.x source as a fusionio::Source: one geometry, and a space
/// per representation of its quantities, created on first use.
class FusionIOFileSource final : public Source
{
public:
  explicit FusionIOFileSource(std::shared_ptr<FusionIOSource> source)
    : source_(std::move(source)),
      geometry_(std::make_shared<FusionIOGeometry>(source_))
  {
  }

  [[nodiscard]] std::shared_ptr<const Geometry> GetGeometry() const override
  {
    return geometry_;
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
        throw pcms_error("fusionio: could not read the time of time slice " +
                         std::to_string(i));
      }
    }
    return times;
  }

  [[nodiscard]] Real ToroidalPeriod() const override
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
      throw pcms_error("fusionio: could not evaluate a time series at t = " +
                       std::to_string(time));
    }
    return value;
  }

  [[nodiscard]] Function<Real> CreateFunction(
    Quantity quantity, const Options& options) const override
  {
    if (!source_->Provides(quantity)) {
      std::vector<std::string> available;
      for (const Quantity known : source_->Quantities()) {
        available.emplace_back(Name(known));
      }
      ThrowUnknownQuantity("fusionio", Name(quantity), available);
    }
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
    const int num_components = NumComponents(quantity);
    return Space(num_components,
                 ProvidesDerivatives(source_->GetFormat(), quantity))
      ->CreateFunction<Real>(
        std::string(Name(quantity)),
        std::make_unique<FusionIOFieldData>(
          FusionIOQuantity{source_, source_->GetField(quantity, fio_options)},
          QuantityBasis(num_components == 3)));
  }

private:
  // The space of a representation: component count and whether Fusion-IO
  // provides the derivatives.
  const std::shared_ptr<FunctionSpace>& Space(int num_components,
                                              bool derivatives) const
  {
    auto& space = spaces_[{num_components, derivatives}];
    if (space == nullptr) {
      space = CreateFusionIOSpace(geometry_, num_components, derivatives);
    }
    return space;
  }

  std::shared_ptr<FusionIOSource> source_;
  std::shared_ptr<const FusionIOGeometry> geometry_;
  mutable std::map<std::pair<int, bool>, std::shared_ptr<FunctionSpace>>
    spaces_;
};

} // namespace

} // namespace fusionio::detail

namespace fusionio
{

std::string_view Name(Quantity quantity)
{
  return detail::EntryOf(quantity).name;
}

std::shared_ptr<Source> Open(Format format, const std::string& path,
                             Evaluation evaluation)
{
  auto source = detail::OpenFusionIOSource(format, path);
  auto wrapped = std::make_shared<detail::FusionIOFileSource>(source);
  if (evaluation == Evaluation::FusionIO) {
    return wrapped;
  }
  if (format != Format::M3DC1) {
    throw pcms::pcms_error(
      "fusionio::Open: only M3D-C1 sources are evaluated natively");
  }
  return detail::CreateM3DC1NativeSource(std::move(source), std::move(wrapped));
}

} // namespace fusionio
