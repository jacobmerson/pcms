#include "nimrod_space.hpp"
#include "host_evaluation.hpp"
#include "quantity_metadata.hpp"
#include "nimeval.hpp"
#include "pcms/utility/profile.h"
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <numbers>
#include <optional>
#include <ranges>
#include <regex>
#include <sstream>

namespace fio::detail
{

// Private implementation built on PCMS's public types.
using namespace pcms;

/// The dump nimeval has initialized and the quantities it can evaluate.
/// nimeval stops the whole process on bad input, so everything passed to it is
/// checked first.
class NimrodSource
{
public:
  struct QuantityShape
  {
    int components;
    /// Whether it is an (R, Z, phi) vector rather than scalars.
    bool vector;
  };

  /// Initializes nimeval with `dump` and the field ids in `fields`, reading
  /// the case's nimrod.in from `case_dir`.
  NimrodSource(const std::string& dump, const std::string& fields,
               const std::string& case_dir);
  ~NimrodSource();
  NimrodSource(const NimrodSource&) = delete;
  NimrodSource& operator=(const NimrodSource&) = delete;

  /// The quantities that can be evaluated, by NIMROD's name.
  [[nodiscard]] const std::map<std::string, QuantityShape>& Quantities()
    const noexcept
  {
    return quantities_;
  }

private:
  std::map<std::string, QuantityShape> quantities_;
};

namespace
{

// nimeval's selector for the part of a field to evaluate.
int NimevalPart(Part part)
{
  switch (part) {
    case Part::Total: return 2;
    case Part::Perturbed: return 0;
    case Part::Equilibrium: return 1;
  }
  throw pcms_error("NimrodFunctionSpace: unknown part");
}

// Whether a source is open: nimeval holds one dump per process.
bool& SourceOpen()
{
  static bool open = false;
  return open;
}

// Makes `dir` the current directory for the guard's lifetime.
class ScopedCurrentPath
{
public:
  explicit ScopedCurrentPath(const std::filesystem::path& dir)
    : previous_(std::filesystem::current_path())
  {
    std::filesystem::current_path(dir);
  }
  ~ScopedCurrentPath()
  {
    std::error_code ignored;
    std::filesystem::current_path(previous_, ignored);
  }
  ScopedCurrentPath(const ScopedCurrentPath&) = delete;
  ScopedCurrentPath& operator=(const ScopedCurrentPath&) = delete;

private:
  std::filesystem::path previous_;
};

// The value assigned to `key` in a Fortran namelist file, lower case and
// unquoted.
std::optional<std::string> NamelistValue(const std::string& text,
                                         const std::string& key)
{
  const std::regex entry("(^|[\\s,])" + key + "\\s*=\\s*'?([^'\\s,/]*)");
  std::smatch match;
  if (!std::regex_search(text, match, entry)) {
    return std::nullopt;
  }
  return match[2].str();
}

// A quantity of a source: NIMROD's name for it and the part to evaluate.
struct NimrodQuantity
{
  std::shared_ptr<NimrodSource> source;
  std::string name;
  // nimeval's equilibrium/perturbation selector
  int eq;
};

using NimrodFieldData = ExternalFieldData<NimrodQuantity>;

// The query points and each point's search guess, shared by every evaluator
// created from this bound source.
class NimrodBoundSource final : public HostBoundSource
{
public:
  explicit NimrodBoundSource(const BindRequest& request)
    : HostBoundSource(request), guesses(points.extent(0), {-1.0, -1.0})
  {
    if (request.policy.mode != OutOfBoundsMode::ERROR) {
      throw pcms_error("NimrodFunctionSpace: only OutOfBoundsMode::ERROR is "
                       "supported");
    }
  }

  // Each point's logical coordinates from its last search, which NIMROD
  // starts the next search from; (-1, -1) searches from scratch.
  mutable std::vector<std::array<double, 2>> guesses;
};

// Evaluates a quantity, or with derivative order 1 the gradient of a scalar
// quantity. NIMROD evaluation is not thread safe, don't call Evaluate
// concurrently.
class NimrodEvaluator final : public PointEvaluator<Real>
{
public:
  NimrodEvaluator(std::shared_ptr<NimrodSource> source,
                  std::shared_ptr<const NimrodBoundSource> bound,
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
    const auto* data = dynamic_cast<const NimrodFieldData*>(&field.GetData());
    if (data == nullptr || data->quantity.source != source_) {
      throw pcms_error(
        "NimrodFunctionSpace: field is not a quantity of this source");
    }
    const auto out = values.GetValues();
    const auto& points = bound_->points;
    auto& guesses = bound_->guesses;
    const int num_components = field.GetLayout().GetNumComponents();
    const auto& quantity = data->quantity;
    const bool vector = data->GetValueType() == FieldValueType::Vector;
    const int dmode = DerivativeOrder();
    // NIMROD writes the value and, for dmode 1, (d/dR, d/dZ, (1/R) d/dphi)
    // after it.
    std::array<double, 4> scalar_and_gradient{};
    const auto host_out = HostRowsLike(out);
    for (size_t i = 0; i < points.extent(0); ++i) {
      // (R, theta, Z) -> NIMROD's (R, Z, phi) with phi = -theta
      const std::array<double, 3> rzp{points(i, 0), points(i, 2),
                                      -points(i, 1)};
      // A point not yet found starts from where the previous point was.
      auto& guess = guesses[i];
      if (guess[0] < 0 && i > 0) {
        guess = guesses[i - 1];
      }
      int ifail = 0;
      double* fval = dmode == 0 ? &host_out(i, 0) : scalar_and_gradient.data();
      c_eval_field_dm(rzp.data(), quantity.name.c_str(), fval, &num_components,
                      guess.data(), &ifail, &dmode, &quantity.eq);
      if (ifail != 0) {
        guess = {-1.0, -1.0};
        throw pcms_error("NimrodFunctionSpace: query point " +
                         std::to_string(i) + " is outside the dump's domain");
      }
      if (dmode == 1) {
        // d/dtheta = -d/dphi
        host_out(i, 0) = scalar_and_gradient[1];
        host_out(i, 1) = -scalar_and_gradient[3];
        host_out(i, 2) = scalar_and_gradient[2];
      } else if (vector) {
        // (v_R, v_Z, v_phi) -> (v_R, v_theta, v_Z)
        const Real v_z = host_out(i, 1);
        host_out(i, 1) = -host_out(i, 2);
        host_out(i, 2) = v_z;
      }
    }
    CopyToOutput(host_out, out);
  }

private:
  std::shared_ptr<NimrodSource> source_;
  std::shared_ptr<const NimrodBoundSource> bound_;
};

// Binds a dump's points and creates the evaluators of a space.
class NimrodEvaluatorFactory final
  : public TypedPointEvaluatorFactory<Real, NimrodBoundSource>
{
public:
  NimrodEvaluatorFactory(std::shared_ptr<NimrodSource> source,
                         int num_components)
    : TypedPointEvaluatorFactory(csys::CylindricalRThetaZ::Create(), 1),
      source_(std::move(source)),
      num_components_(num_components)
  {
  }

protected:
  std::shared_ptr<NimrodBoundSource> BindPoints(
    const BindRequest& request) const override
  {
    return std::make_shared<NimrodBoundSource>(request);
  }

  std::unique_ptr<PointEvaluator<Real>> CreateEvaluator(
    std::shared_ptr<const NimrodBoundSource> bound,
    int derivative_order) const override
  {
    CheckGradientComponents("NimrodFunctionSpace", derivative_order,
                            num_components_);
    return std::make_unique<NimrodEvaluator>(source_, std::move(bound),
                                             derivative_order);
  }

private:
  std::shared_ptr<NimrodSource> source_;
  int num_components_;
};

} // namespace

NimrodSource::NimrodSource(const std::string& dump, const std::string& fields,
                           const std::string& case_dir)
{
  namespace fs = std::filesystem;
  if (fields.empty()) {
    throw pcms_error("fio::OpenNimrod: no field ids given");
  }
  const fs::path dir =
    case_dir.empty() ? fs::current_path() : fs::path(case_dir);
  std::ifstream input(dir / "nimrod.in");
  if (!input) {
    throw pcms_error("fio::OpenNimrod: cannot read " +
                     (dir / "nimrod.in").string());
  }
  std::stringstream buffer;
  buffer << input.rdbuf();
  std::string text = std::regex_replace(buffer.str(), std::regex("!.*"), "");
  std::ranges::transform(text, text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  // Points are passed to NIMROD as (R, Z, phi), which only holds for
  // toroidal cases.
  if (NamelistValue(text, "geom") != "tor") {
    throw pcms_error("fio::OpenNimrod: only toroidal cases (geom='tor') are "
                     "supported");
  }
  const int nisp = std::stoi(NamelistValue(text, "nisp").value_or("0"));
  const int ds_nqty = std::stoi(NamelistValue(text, "ds_nqty").value_or("1"));

  // The quantities each field id initializes, as in nimpy.
  const struct
  {
    const char* name;
    char id;
    QuantityShape shape;
  } supported[] = {{"b", 'b', {3, true}},   {"v", 'v', {3, true}},
                   {"j", 'j', {3, true}},   {"n", 'n', {nisp + 1, false}},
                   {"p", 'p', {1, false}},  {"ti", 't', {1, false}},
                   {"te", 't', {1, false}}, {"d", 'd', {ds_nqty, false}}};
  for (const char id : fields) {
    bool known = false;
    for (const auto& entry : supported) {
      if (entry.id == id) {
        quantities_.emplace(entry.name, entry.shape);
        known = true;
      }
    }
    if (!known) {
      throw pcms_error(std::string("fio::OpenNimrod: unsupported field id '") +
                       id + "'; supported ids are bvjnptd");
    }
  }
  const fs::path dump_path = fs::absolute(dump);
  if (!fs::is_regular_file(dump_path)) {
    throw pcms_error("fio::OpenNimrod: cannot find dump '" + dump + "'");
  }
  if (access(dir.c_str(), W_OK) != 0) {
    throw pcms_error("fio::OpenNimrod: NIMROD writes scratch files to the "
                     "case directory, but " +
                     dir.string() + " is not writable");
  }

  if (SourceOpen()) {
    throw pcms_error("fio::OpenNimrod: another NIMROD source is open; NIMROD "
                     "can evaluate one dump per process");
  }
  {
    const ScopedCurrentPath in_case_dir(dir);
    c_nimfield_init(dump_path.c_str(), fields.c_str());
  }
  SourceOpen() = true;
}

NimrodSource::~NimrodSource()
{
  SourceOpen() = false;
}

std::shared_ptr<NimrodSource> OpenNimrodSource(const std::string& dump,
                                               const std::string& fields,
                                               const std::string& case_dir)
{
  return std::make_shared<NimrodSource>(dump, fields, case_dir);
}

std::vector<std::string> AvailableQuantities(const NimrodSource& source)
{
  const auto names = std::views::keys(source.Quantities());
  return {names.begin(), names.end()};
}

std::shared_ptr<NimrodFunctionSpace> NimrodFunctionSpace::Create(
  std::shared_ptr<NimrodSource> source, int num_components)
{
  return std::shared_ptr<NimrodFunctionSpace>(
    new NimrodFunctionSpace(std::move(source), num_components));
}

NimrodFunctionSpace::NimrodFunctionSpace(std::shared_ptr<NimrodSource> source,
                                         int num_components)
  : FunctionSpace(
      std::make_shared<NimrodEvaluatorFactory>(source, num_components)),
    source_(std::move(source)),
    layout_(std::make_shared<EmptyFieldLayout>(
      csys::CylindricalRThetaZ::Create(), num_components))
{
  if (source_ == nullptr) {
    throw pcms_error("NimrodFunctionSpace: source must not be null");
  }
}

Function<Real> NimrodFunctionSpace::CreateFunction(
  std::string name, const std::string& quantity,
  const NimrodOptions& options) const
{
  const auto& quantities = source_->Quantities();
  const auto found = quantities.find(quantity);
  if (found == quantities.end()) {
    ThrowUnknownQuantity("NimrodFunctionSpace", quantity,
                         AvailableQuantities(*source_));
  }
  const auto [components, vector] = found->second;
  CheckComponents("NimrodFunctionSpace", quantity, components,
                  layout_->GetNumComponents());
  return FunctionSpace::CreateFunction<Real>(
    std::move(name),
    std::make_unique<NimrodFieldData>(
      NimrodQuantity{source_, quantity, NimevalPart(options.part)},
      QuantityBasis(vector)));
}

FieldVariant NimrodFunctionSpace::CreateFieldImpl(Type /*storage_type*/,
                                                  ValueBasis /*basis*/) const
{
  throw pcms_error("NimrodFunctionSpace: a value type does not name a "
                   "quantity; use CreateFunction(name, quantity)");
}

FieldVariant NimrodFunctionSpace::CreateFieldImpl(FieldDataVariant data) const
{
  return WrapField<Real>(
    layout_, std::get<std::unique_ptr<FieldData<Real>>>(std::move(data)));
}

namespace
{

/// A NIMROD dump as a fio::Source.
class NimrodFileSource final : public Source
{
public:
  explicit NimrodFileSource(std::shared_ptr<NimrodSource> source)
    : source_(std::move(source))
  {
  }

  [[nodiscard]] std::vector<QuantityInfo> Quantities() const override
  {
    std::vector<QuantityInfo> quantities;
    for (const auto& [quantity, name] : Names) {
      const auto found = source_->Quantities().find(name);
      if (found != source_->Quantities().end()) {
        quantities.push_back({quantity, found->second.components});
      }
    }
    return quantities;
  }

  [[nodiscard]] std::vector<Real> Times() const override { return {}; }

  [[nodiscard]] Real Period() const override { return 2 * std::numbers::pi; }

  [[nodiscard]] std::optional<Real> EvaluateSeries(Series /*series*/,
                                                   Real /*time*/) const override
  {
    return std::nullopt;
  }

  [[nodiscard]] Function<Real> CreateFunction(
    Quantity quantity, const Options& options) const override
  {
    if (options.timeslice || options.linear_scale || options.phase) {
      throw pcms_error("fio: NIMROD sources support only the part option");
    }
    if (options.species && quantity != Quantity::Temperature) {
      throw pcms_error("fio: NIMROD sources support the species option only "
                       "for temperature");
    }
    const std::string name = NimrodName(quantity, options.species);
    const auto found = source_->Quantities().find(name);
    if (found == source_->Quantities().end()) {
      throw pcms_error("fio: this NIMROD source does not provide " +
                       std::string(fio::Name(quantity)) + " (NIMROD's \"" +
                       name + "\"); open it with that field id");
    }
    auto function =
      NimrodFunctionSpace::Create(source_, found->second.components)
        ->CreateFunction(std::string(fio::Name(quantity)), name,
                         {.part = options.part.value_or(Part::Total)});
    SetQuantityMetadata(function, quantity);
    if (quantity == Quantity::Temperature) {
      // NIMROD's temperature units are the case's choice (kboltz in
      // nimrod.in: eV or K); not read yet.
      function.SetUnits(std::nullopt);
    }
    return function;
  }

private:
  // NIMROD's names for the quantities it shares with Fusion-IO. Its field id
  // 't' provides both temperatures; "ti" stands for them here.
  static constexpr std::pair<Quantity, const char*> Names[] = {
    {Quantity::MagneticField, "b"},  {Quantity::FluidVelocity, "v"},
    {Quantity::CurrentDensity, "j"}, {Quantity::Density, "n"},
    {Quantity::TotalPressure, "p"},  {Quantity::Temperature, "ti"},
  };

  static std::string NimrodName(Quantity quantity,
                                std::optional<Species> species)
  {
    if (quantity == Quantity::Temperature) {
      return species == Species::Electron ? "te" : "ti";
    }
    for (const auto& [known, name] : Names) {
      if (known == quantity) {
        return name;
      }
    }
    throw pcms_error("fio: NIMROD sources do not provide " +
                     std::string(fio::Name(quantity)));
  }

  std::shared_ptr<NimrodSource> source_;
};

} // namespace

} // namespace fio::detail

namespace fio
{

std::shared_ptr<Source> OpenNimrod(const std::string& dump,
                                   const std::string& fields,
                                   const std::string& case_dir)
{
  return std::make_shared<detail::NimrodFileSource>(
    detail::OpenNimrodSource(dump, fields, case_dir));
}

} // namespace fio
