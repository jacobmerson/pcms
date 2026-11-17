#include "nimrod_space.hpp"
#include "host_evaluation.hpp"
#include "nimeval.hpp"
#include "pcms/discretization/discretization/empty.hpp"
#include "pcms/field/layout/empty.h"
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

namespace fusionio::detail
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

  /// A quantity whose evaluation tells whether a point is inside the dump's
  /// domain: the first one the dump was opened with, both parts.
  struct Probe
  {
    std::string name;
    int components;
    // nimeval's equilibrium/perturbation selector
    int eq;
  };

  [[nodiscard]] const Probe& DomainProbe() const noexcept { return probe_; }

private:
  std::map<std::string, QuantityShape> quantities_;
  Probe probe_;
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
  throw pcms_error("NimrodGeometry: unknown part");
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

// The query points of a NIMROD dump with each point's search guess: the
// logical coordinates nimeval found the point at, which it starts the next
// search from ((-1, -1) searches from scratch). Every quantity of the dump
// shares its domain, so one binding serves them all.
//
// nimeval has no location call separate from evaluation, so under
// OutOfBoundsMode::ERROR nothing is located at bind: the first evaluation of
// each point finds it (and sets its guess) or throws. Under FILL the bind
// evaluates the dump's probe quantity at every point (as the Fusion-IO probe
// formats do): the points nimeval cannot find are missing and every found
// point's guess is set.
//
// The guesses are state of the binding, sized to its capacity, and survive a
// Geometry::Rebind, so that points that move start their search where they
// were; new points search from scratch. nimeval writes back the guess it
// used, hence the mutable guesses; for a found point that is where it was
// found, so evaluations do not change the binding observably.
//
// Not thread safe: neither is nimeval.
class NimrodBinding final : public HostBoundSource
{
public:
  using Guess = std::array<double, 2>;
  static constexpr Guess NoGuess{-1.0, -1.0};

  NimrodBinding(std::shared_ptr<const Geometry> geometry,
                const BindRequest& request,
                std::shared_ptr<NimrodSource> source)
    : HostBoundSource(std::move(geometry), request),
      source_(std::move(source)),
      guesses_(static_cast<size_t>(NumPoints()), NoGuess)
  {
    Locate();
  }

  /// The guess of query point i, which nimeval updates.
  [[nodiscard]] Guess& GuessOf(size_t i) const noexcept { return guesses_[i]; }

  /// nimeval's (R, Z, phi) for query point i (points are (R, theta, Z) with
  /// phi = -theta).
  [[nodiscard]] std::array<double, 3> RZPhi(size_t i) const
  {
    return {points(i, 0), points(i, 2), -points(i, 1)};
  }

protected:
  void RebindImpl() override
  {
    RefreshHostViews();
    const auto n = static_cast<size_t>(NumPoints());
    if (guesses_.size() < n) {
      guesses_.resize(n, NoGuess);
    }
    Locate();
  }

private:
  // Under FILL: evaluates the probe at every point, sets the found points and
  // their guesses. A point without a guess starts from its predecessor's.
  void Locate()
  {
    if (Policy().mode != OutOfBoundsMode::FILL) {
      return;
    }
    const NimrodSource::Probe& probe = source_->DomainProbe();
    const auto n = static_cast<size_t>(NumPoints());
    std::vector<double> values(static_cast<size_t>(probe.components));
    Kokkos::View<bool*, HostMemorySpace> inside(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "nimrod_inside"), n);
    const int dmode = 0;
    for (size_t i = 0; i < n; ++i) {
      const auto rzp = RZPhi(i);
      Guess& guess = guesses_[i];
      if (guess[0] < 0 && i > 0) {
        guess = guesses_[i - 1];
      }
      int ifail = 0;
      c_eval_field_dm(rzp.data(), probe.name.c_str(), values.data(),
                      &probe.components, guess.data(), &ifail, &dmode,
                      &probe.eq);
      inside(i) = ifail == 0;
      if (ifail != 0) {
        guess = NoGuess;
      }
    }
    SetFound(inside);
  }

  std::shared_ptr<NimrodSource> source_;
  mutable std::vector<Guess> guesses_;
};

// The geometry of an initialized NIMROD dump: (R, theta, Z) points, no
// discretization PCMS can see, nimeval's point search. KnownOwners are
// ignored: the dump's mesh is not a PCMS discretization.
// OutOfBoundsMode::NEAREST_BOUNDARY is not supported.
class NimrodGeometry final : public Geometry
{
public:
  explicit NimrodGeometry(std::shared_ptr<NimrodSource> source)
    : source_(std::move(source)),
      discretization_(std::make_shared<EmptyDiscretization>()),
      system_(csys::CylindricalRThetaZ::Create())
  {
  }

  [[nodiscard]] std::shared_ptr<const Discretization> GetDiscretization()
    const noexcept override
  {
    return discretization_;
  }

  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>&
  GetCoordinateSystem() const noexcept override
  {
    return system_;
  }

protected:
  [[nodiscard]] std::shared_ptr<BoundSource> BindImpl(
    const BindRequest& request) const override
  {
    return std::make_shared<NimrodBinding>(shared_from_this(), request,
                                           source_);
  }

private:
  std::shared_ptr<NimrodSource> source_;
  std::shared_ptr<const Discretization> discretization_;
  std::shared_ptr<const CoordinateSystem> system_;
};

// Evaluates a quantity, or the gradient of a single-component quantity.
// NIMROD evaluation is not thread safe, don't call Evaluate concurrently.
class NimrodEvaluator final : public PointEvaluator<Real>
{
public:
  NimrodEvaluator(std::shared_ptr<const NimrodBinding> bound,
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
    const auto* data = dynamic_cast<const NimrodFieldData*>(&field.GetData());
    if (data == nullptr) {
      throw pcms_error(
        "NimrodGeometry: the field is not a quantity of a NIMROD source");
    }
    const auto out = values.GetValues();
    const int num_components = field.GetLayout().GetNumComponents();
    const auto& quantity = data->quantity;
    const bool vector = data->GetValueType() == FieldValueType::Vector;
    const int dmode = Operator() == DifferentialOperator::Gradient ? 1 : 0;
    const bool located = bound_->Policy().mode == OutOfBoundsMode::FILL;
    // NIMROD writes the value and, for dmode 1, (d/dR, d/dZ, (1/R) d/dphi)
    // after it.
    std::array<double, 4> scalar_and_gradient{};
    const auto host_out = HostRowsLike(out);
    for (size_t k = 0; k < bound_->found.extent(0); ++k) {
      const auto i = static_cast<size_t>(bound_->found(k));
      const auto rzp = bound_->RZPhi(i);
      // A point not yet found starts from where the previous point was.
      auto& guess = bound_->GuessOf(i);
      if (guess[0] < 0 && i > 0) {
        guess = bound_->GuessOf(i - 1);
      }
      int ifail = 0;
      double* fval = dmode == 0 ? &host_out(i, 0) : scalar_and_gradient.data();
      c_eval_field_dm(rzp.data(), quantity.name.c_str(), fval, &num_components,
                      guess.data(), &ifail, &dmode, &quantity.eq);
      if (ifail != 0) {
        guess = NimrodBinding::NoGuess;
        throw pcms_error("NimrodGeometry: query point " + std::to_string(i) +
                         (located ? " was located at bind but not at evaluation"
                                  : " is outside the dump's domain"));
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
  std::shared_ptr<const NimrodBinding> bound_;
};

// Creates the evaluators of the quantities of one component count. NIMROD is
// one representation: every single-component quantity of a dump has a
// gradient (dmode 1).
class NimrodEvaluatorFactory final
  : public TypedPointEvaluatorFactory<Real, NimrodBinding>
{
public:
  NimrodEvaluatorFactory(std::shared_ptr<const NimrodGeometry> geometry,
                         std::shared_ptr<const FieldLayout> layout)
    : TypedPointEvaluatorFactory(std::move(geometry),
                                 {DifferentialOperator::Gradient}),
      layout_(std::move(layout))
  {
  }

protected:
  std::unique_ptr<PointEvaluator<Real>> CreateEvaluator(
    std::shared_ptr<const NimrodBinding> bound,
    DifferentialOperator op) const override
  {
    CheckGradientComponents("NimrodGeometry", op, layout_->GetNumComponents());
    return std::make_unique<NimrodEvaluator>(std::move(bound), op, layout_);
  }

private:
  std::shared_ptr<const FieldLayout> layout_;
};

// The space of the quantities with `num_components` components of a dump.
std::shared_ptr<FunctionSpace> CreateNimrodSpace(
  std::shared_ptr<const NimrodGeometry> geometry, int num_components)
{
  auto layout = std::make_shared<EmptyFieldLayout>(
    geometry->GetCoordinateSystem(), num_components);
  auto evaluator =
    std::make_shared<NimrodEvaluatorFactory>(std::move(geometry), layout);
  return std::make_shared<FunctionSpace>(std::move(layout), CreateNoFieldData,
                                         std::move(evaluator));
}

} // namespace

NimrodSource::NimrodSource(const std::string& dump, const std::string& fields,
                           const std::string& case_dir)
{
  namespace fs = std::filesystem;
  if (fields.empty()) {
    throw pcms_error("fusionio::OpenNimrod: no field ids given");
  }
  const fs::path dir =
    case_dir.empty() ? fs::current_path() : fs::path(case_dir);
  std::ifstream input(dir / "nimrod.in");
  if (!input) {
    throw pcms_error("fusionio::OpenNimrod: cannot read " +
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
    throw pcms_error(
      "fusionio::OpenNimrod: only toroidal cases (geom='tor') are "
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
      throw pcms_error(
        std::string("fusionio::OpenNimrod: unsupported field id '") + id +
        "'; supported ids are bvjnptd");
    }
  }
  // The first field id's first quantity probes the domain.
  for (const auto& entry : supported) {
    if (entry.id == fields[0]) {
      probe_ = {entry.name, entry.shape.components, 2};
      break;
    }
  }
  const fs::path dump_path = fs::absolute(dump);
  if (!fs::is_regular_file(dump_path)) {
    throw pcms_error("fusionio::OpenNimrod: cannot find dump '" + dump + "'");
  }
  if (access(dir.c_str(), W_OK) != 0) {
    throw pcms_error("fusionio::OpenNimrod: NIMROD writes scratch files to the "
                     "case directory, but " +
                     dir.string() + " is not writable");
  }

  if (SourceOpen()) {
    throw pcms_error(
      "fusionio::OpenNimrod: another NIMROD source is open; NIMROD "
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

namespace
{

/// A NIMROD dump as a fusionio::Source: one geometry, and a space per
/// component count, created on first use.
class NimrodFileSource final : public Source
{
public:
  explicit NimrodFileSource(std::shared_ptr<NimrodSource> source)
    : source_(std::move(source)),
      geometry_(std::make_shared<NimrodGeometry>(source_))
  {
  }

  [[nodiscard]] std::shared_ptr<const Geometry> GetGeometry() const override
  {
    return geometry_;
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

  [[nodiscard]] Real ToroidalPeriod() const override
  {
    return 2 * std::numbers::pi;
  }

  [[nodiscard]] std::optional<Box> BoundingBox() const override
  {
    return std::nullopt;
  }

  [[nodiscard]] std::optional<Real> EvaluateSeries(Series /*series*/,
                                                   Real /*time*/) const override
  {
    return std::nullopt;
  }

  [[nodiscard]] Function<Real> CreateFunction(
    Quantity quantity, const Options& options) const override
  {
    if (options.timeslice || options.linear_scale || options.phase) {
      throw pcms_error("fusionio: NIMROD sources support only the part option");
    }
    if (options.species && quantity != Quantity::Temperature) {
      throw pcms_error(
        "fusionio: NIMROD sources support the species option only "
        "for temperature");
    }
    const std::string name = NimrodName(quantity, options.species);
    const auto found = source_->Quantities().find(name);
    if (found == source_->Quantities().end()) {
      throw pcms_error("fusionio: this NIMROD source does not provide " +
                       std::string(fusionio::Name(quantity)) + " (NIMROD's \"" +
                       name + "\"); open it with that field id");
    }
    const auto [components, vector] = found->second;
    auto& space = spaces_[components];
    if (space == nullptr) {
      space = CreateNimrodSpace(geometry_, components);
    }
    return space->CreateFunction<Real>(
      std::string(fusionio::Name(quantity)),
      std::make_unique<NimrodFieldData>(
        NimrodQuantity{source_, name,
                       NimevalPart(options.part.value_or(Part::Total))},
        QuantityBasis(vector)));
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
    throw pcms_error("fusionio: NIMROD sources do not provide " +
                     std::string(fusionio::Name(quantity)));
  }

  std::shared_ptr<NimrodSource> source_;
  std::shared_ptr<const NimrodGeometry> geometry_;
  mutable std::map<int, std::shared_ptr<FunctionSpace>> spaces_;
};

} // namespace

} // namespace fusionio::detail

namespace fusionio
{

std::shared_ptr<Source> OpenNimrod(const std::string& dump,
                                   const std::string& fields,
                                   const std::string& case_dir)
{
  return std::make_shared<detail::NimrodFileSource>(
    detail::OpenNimrodSource(dump, fields, case_dir));
}

} // namespace fusionio
