#include "pcms/field/function_space/nimrod.hpp"
#include "pcms/field/function_space/nimeval.hpp"
#include "pcms/field/coordinate_systems/cylindrical.hpp"
#include "pcms/utility/arrays.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/profile.h"
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <ranges>
#include <regex>
#include <sstream>

namespace pcms
{

/// The dump nimeval has initialized and the quantities it can evaluate.
/// nimeval stops the whole process on bad input, so everything passed to it is
/// checked first.
class NimrodSource
{
public:
  struct Quantity
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
  [[nodiscard]] const std::map<std::string, Quantity>& Quantities()
    const noexcept
  {
    return quantities_;
  }

private:
  std::map<std::string, Quantity> quantities_;
};

namespace
{

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

// Names one quantity of a source. PCMS holds none of its values, so it
// provides no DOFAccess.
class NimrodFieldData final : public FieldData<Real>
{
public:
  NimrodFieldData(std::shared_ptr<NimrodSource> source, std::string quantity,
                  int eq, ValueBasis basis)
    : source(std::move(source)),
      quantity(std::move(quantity)),
      eq(eq),
      basis_(std::move(basis))
  {
  }

  [[nodiscard]]
  FieldValueType GetValueType() const override
  {
    return ValueTypeOfRank(basis_.Rank());
  }

  [[nodiscard]]
  const ValueBasis& GetValueBasis() const override
  {
    return basis_;
  }

  std::shared_ptr<NimrodSource> source;
  std::string quantity;
  // nimeval's equilibrium/perturbation selector
  int eq;

private:
  ValueBasis basis_;
};

// The query points and each point's search guess, shared by every evaluator
// created from this bound source.
class NimrodBoundSource final : public BoundSource
{
public:
  explicit NimrodBoundSource(const EvaluationRequest& request)
    : BoundSource(request)
  {
    if (request.policy.mode != OutOfBoundsMode::ERROR) {
      throw pcms_error("NimrodFunctionSpace: only OutOfBoundsMode::ERROR is "
                       "supported");
    }
    const auto coords = request.coords.GetValues();
    // Copy to a LayoutRight host view so each point's coordinates are
    // contiguous.
    Kokkos::View<Real**, DeviceMemorySpace> staged(
      "nimrod_points", coords.extent(0), coords.extent(1));
    CopyDeviceRank2ViewToRank2View(MakeRank2View(staged), coords);
    points =
      decltype(points)("nimrod_points", coords.extent(0), coords.extent(1));
    DeepCopyMismatchLayouts(points, staged);
    guesses.assign(points.extent(0), {-1.0, -1.0});
  }

  Kokkos::View<Real**, Kokkos::LayoutRight, HostMemorySpace> points;
  // Each point's logical coordinates from its last search, which NIMROD
  // starts the next search from; (-1, -1) searches from scratch.
  mutable std::vector<std::array<double, 2>> guesses;
};

// NIMROD evaluation is not thread safe, don't call Evaluate concurrently
class NimrodEvaluator final : public PointEvaluator<Real>
{
public:
  NimrodEvaluator(std::shared_ptr<NimrodSource> source,
                  std::shared_ptr<const NimrodBoundSource> bound)
    : source_(std::move(source)), bound_(std::move(bound))
  {
  }

  void Evaluate(const Field<Real>& field,
                ValueView<Real, DeviceMemorySpace> values) const override
  {
    PCMS_FUNCTION_TIMER;
    CheckEvaluateWriteTag("NimrodFunctionSpace", field, values);
    const auto* data = dynamic_cast<const NimrodFieldData*>(&field.GetData());
    if (data == nullptr || data->source != source_) {
      throw pcms_error(
        "NimrodFunctionSpace: field is not a quantity of this source");
    }
    const auto out = values.GetValues();
    const auto& points = bound_->points;
    auto& guesses = bound_->guesses;
    const int num_components = field.GetLayout().GetNumComponents();
    if (out.extent(0) != points.extent(0) ||
        out.extent(1) != static_cast<size_t>(num_components)) {
      throw pcms_error("NimrodFunctionSpace: output is shaped [" +
                       std::to_string(out.extent(0)) + "][" +
                       std::to_string(out.extent(1)) + "] but expected [" +
                       std::to_string(points.extent(0)) + "][" +
                       std::to_string(num_components) + "]");
    }
    const bool vector = data->GetValueType() == FieldValueType::Vector;
    const int dmode = 0;
    Kokkos::View<Real**, Kokkos::LayoutRight, HostMemorySpace> host_out(
      "nimrod_values", out.extent(0), out.extent(1));
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
      c_eval_field_dm(rzp.data(), data->quantity.c_str(), &host_out(i, 0),
                      &num_components, guess.data(), &ifail, &dmode, &data->eq);
      if (ifail != 0) {
        guess = {-1.0, -1.0};
        throw pcms_error("NimrodFunctionSpace: query point " +
                         std::to_string(i) + " is outside the dump's domain");
      }
      if (vector) {
        // (v_R, v_Z, v_phi) -> (v_R, v_theta, v_Z)
        const Real v_z = host_out(i, 1);
        host_out(i, 1) = -host_out(i, 2);
        host_out(i, 2) = v_z;
      }
    }
    Kokkos::View<Real**, DeviceMemorySpace> device_out(
      "nimrod_values", out.extent(0), out.extent(1));
    DeepCopyMismatchLayouts(device_out, host_out);
    CopyDeviceRank2ViewToRank2View(out, MakeConstRank2View(device_out));
  }

  Kokkos::View<const LO*, DeviceMemorySpace> FilledPoints() const override
  {
    return {};
  }

private:
  std::shared_ptr<NimrodSource> source_;
  std::shared_ptr<const NimrodBoundSource> bound_;
};

} // namespace

NimrodSource::NimrodSource(const std::string& dump, const std::string& fields,
                           const std::string& case_dir)
{
  namespace fs = std::filesystem;
  if (fields.empty()) {
    throw pcms_error("OpenNimrodSource: no field ids given");
  }
  const fs::path dir =
    case_dir.empty() ? fs::current_path() : fs::path(case_dir);
  std::ifstream input(dir / "nimrod.in");
  if (!input) {
    throw pcms_error("OpenNimrodSource: cannot read " +
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
    throw pcms_error("OpenNimrodSource: only toroidal cases (geom='tor') are "
                     "supported");
  }
  const int nisp = std::stoi(NamelistValue(text, "nisp").value_or("0"));
  const int ds_nqty = std::stoi(NamelistValue(text, "ds_nqty").value_or("1"));

  // The quantities each field id initializes, as in nimpy.
  const struct
  {
    const char* name;
    char id;
    Quantity quantity;
  } supported[] = {{"b", 'b', {3, true}},   {"v", 'v', {3, true}},
                   {"j", 'j', {3, true}},   {"n", 'n', {nisp + 1, false}},
                   {"p", 'p', {1, false}},  {"ti", 't', {1, false}},
                   {"te", 't', {1, false}}, {"d", 'd', {ds_nqty, false}}};
  for (const char id : fields) {
    bool known = false;
    for (const auto& entry : supported) {
      if (entry.id == id) {
        quantities_.emplace(entry.name, entry.quantity);
        known = true;
      }
    }
    if (!known) {
      throw pcms_error(std::string("OpenNimrodSource: unsupported field id '") +
                       id + "'; supported ids are bvjnptd");
    }
  }
  const fs::path dump_path = fs::absolute(dump);
  if (!fs::is_regular_file(dump_path)) {
    throw pcms_error("OpenNimrodSource: cannot find dump '" + dump + "'");
  }
  if (access(dir.c_str(), W_OK) != 0) {
    throw pcms_error("OpenNimrodSource: NIMROD writes scratch files to the "
                     "case directory, but " +
                     dir.string() + " is not writable");
  }

  if (SourceOpen()) {
    throw pcms_error("OpenNimrodSource: another NIMROD source is open; NIMROD "
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
  : source_(std::move(source)),
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
    std::string available;
    for (const auto& known : AvailableQuantities(*source_)) {
      available += (available.empty() ? "" : ", ") + known;
    }
    throw pcms_error(
      "NimrodFunctionSpace: \"" + quantity +
      "\" is not a quantity of this source; it has: " + available);
  }
  const auto [components, vector] = found->second;
  if (components != layout_->GetNumComponents()) {
    throw pcms_error("NimrodFunctionSpace: \"" + quantity + "\" has " +
                     std::to_string(components) +
                     " components but the space holds " +
                     std::to_string(layout_->GetNumComponents()));
  }
  const int eqs[] = {2, 0, 1};
  ValueBasis basis = vector
                       ? ValueBasis{csys::CylindricalRThetaZ::Create(),
                                    ComponentScaling::Physical, values::Vector}
                       : ValueBasis{};
  return FunctionSpace::CreateFunction<Real>(
    std::move(name), std::make_unique<NimrodFieldData>(
                       source_, quantity, eqs[static_cast<int>(options.part)],
                       std::move(basis)));
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

std::shared_ptr<BoundSource> NimrodFunctionSpace::BindImpl(
  const EvaluationRequest& request) const
{
  return std::make_shared<NimrodBoundSource>(request);
}

PointEvaluatorVariant NimrodFunctionSpace::CreatePointEvaluatorImpl(
  Type value_type, std::shared_ptr<const BoundSource> bound,
  int derivative_order) const
{
  if (value_type != Type::Real) {
    throw pcms_error("NimrodFunctionSpace: only Real evaluation is "
                     "supported");
  }
  if (derivative_order != 0) {
    throw pcms_error("NimrodFunctionSpace: derivatives are not supported");
  }
  detail::BoundSourceAs<NimrodBoundSource>(*bound, "NimrodFunctionSpace");
  return std::unique_ptr<PointEvaluator<Real>>(
    std::make_unique<NimrodEvaluator>(
      source_, std::static_pointer_cast<const NimrodBoundSource>(bound)));
}

} // namespace pcms
