#include "description.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/coordinate_systems/cartesian.hpp"
#include "pcms/utility/coordinate_systems/cylindrical.hpp"
#include "pcms/utility/coordinate_systems/parameter.hpp"
#include "pcms/utility/detail/coordinate_system.hpp"
#include "pcms/version.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <system_error>

namespace pcms::detail
{

namespace
{

// indexed by ValueKind
constexpr std::array<std::string_view, 3> ValueKindNames{
  "tensor", "coordinates", "array"};

template <typename T>
T ParseNumber(std::string_view text)
{
  T value = 0;
  const auto result =
    std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
    throw pcms_error("Setup: '" + std::string(text) + "' is not a number");
  }
  return value;
}

} // namespace

std::string SerializeDescription(const Description& description)
{
  std::string text;
  for (const auto& [key, value] : description) {
    if (key.find_first_of("=\n") != std::string::npos ||
        value.find('\n') != std::string::npos) {
      throw pcms_error("Setup: invalid description entry '" + key + "'");
    }
    text += key + '=' + value + '\n';
  }
  return text;
}

Description CreateDescription()
{
  return {{"pcms_schema", std::string(DescriptionSchema)},
          {"pcms_version", pcms_version()}};
}

Description ParseDescription(std::string_view text)
{
  Description description;
  while (!text.empty()) {
    const auto end = std::min(text.find('\n'), text.size());
    const auto line = text.substr(0, end);
    text.remove_prefix(std::min(end + 1, text.size()));
    const auto equals = line.find('=');
    description.emplace(line.substr(0, equals), line.substr(equals + 1));
  }
  if (description["pcms_schema"] != DescriptionSchema ||
      description["pcms_version"] != pcms_version()) {
    throw pcms_error(
      "Setup: the application's description is of schema '" +
      description["pcms_schema"] + "' from PCMS '" +
      description["pcms_version"] + "', the server reads schema " +
      std::string(DescriptionSchema) + " from PCMS '" + pcms_version() + "'");
  }
  return description;
}

const std::string& Require(const Description& description,
                           const std::string& key)
{
  const auto it = description.find(key);
  if (it == description.end()) {
    throw pcms_error("Setup: the description has no '" + key + "'");
  }
  return it->second;
}

std::string FormatReal(Real value)
{
  std::array<char, 32> buffer{};
  const auto end =
    std::to_chars(buffer.data(), buffer.data() + buffer.size(), value).ptr;
  return {buffer.data(), end};
}

Real ParseReal(std::string_view text)
{
  return ParseNumber<Real>(text);
}

GO ParseInteger(std::string_view text)
{
  return ParseNumber<GO>(text);
}

std::size_t DescribeSystem(
  const std::shared_ptr<const CoordinateSystem>& system,
  DescribedSystems& systems, Description& description)
{
  const auto n = static_cast<std::size_t>(
    std::find(systems.begin(), systems.end(), system) - systems.begin());
  if (n < systems.size()) {
    return n;
  }
  systems.push_back(system);
  const auto prefix = "system." + std::to_string(n);
  description[prefix + ".kind"] = std::string(system->Kind());
  description[prefix + ".dimension"] = std::to_string(system->Dimension());
  if (!system->HasMetric()) {
    description[prefix + ".parameter"] = "1";
  }
  for (int d = 0; d < system->Dimension(); ++d) {
    if (const auto period = system->Period(d)) {
      description[prefix + ".period." + std::to_string(d)] =
        FormatReal(*period);
    }
  }
  description["systems"] = std::to_string(systems.size());
  return n;
}

DescribedSystems SystemsFromDescription(const Description& description)
{
  DescribedSystems systems;
  const auto it = description.find("systems");
  const auto count = it == description.end() ? 0 : ParseInteger(it->second);
  for (GO n = 0; n < count; ++n) {
    const auto prefix = "system." + std::to_string(n);
    const auto& kind = Require(description, prefix + ".kind");
    const auto dimension = static_cast<int>(
      ParseInteger(Require(description, prefix + ".dimension")));
    if (description.count(prefix + ".parameter") != 0) {
      std::vector<std::optional<Real>> periods(dimension);
      for (int d = 0; d < dimension; ++d) {
        const auto period =
          description.find(prefix + ".period." + std::to_string(d));
        if (period != description.end()) {
          periods[d] = ParseReal(period->second);
        }
      }
      systems.push_back(csys::CreateParameterSystem(kind, dimension, periods));
    } else if (kind == "cartesian") {
      systems.push_back(csys::CreateCartesian(dimension));
    } else if (kind == "cartesian (dimension deferred)") {
      systems.push_back(DeferredCartesian());
    } else if (kind == "cylindrical_rz") {
      systems.push_back(csys::CreateCylindricalRZ());
    } else if (kind == "cylindrical_rthetaz") {
      systems.push_back(csys::CreateCylindricalRThetaZ());
    } else {
      throw pcms_error("Setup: unknown coordinate system kind '" + kind + "'");
    }
  }
  return systems;
}

void DescribeValueBasis(const ValueBasis& basis, const std::string& prefix,
                        DescribedSystems& systems, Description& description)
{
  std::string variance;
  for (int i = 0; i < basis.variance.Rank(); ++i) {
    variance += basis.variance[i] == Variance::Covariant ? 'c' : 'v';
  }
  description[prefix + ".variance"] = variance;
  if (basis.scaling) {
    description[prefix + ".scaling"] =
      *basis.scaling == ComponentScaling::Physical ? "physical" : "natural";
  }
  description[prefix + ".parity"] =
    basis.parity == Parity::Even ? "even" : "odd";
  description[prefix + ".kind"] =
    ValueKindNames[static_cast<size_t>(basis.kind)];
  if (basis.system != nullptr) {
    description[prefix + ".system"] =
      std::to_string(DescribeSystem(basis.system, systems, description));
  }
}

ValueBasis ValueBasisFromDescription(const Description& description,
                                     const std::string& prefix,
                                     const DescribedSystems& systems)
{
  std::array<Variance, VarianceSignature::max_rank> v{};
  const auto& variance = Require(description, prefix + ".variance");
  for (size_t i = 0; i < variance.size() && i < v.size(); ++i) {
    v[i] = variance[i] == 'c' ? Variance::Covariant : Variance::Contravariant;
  }
  ValueBasis basis;
  switch (variance.size()) {
    case 0: break;
    case 1: basis.variance = {v[0]}; break;
    case 2: basis.variance = {v[0], v[1]}; break;
    case 3: basis.variance = {v[0], v[1], v[2]}; break;
    default: basis.variance = {v[0], v[1], v[2], v[3]};
  }
  if (const auto it = description.find(prefix + ".scaling");
      it != description.end()) {
    basis.scaling = it->second == "physical" ? ComponentScaling::Physical
                                             : ComponentScaling::Natural;
  }
  basis.parity = Require(description, prefix + ".parity") == "even"
                   ? Parity::Even
                   : Parity::Odd;
  const auto& kind = Require(description, prefix + ".kind");
  basis.kind = static_cast<ValueKind>(
    std::find(ValueKindNames.begin(), ValueKindNames.end(), kind) -
    ValueKindNames.begin());
  if (const auto it = description.find(prefix + ".system");
      it != description.end()) {
    basis.system = systems.at(static_cast<size_t>(ParseInteger(it->second)));
  }
  return basis;
}

void DescribeMetadata(const FieldMetadata& metadata, const std::string& prefix,
                      Description& description)
{
  description[prefix + ".name"] = metadata.name;
  if (metadata.units) {
    description[prefix + ".units"] = metadata.units->ToString();
  }
  if (metadata.extent) {
    description[prefix + ".extent"] =
      *metadata.extent == Extent::Intensive ? "intensive" : "extensive";
  }
}

FieldMetadata MetadataFromDescription(const Description& description,
                                      const std::string& prefix)
{
  FieldMetadata metadata{.name = Require(description, prefix + ".name")};
  if (const auto it = description.find(prefix + ".units");
      it != description.end()) {
    metadata.units = Units::Parse(it->second);
  }
  if (const auto it = description.find(prefix + ".extent");
      it != description.end()) {
    metadata.extent =
      it->second == "intensive" ? Extent::Intensive : Extent::Extensive;
  }
  return metadata;
}

} // namespace pcms::detail
