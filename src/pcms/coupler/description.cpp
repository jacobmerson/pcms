#include "description.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/coordinate_systems/cartesian.hpp"
#include "pcms/utility/coordinate_systems/cylindrical.hpp"
#include "pcms/utility/detail/coordinate_system.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <system_error>

namespace pcms::detail
{

namespace
{

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
  if (description["pcms_schema"] != DescriptionSchema) {
    throw pcms_error("Setup: the description is not of schema " +
                     std::string(DescriptionSchema));
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

void DescribeChart(const CoordinateSystem& chart, const std::string& prefix,
                   Description& description)
{
  description[prefix + ".kind"] = std::string(chart.Kind());
  description[prefix + ".dimension"] = std::to_string(chart.Dimension());
}

std::shared_ptr<const CoordinateSystem> ChartFromDescription(
  const Description& description, const std::string& prefix)
{
  const auto& kind = Require(description, prefix + ".kind");
  if (kind == "cartesian") {
    return csys::CreateCartesian(static_cast<int>(
      ParseInteger(Require(description, prefix + ".dimension"))));
  }
  if (kind == "cartesian (dimension deferred)") {
    return DeferredCartesian();
  }
  if (kind == "cylindrical_rz") {
    return csys::CreateCylindricalRZ();
  }
  if (kind == "cylindrical_rthetaz") {
    return csys::CreateCylindricalRThetaZ();
  }
  throw pcms_error("Setup: unknown chart kind '" + kind + "'");
}

void DescribeValueBasis(const ValueBasis& basis, const std::string& prefix,
                        Description& description)
{
  std::string variance;
  for (int i = 0; i < basis.variance.Rank(); ++i) {
    variance += basis.variance[i] == Variance::Covariant ? 'c' : 'v';
  }
  description[prefix + ".variance"] = variance;
  description[prefix + ".scaling"] =
    basis.scaling == ComponentScaling::Physical ? "physical" : "natural";
  description[prefix + ".parity"] =
    basis.parity == Parity::Even ? "even" : "odd";
  if (basis.system != nullptr) {
    DescribeChart(*basis.system, prefix + ".chart", description);
  }
}

ValueBasis ValueBasisFromDescription(const Description& description,
                                     const std::string& prefix)
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
  basis.scaling = Require(description, prefix + ".scaling") == "physical"
                    ? ComponentScaling::Physical
                    : ComponentScaling::Natural;
  basis.parity = Require(description, prefix + ".parity") == "even"
                   ? Parity::Even
                   : Parity::Odd;
  if (description.count(prefix + ".chart.kind") != 0) {
    basis.system = ChartFromDescription(description, prefix + ".chart");
  }
  return basis;
}

} // namespace pcms::detail
