#include "description.hpp"
#include "partition.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/coordinate_systems/cartesian.hpp"
#include "pcms/utility/coordinate_systems/cylindrical.hpp"
#include "pcms/utility/detail/coordinate_system.hpp"
#include <array>
#include <charconv>
#include <system_error>

namespace pcms::detail
{

std::string SerializeDescription(const Description& description)
{
  std::string text;
  for (const auto& [key, value] : description) {
    if (key.find_first_of("=\n") != std::string::npos ||
        value.find('\n') != std::string::npos) {
      throw pcms_error("Setup: the description entry '" + key +
                       "' contains '=' in its key or a line break");
    }
    text += key;
    text += '=';
    text += value;
    text += '\n';
  }
  return text;
}

Description ParseDescription(std::string_view text)
{
  Description description;
  while (!text.empty()) {
    const auto end = text.find('\n');
    const auto line = text.substr(0, end);
    text =
      end == std::string_view::npos ? std::string_view{} : text.substr(end + 1);
    if (line.empty()) {
      continue;
    }
    const auto equals = line.find('=');
    if (equals == std::string_view::npos) {
      throw pcms_error("Setup: malformed description line '" +
                       std::string(line) + "'");
    }
    description.emplace(std::string(line.substr(0, equals)),
                        std::string(line.substr(equals + 1)));
  }
  const auto it = description.find("pcms_schema");
  if (it == description.end() || it->second != DescriptionSchema) {
    throw pcms_error("Setup: the application's description has schema '" +
                     (it == description.end() ? std::string{} : it->second) +
                     "', this server reads schema '" +
                     std::string(DescriptionSchema) + "'");
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
  // shortest representation that parses back to the same value
  std::array<char, 32> buffer{};
  const auto result =
    std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
  PCMS_ALWAYS_ASSERT(result.ec == std::errc{});
  return {buffer.data(), result.ptr};
}

Real ParseReal(std::string_view text)
{
  Real value = 0;
  const auto result =
    std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
    throw pcms_error("Setup: '" + std::string(text) + "' is not a number");
  }
  return value;
}

GO ParseInteger(std::string_view text)
{
  GO value = 0;
  const auto result =
    std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
    throw pcms_error("Setup: '" + std::string(text) + "' is not an integer");
  }
  return value;
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
  const auto dimension =
    static_cast<int>(ParseInteger(Require(description, prefix + ".dimension")));
  std::shared_ptr<const CoordinateSystem> chart;
  if (kind == "cartesian") {
    chart = csys::CreateCartesian(dimension);
  } else if (kind == "cartesian (dimension deferred)") {
    chart = DeferredCartesian();
  } else if (kind == "cylindrical_rz") {
    chart = csys::CreateCylindricalRZ();
  } else if (kind == "cylindrical_rthetaz") {
    chart = csys::CreateCylindricalRThetaZ();
  } else {
    throw pcms_error("Setup: unknown chart kind '" + kind + "'");
  }
  if (chart->Dimension() != dimension) {
    throw pcms_error("Setup: the chart '" + kind + "' has dimension " +
                     std::to_string(chart->Dimension()) + ", not " +
                     std::to_string(dimension));
  }
  return chart;
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
  ValueBasis basis;
  const auto& variance = Require(description, prefix + ".variance");
  if (variance.size() > static_cast<size_t>(VarianceSignature::max_rank)) {
    throw pcms_error("Setup: the value rank is too high");
  }
  // VarianceSignature is built from an initializer list only
  std::array<Variance, VarianceSignature::max_rank> entries{};
  for (size_t i = 0; i < variance.size(); ++i) {
    if (variance[i] != 'v' && variance[i] != 'c') {
      throw pcms_error("Setup: unknown index variance '" + variance + "'");
    }
    entries[i] =
      variance[i] == 'v' ? Variance::Contravariant : Variance::Covariant;
  }
  VarianceSignature signature;
  switch (variance.size()) {
    case 0: break;
    case 1: signature = {entries[0]}; break;
    case 2: signature = {entries[0], entries[1]}; break;
    case 3: signature = {entries[0], entries[1], entries[2]}; break;
    default: signature = {entries[0], entries[1], entries[2], entries[3]};
  }
  basis.variance = signature;
  const auto& scaling = Require(description, prefix + ".scaling");
  basis.scaling = scaling == "physical" ? ComponentScaling::Physical
                                        : ComponentScaling::Natural;
  const auto& parity = Require(description, prefix + ".parity");
  basis.parity = parity == "even" ? Parity::Even : Parity::Odd;
  if (description.count(prefix + ".chart.kind") != 0) {
    basis.system = ChartFromDescription(description, prefix + ".chart");
  }
  return basis;
}

Kokkos::View<Real**, Kokkos::LayoutRight, HostMemorySpace>
HostDOFHolderCoordinates(const FieldLayout& layout)
{
  const auto coords = layout.GetDOFHolderCoordinates().GetValues();
  Kokkos::View<Real**, Kokkos::LayoutRight, DeviceMemorySpace> staged(
    "dof_holder_coords", coords.extent(0), coords.extent(1));
  Kokkos::parallel_for(
    "copy_dof_holder_coords",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0,
                                                            coords.extent(0)),
    KOKKOS_LAMBDA(int i) {
      for (size_t d = 0; d < coords.extent(1); ++d) {
        staged(i, d) = coords(i, d);
      }
    });
  return Kokkos::create_mirror_view_and_copy(HostMemorySpace(), staged);
}

int RankOf(const redev::Partition& partition, LO class_dim, LO class_id,
           const Real* coord, int dim)
{
  std::array<Real, 3> point{};
  for (int d = 0; d < dim; ++d) {
    point[d] = coord[d];
  }
  return std::visit(coupler::GetRank{class_id, class_dim, point}, partition);
}

redev::Partition RoutingPartition(const redev::Partition& partition, int dim)
{
  if (const auto* rcb = std::get_if<redev::RCBPtn>(&partition)) {
    auto ranks = rcb->GetRanks();
    auto cuts = rcb->GetCuts();
    return redev::RCBPtn(dim, ranks, cuts);
  }
  return partition;
}

const std::vector<const SpaceKind*>& SpaceKinds()
{
  static const std::vector<const SpaceKind*> kinds = {
    &UniformGridSpaceKind(),
#ifdef PCMS_ENABLE_OMEGA_H
    &OmegaHSpaceKind(),
#endif
  };
  return kinds;
}

const SpaceKind& FindSpaceKind(const FunctionSpace& space)
{
  for (const SpaceKind* kind : SpaceKinds()) {
    if (kind->Describes(space)) {
      return *kind;
    }
  }
  throw pcms_error(
    "Setup: no kind of space describes this function's space (supported: "
    "uniform-grid Lagrange of order 0 and 1, Omega_h Lagrange of order 1)");
}

const SpaceKind& FindSpaceKind(std::string_view name)
{
  for (const SpaceKind* kind : SpaceKinds()) {
    if (kind->Name() == name) {
      return *kind;
    }
  }
  throw pcms_error("Setup: the server does not know the space kind '" +
                   std::string(name) + "'");
}

} // namespace pcms::detail
