#include "description.hpp"
#include "pcms/backends/uniform_grid/geometry.hpp"
#include "pcms/backends/uniform_grid/layout.hpp"
#include "pcms/backends/uniform_grid/uniform_grid_geometry.hpp"
#include "pcms/field/embedding.hpp"
#include "pcms/field/function_spaces.hpp"
#include <bit>

namespace pcms::detail
{

namespace
{

std::string AxisKey(const std::string& prefix, const char* name, unsigned d)
{
  return prefix + "." + name + "." + std::to_string(d);
}

template <unsigned Dim>
std::shared_ptr<const Geometry> RebuildGrid(const Description& description,
                                            const std::string& prefix,
                                            const DescribedSystems& systems)
{
  UniformGrid<Dim> grid{};
  for (unsigned d = 0; d < Dim; ++d) {
    grid.bot_left[d] =
      ParseReal(Require(description, AxisKey(prefix, "bot_left", d)));
    grid.edge_length[d] =
      ParseReal(Require(description, AxisKey(prefix, "edge_length", d)));
    grid.divisions[d] = static_cast<LO>(
      ParseInteger(Require(description, AxisKey(prefix, "divisions", d))));
  }
  return uniform_grid::CreateGeometry(
    grid, systems.at(static_cast<size_t>(
            ParseInteger(Require(description, prefix + ".system")))));
}

} // namespace

std::optional<std::vector<std::vector<GO>>> DescribeUniformGrid(
  const Geometry& geometry, const std::string& prefix,
  DescribedSystems& systems, Description& description)
{
  const bool described = uniform_grid::VisitUniformGrid(
    geometry.shared_from_this(),
    [&]<unsigned Dim>(
      const std::shared_ptr<const uniform_grid::UniformGridGeometry<Dim>>&
        grid_geometry) {
      const auto& grid = grid_geometry->GetGrid();
      description[prefix + ".dimension"] = std::to_string(Dim);
      for (unsigned d = 0; d < Dim; ++d) {
        description[AxisKey(prefix, "bot_left", d)] =
          FormatReal(grid.bot_left[d]);
        description[AxisKey(prefix, "edge_length", d)] =
          FormatReal(grid.edge_length[d]);
        description[AxisKey(prefix, "divisions", d)] =
          std::to_string(grid.divisions[d]);
      }
      return true;
    });
  if (!described) {
    return std::nullopt;
  }
  description[prefix + ".kind"] = "uniform_grid";
  description[prefix + ".system"] = std::to_string(
    DescribeSystem(geometry.GetCoordinateSystem(), systems, description));
  return std::vector<std::vector<GO>>{};
}

std::shared_ptr<const Geometry> RebuildUniformGrid(
  const Description& description, const std::string& prefix,
  const DescribedSystems& systems)
{
  const auto& dimension = Require(description, prefix + ".dimension");
  if (dimension == "1") {
    return RebuildGrid<1>(description, prefix, systems);
  }
  if (dimension == "2") {
    return RebuildGrid<2>(description, prefix, systems);
  }
  return RebuildGrid<3>(description, prefix, systems);
}

int LagrangeOrder(const FieldLayout& layout)
{
  if (const auto* grid =
        dynamic_cast<const uniform_grid::UniformGridFieldLayout<1>*>(&layout)) {
    return grid->GetOrder();
  }
  if (const auto* grid =
        dynamic_cast<const uniform_grid::UniformGridFieldLayout<2>*>(&layout)) {
    return grid->GetOrder();
  }
  if (const auto* grid =
        dynamic_cast<const uniform_grid::UniformGridFieldLayout<3>*>(&layout)) {
    return grid->GetOrder();
  }
  if (layout.GetDOFHolderEntityDim() != 0) {
    throw pcms_error("Application::Setup: only uniform-grid Lagrange spaces of "
                     "order 0 and 1 and Omega_h ones of order 1 can be "
                     "described");
  }
  return 1;
}

void DescribeLagrangeFunction(const Function<Real>& function,
                              const std::string& key, DescribedSystems& systems,
                              Description& description)
{
  const auto& layout = *function.GetSpace().GetLayout();
  description[key + ".order"] = std::to_string(LagrangeOrder(layout));
  description[key + ".components"] = std::to_string(layout.GetNumComponents());
  DescribeMetadata(function.GetMetadata(), key, description);
  DescribeValueBasis(function.GetData().GetValueBasis(), key + ".basis",
                     systems, description);
}

std::shared_ptr<const Function<Real>> RebuildLagrangeFunction(
  const Description& description, const std::string& key,
  std::shared_ptr<const Geometry> geometry, const DescribedSystems& systems,
  std::span<const Real>& values)
{
  const auto components =
    static_cast<int>(ParseInteger(Require(description, key + ".components")));
  const auto space = CreateLagrangeSpace(
    std::move(geometry),
    static_cast<int>(ParseInteger(Require(description, key + ".order"))),
    components);
  auto function = std::make_shared<Function<Real>>(space->CreateFunction<Real>(
    MetadataFromDescription(description, key),
    ValueBasisFromDescription(description, key + ".basis", systems)));
  const auto holders =
    static_cast<size_t>(space->GetLayout()->GetNumDofHolder());
  if (values.size() < holders * components) {
    throw pcms_error("Application::Setup: too few values for '" + key + "'");
  }
  function->SetDOFHolderDataHost(ValueView<const Real, HostMemorySpace>(
    function->GetData().GetValueBasis(),
    Rank2View<const Real, HostMemorySpace>(values.data(), holders,
                                           components)));
  values = values.subspan(holders * components);
  return function;
}

std::optional<std::vector<std::vector<GO>>> DescribeEmbedded(
  const Geometry& geometry, const std::string& prefix,
  DescribedSystems& systems, Description& description, MPI_Comm comm,
  std::size_t num_servers)
{
  const auto* embedded = dynamic_cast<const EmbeddedGeometry*>(&geometry);
  if (embedded == nullptr) {
    return std::nullopt;
  }
  if (embedded->GetParent() != nullptr) {
    throw pcms_error("Application::Setup: an embedded geometry linked into a "
                     "parent cannot be described yet");
  }
  if (!DescribeUniformGrid(*embedded->GetParameterGeometry(),
                           prefix + ".parameter", systems, description)) {
    throw pcms_error("Application::Setup: only embedded geometries over a "
                     "uniform grid can be described");
  }
  description[prefix + ".kind"] = "embedded";
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  // the values of the position, the metric and the volume, from the first rank
  std::vector<GO> values;
  const auto describe = [&](const Function<Real>& function,
                            const std::string& key) {
    DescribeLagrangeFunction(function, key, systems, description);
    if (rank == 0) {
      const auto host = function.GetDOFHolderDataHost().GetValues();
      for (size_t h = 0; h < host.extent(0); ++h) {
        for (size_t c = 0; c < host.extent(1); ++c) {
          values.push_back(std::bit_cast<GO>(host(h, c)));
        }
      }
    }
  };
  describe(*embedded->GetPosition(), prefix + ".position");
  if (embedded->GetMetric() != nullptr) {
    describe(*embedded->GetMetric(), prefix + ".metric");
  }
  if (embedded->GetVolume() != nullptr) {
    describe(*embedded->GetVolume(), prefix + ".volume");
  }
  return std::vector<std::vector<GO>>(num_servers, values);
}

std::shared_ptr<const Geometry> RebuildEmbedded(
  const Description& description, const std::string& prefix,
  const DescribedSystems& systems, const std::vector<std::vector<GO>>& blocks)
{
  const auto parameter =
    RebuildUniformGrid(description, prefix + ".parameter", systems);
  std::vector<Real> values;
  for (const auto& block : blocks) {
    for (const GO value : block) {
      values.push_back(std::bit_cast<Real>(value));
    }
  }
  std::span<const Real> rest(values);
  const auto rebuild = [&](const std::string& key) {
    return RebuildLagrangeFunction(description, key, parameter, systems, rest);
  };
  EmbeddingOptions options;
  auto position = rebuild(prefix + ".position");
  if (description.count(prefix + ".metric.order") != 0) {
    options.metric = rebuild(prefix + ".metric");
  }
  if (description.count(prefix + ".volume.order") != 0) {
    options.volume = rebuild(prefix + ".volume");
  }
  return CreateEmbeddedGeometry(parameter, std::move(position),
                                std::move(options));
}

} // namespace pcms::detail
