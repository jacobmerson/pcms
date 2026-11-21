#include "description.hpp"
#include "pcms/backends/uniform_grid/geometry.hpp"
#include "pcms/backends/uniform_grid/uniform_grid_geometry.hpp"

namespace pcms::detail
{

namespace
{

std::string AxisKey(const std::string& prefix, const char* name, unsigned d)
{
  return prefix + "." + name + "." + std::to_string(d);
}

template <unsigned Dim>
bool DescribeGrid(const Geometry& geometry, const std::string& prefix,
                  Description& description)
{
  const auto* grid_geometry =
    dynamic_cast<const uniform_grid::UniformGridGeometry<Dim>*>(&geometry);
  if (grid_geometry == nullptr) {
    return false;
  }
  const auto& grid = grid_geometry->GetGrid();
  description[prefix + ".dimension"] = std::to_string(Dim);
  for (unsigned d = 0; d < Dim; ++d) {
    description[AxisKey(prefix, "bot_left", d)] = FormatReal(grid.bot_left[d]);
    description[AxisKey(prefix, "edge_length", d)] =
      FormatReal(grid.edge_length[d]);
    description[AxisKey(prefix, "divisions", d)] =
      std::to_string(grid.divisions[d]);
  }
  return true;
}

template <unsigned Dim>
std::shared_ptr<const Geometry> RebuildGrid(const Description& description,
                                            const std::string& prefix)
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
    grid, ChartFromDescription(description, prefix + ".chart"));
}

} // namespace

std::optional<std::vector<std::vector<GO>>> DescribeUniformGrid(
  const Geometry& geometry, const std::string& prefix, Description& description)
{
  if (!DescribeGrid<2>(geometry, prefix, description) &&
      !DescribeGrid<3>(geometry, prefix, description)) {
    return std::nullopt;
  }
  description[prefix + ".kind"] = "uniform_grid";
  DescribeChart(*geometry.GetCoordinateSystem(), prefix + ".chart",
                description);
  return std::vector<std::vector<GO>>{};
}

std::shared_ptr<const Geometry> RebuildUniformGrid(
  const Description& description, const std::string& prefix)
{
  if (Require(description, prefix + ".dimension") == "2") {
    return RebuildGrid<2>(description, prefix);
  }
  return RebuildGrid<3>(description, prefix);
}

} // namespace pcms::detail
