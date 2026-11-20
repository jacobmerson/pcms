// The uniform-grid Lagrange spaces (order 0 and 1) as a described kind: the
// description holds the grid, and each application rank sends only the range
// of grid cells whose values it holds.

#include "description.hpp"
#include "pcms/backends/uniform_grid/geometry.hpp"
#include "pcms/backends/uniform_grid/layout.hpp"
#include "pcms/backends/uniform_grid/uniform_grid_geometry.hpp"
#include "pcms/field/function_spaces.hpp"
#include "pcms/utility/assert.hpp"
#include <array>
#include <sstream>

namespace pcms::detail
{

namespace
{

using uniform_grid::UniformGridFieldLayout;
using uniform_grid::UniformGridGeometry;

template <unsigned Dim>
std::vector<SetupBlock> DescribeGrid(const UniformGrid<Dim>& grid,
                                     const std::string& prefix,
                                     Description& description,
                                     const SetupContext& context)
{
  std::string bot_left;
  std::string edge_length;
  std::string divisions;
  for (unsigned d = 0; d < Dim; ++d) {
    const char* separator = d == 0 ? "" : " ";
    bot_left += separator + FormatReal(grid.bot_left[d]);
    edge_length += separator + FormatReal(grid.edge_length[d]);
    divisions += separator + std::to_string(grid.divisions[d]);
  }
  description[prefix + ".dimension"] = std::to_string(Dim);
  description[prefix + ".bot_left"] = bot_left;
  description[prefix + ".edge_length"] = edge_length;
  description[prefix + ".divisions"] = divisions;
  // A uniform-grid layout is replicated on every rank: the first rank sends
  // all the values, the others none. Its range is all the cells.
  std::vector<GO> range(size_t{2} * Dim, 0);
  if (context.rank == 0) {
    for (unsigned d = 0; d < Dim; ++d) {
      range[size_t{2} * d + 1] = grid.divisions[d];
    }
  }
  return std::vector<SetupBlock>(static_cast<size_t>(context.peer_size),
                                 SetupBlock{range, {}});
}

std::vector<std::string> Words(const std::string& text)
{
  std::vector<std::string> words;
  std::istringstream stream(text);
  for (std::string word; stream >> word;) {
    words.push_back(word);
  }
  return words;
}

template <unsigned Dim>
UniformGrid<Dim> GridFromDescription(const Description& description,
                                     const std::string& prefix)
{
  const auto bot_left = Words(Require(description, prefix + ".bot_left"));
  const auto edge_length = Words(Require(description, prefix + ".edge_length"));
  const auto divisions = Words(Require(description, prefix + ".divisions"));
  if (bot_left.size() != Dim || edge_length.size() != Dim ||
      divisions.size() != Dim) {
    throw pcms_error("Setup: the grid '" + prefix + "' is not " +
                     std::to_string(Dim) + "D");
  }
  UniformGrid<Dim> grid{};
  for (unsigned d = 0; d < Dim; ++d) {
    grid.bot_left[d] = ParseReal(bot_left[d]);
    grid.edge_length[d] = ParseReal(edge_length[d]);
    grid.divisions[d] = static_cast<LO>(ParseInteger(divisions[d]));
  }
  return grid;
}

/// The cells [begin[d], end[d]) per axis whose values an application rank
/// holds, and the DOF holders of a layout of `order` on them: the cells, or
/// the vertices [begin[d], end[d]] of the cells.
template <unsigned Dim>
struct CellRange
{
  std::array<LO, Dim> begin{};
  std::array<LO, Dim> end{};

  [[nodiscard]] bool Empty() const
  {
    for (unsigned d = 0; d < Dim; ++d) {
      if (end[d] <= begin[d]) {
        return true;
      }
    }
    return false;
  }

  /// @return the holder's index among the range's holders, -1 if the range
  ///         does not hold it
  [[nodiscard]] LO IndexOf(const std::array<LO, Dim>& holder, int order) const
  {
    if (Empty()) {
      return -1;
    }
    LO index = 0;
    LO stride = 1;
    for (unsigned d = 0; d < Dim; ++d) {
      const LO extent = end[d] - begin[d] + order;
      const LO offset = holder[d] - begin[d];
      if (offset < 0 || offset >= extent) {
        return -1;
      }
      index += offset * stride;
      stride *= extent;
    }
    return index;
  }
};

template <unsigned Dim>
std::vector<Claim> GridClaims(const UniformGrid<Dim>& grid,
                              const std::vector<CellRange<Dim>>& ranges,
                              const redev::Partition& partition, int rank,
                              const FieldLayout& layout)
{
  const auto* grid_layout =
    dynamic_cast<const UniformGridFieldLayout<Dim>*>(&layout);
  PCMS_ALWAYS_ASSERT(grid_layout != nullptr);
  const int order = grid_layout->GetOrder();
  const auto coords = HostDOFHolderCoordinates(layout);
  const auto class_dims = layout.GetDOFHolderClassificationDimensionsHost();
  const auto class_ids = layout.GetDOFHolderClassificationIdsHost();
  std::vector<Claim> claims;
  for (LO holder = 0; holder < layout.GetNumDofHolder(); ++holder) {
    if (RankOf(partition, class_dims[holder], class_ids[holder],
               &coords(holder, 0), static_cast<int>(Dim)) != rank) {
      continue;
    }
    // the holder's index per axis; x varies fastest
    std::array<LO, Dim> index{};
    LO rest = holder;
    for (unsigned d = 0; d < Dim; ++d) {
      const LO extent = grid.divisions[d] + order;
      index[d] = rest % extent;
      rest /= extent;
    }
    // the lowest application rank that holds the values sends them
    for (size_t app_rank = 0; app_rank < ranges.size(); ++app_rank) {
      const LO app_index = ranges[app_rank].IndexOf(index, order);
      if (app_index >= 0) {
        claims.push_back({holder, static_cast<LO>(app_rank), app_index});
        break;
      }
    }
  }
  return claims;
}

template <unsigned Dim>
RebuiltGeometry RebuildGrid(const Description& description,
                            const std::string& prefix,
                            std::span<const ReceivedBlock> blocks,
                            const SetupContext& context)
{
  const auto grid = GridFromDescription<Dim>(description, prefix);
  auto geometry = uniform_grid::CreateGeometry(
    grid, ChartFromDescription(description, prefix + ".chart"));
  std::vector<CellRange<Dim>> ranges(blocks.size());
  for (size_t app_rank = 0; app_rank < blocks.size(); ++app_rank) {
    const auto& ints = blocks[app_rank].ints;
    if (ints.size() != size_t{2} * Dim) {
      throw pcms_error("Setup: malformed grid block");
    }
    for (unsigned d = 0; d < Dim; ++d) {
      ranges[app_rank].begin[d] = static_cast<LO>(ints[size_t{2} * d]);
      ranges[app_rank].end[d] = static_cast<LO>(ints[size_t{2} * d + 1]);
    }
  }
  auto claims = [grid, ranges = std::move(ranges),
                 partition = context.partition,
                 rank = context.rank](const FieldLayout& layout) {
    return GridClaims<Dim>(grid, ranges, partition, rank, layout);
  };
  return {std::move(geometry), std::move(claims)};
}

class UniformGridKind final : public SpaceKind
{
public:
  [[nodiscard]] std::string_view Name() const noexcept override
  {
    return "uniform_grid";
  }

  [[nodiscard]] bool Describes(const FunctionSpace& space) const override
  {
    const auto* layout = space.GetLayout().get();
    return dynamic_cast<const UniformGridFieldLayout<2>*>(layout) != nullptr ||
           dynamic_cast<const UniformGridFieldLayout<3>*>(layout) != nullptr;
  }

  [[nodiscard]] std::vector<SetupBlock> DescribeGeometry(
    const Geometry& geometry, const std::string& prefix,
    Description& description, const SetupContext& context) const override
  {
    DescribeChart(*geometry.GetCoordinateSystem(), prefix + ".chart",
                  description);
    if (const auto* grid =
          dynamic_cast<const UniformGridGeometry<2>*>(&geometry)) {
      return DescribeGrid(grid->GetGrid(), prefix, description, context);
    }
    return DescribeGrid(
      dynamic_cast<const UniformGridGeometry<3>&>(geometry).GetGrid(), prefix,
      description, context);
  }

  void DescribeSpace(const FunctionSpace& space, const std::string& prefix,
                     Description& description) const override
  {
    const auto& layout = *space.GetLayout();
    const auto* layout2 =
      dynamic_cast<const UniformGridFieldLayout<2>*>(&layout);
    const int order =
      layout2 != nullptr
        ? layout2->GetOrder()
        : dynamic_cast<const UniformGridFieldLayout<3>&>(layout).GetOrder();
    description[prefix + ".order"] = std::to_string(order);
    description[prefix + ".components"] =
      std::to_string(layout.GetNumComponents());
  }

  [[nodiscard]] RebuiltGeometry RebuildGeometry(
    const Description& description, const std::string& prefix,
    std::span<const ReceivedBlock> blocks,
    const SetupContext& context) const override
  {
    const auto dim = ParseInteger(Require(description, prefix + ".dimension"));
    if (dim == 2) {
      return RebuildGrid<2>(description, prefix, blocks, context);
    }
    if (dim == 3) {
      return RebuildGrid<3>(description, prefix, blocks, context);
    }
    throw pcms_error("Setup: a uniform grid is 2D or 3D");
  }

  [[nodiscard]] std::shared_ptr<const FunctionSpace> RebuildSpace(
    const RebuiltGeometry& geometry, const Description& description,
    const std::string& prefix) const override
  {
    return CreateLagrangeSpace(
      geometry.geometry,
      static_cast<int>(ParseInteger(Require(description, prefix + ".order"))),
      static_cast<int>(
        ParseInteger(Require(description, prefix + ".components"))));
  }
};

} // namespace

const SpaceKind& UniformGridSpaceKind()
{
  static const UniformGridKind kind;
  return kind;
}

} // namespace pcms::detail
