#include "description.hpp"
#include "pcms/coupler/coupler.hpp"
#include "pcms/coupler/detail/field_exchange_planner.hpp"
#include "pcms/field/embedding.hpp"
#include "pcms/field/function_spaces.hpp"
#include "pcms/field/gid_permutation.hpp"
#include "pcms/utility/detail/types.hpp"
#include <algorithm>
#include <array>

namespace pcms
{

namespace
{

/// Value type names, indexed by Type.
constexpr std::array<std::string_view, 5> TypeNames{"float64", "int32", "int64",
                                                    "int8", "float32"};

std::string Prefix(const char* group, size_t index)
{
  return std::string(group) + "." + std::to_string(index);
}

/// Sends messages[peer] to each peer rank.
void Send(redev::Channel& channel, redev::BidirectionalComm<GO>& comm,
          const std::vector<std::vector<GO>>& messages)
{
  std::vector<GO> all;
  redev::LOs dest;
  redev::LOs offsets{0};
  for (size_t peer = 0; peer < messages.size(); ++peer) {
    all.insert(all.end(), messages[peer].begin(), messages[peer].end());
    dest.push_back(static_cast<redev::LO>(peer));
    offsets.push_back(static_cast<redev::LO>(all.size()));
  }
  comm.SetOutMessageLayout(dest, offsets);
  channel.SendPhase([&] { comm.Send(all.data(), redev::Mode::Synchronous); });
}

/// @return the message from each peer rank
std::vector<std::vector<GO>> Receive(redev::Channel& channel,
                                     redev::BidirectionalComm<GO>& comm,
                                     MPI_Comm mpi_comm)
{
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(mpi_comm, &rank);
  MPI_Comm_size(mpi_comm, &size);
  const auto all =
    channel.ReceivePhase([&] { return comm.Recv(redev::Mode::Synchronous); });
  const auto& sources = comm.GetInMessageLayout().srcRanks;
  const auto peers = sources.size() / static_cast<size_t>(size);
  std::vector<std::vector<GO>> messages;
  for (size_t peer = 0; peer < peers; ++peer) {
    const auto begin = static_cast<size_t>(sources[peer * size + rank]);
    const auto end = peer + 1 < peers
                       ? static_cast<size_t>(sources[(peer + 1) * size + rank])
                       : all.size();
    messages.emplace_back(all.begin() + begin, all.begin() + end);
  }
  return messages;
}

/// @return for each DOF holder of `layout`, whether this server rank
///         receives it: all of a distributed layout, else those the
///         partition assigns to `rank`
std::vector<bool> Received(const FieldLayout& layout,
                           const redev::Partition& partition, int rank)
{
  std::vector<bool> received(layout.GetNumDofHolder(), layout.IsDistributed());
  if (layout.IsDistributed()) {
    return received;
  }
  const OverlapMask all(received.size());
  const auto plan = detail::GenericFieldExchangePlanner{}.BuildExchangePlan(
    layout, partition, &all);
  for (size_t k = 0; k < plan.dest_ranks.size(); ++k) {
    for (size_t h = 0; h < received.size() && plan.dest_ranks[k] == rank; ++h) {
      received[h] = plan.offsets[k] <= plan.permutation[h] &&
                    plan.permutation[h] < plan.offsets[k + 1];
    }
  }
  return received;
}

/// The plan that sends holder lists[peer][k] to position k of peer's part.
detail::ExchangePlan PlanFromLists(const std::vector<std::vector<LO>>& lists,
                                   LO num_holders)
{
  detail::ExchangePlan plan;
  plan.permutation = field::MakeUnmappedPermutation(num_holders);
  plan.offsets.push_back(0);
  for (size_t peer = 0; peer < lists.size(); ++peer) {
    if (lists[peer].empty()) {
      continue;
    }
    plan.dest_ranks.push_back(static_cast<redev::LO>(peer));
    for (const LO holder : lists[peer]) {
      plan.permutation[holder] = static_cast<LO>(plan.msg_size++);
    }
    plan.offsets.push_back(static_cast<redev::LO>(plan.msg_size));
  }
  return plan;
}

std::vector<std::vector<GO>> DescribeGeometry(
  const Geometry& geometry, const std::string& prefix,
  detail::DescribedSystems& systems, detail::Description& description,
  MPI_Comm comm, [[maybe_unused]] const redev::Partition& partition,
  size_t num_servers)
{
  auto blocks =
    detail::DescribeUniformGrid(geometry, prefix, systems, description);
  if (!blocks) {
    blocks = detail::DescribeEmbedded(geometry, prefix, systems, description,
                                      comm, num_servers);
  }
#ifdef PCMS_ENABLE_OMEGA_H
  if (!blocks) {
    blocks = detail::DescribeOmegaH(geometry, prefix, systems, description,
                                    comm, partition, num_servers);
  }
#endif
  if (!blocks) {
    throw pcms_error("Application::Setup: only uniform grids, embedded "
                     "geometries over them and Omega_h meshes can be "
                     "described");
  }
  blocks->resize(num_servers);
  return *std::move(blocks);
}

} // namespace

void ApplicationComm::Expect(std::vector<std::string> names)
{
  expected_functions_ = std::move(names);
}

void ApplicationComm::Setup()
{
  PCMS_FUNCTION_TIMER;
  if (set_up_ || InSendPhase() || InReceivePhase()) {
    throw pcms_error("Application::Setup: called twice or in a phase");
  }
  if (redev_.GetProcessType() == redev::ProcessType::Client) {
    SetupClient();
  } else {
    SetupServer();
  }
  set_up_ = true;
}

void ApplicationComm::AddDescribedCommunicators(
  const std::vector<const FieldLayout*>& layouts,
  const std::vector<size_t>& function_spaces,
  const std::vector<std::vector<std::vector<LO>>>& lists)
{
  std::vector<detail::FieldLayoutCommunicator*> communicators;
  for (size_t j = 0; j < layouts.size(); ++j) {
    owned_field_layout_communicators_.push_back(
      std::make_unique<detail::FieldLayoutCommunicator>(
        "pcms_described_space_" + std::to_string(j), mpi_comm_, redev_,
        channel_, *layouts[j],
        PlanFromLists(lists[j], layouts[j]->GetNumDofHolder())));
    communicators.push_back(owned_field_layout_communicators_.back().get());
  }
  for (size_t i = 0; i < described_functions_.size(); ++i) {
    const auto& name = described_functions_[i];
    std::visit(
      [&](auto& function) {
        using T =
          typename std::decay_t<decltype(function.GetData())>::value_type;
        if (!field_communicators_
               .emplace(name,
                        std::make_unique<detail::FieldCommunicator<T>>(
                          name, *communicators[function_spaces[i]], function))
               .second) {
          throw pcms_error("Field with this name already exists");
        }
      },
      functions_.at(name));
  }
}

void ApplicationComm::SetupClient()
{
  int rank = 0;
  MPI_Comm_rank(mpi_comm_, &rank);
  const auto& partition = redev_.GetPartition();
  const auto server_ranks =
    std::visit([](const auto& ptn) { return ptn.GetRanks(); }, partition);
  const auto num_servers = static_cast<size_t>(
    *std::max_element(server_ranks.begin(), server_ranks.end()) + 1);

  auto description = detail::CreateDescription();
  detail::DescribedSystems systems;
  std::vector<const Geometry*> geometries;
  std::vector<std::vector<std::vector<GO>>> blocks;
  std::vector<const FunctionSpace*> spaces;
  std::vector<const FieldLayout*> layouts;
  std::vector<GO> counts;
  std::vector<size_t> function_spaces;
  for (size_t i = 0; i < described_functions_.size(); ++i) {
    std::visit(
      [&](const auto& function) {
        using T =
          typename std::decay_t<decltype(function.GetData())>::value_type;
        const auto& space = function.GetSpace();
        const auto* geometry = space.GetGeometry().get();
        const auto g = static_cast<size_t>(
          std::find(geometries.begin(), geometries.end(), geometry) -
          geometries.begin());
        if (g == geometries.size()) {
          geometries.push_back(geometry);
          blocks.push_back(DescribeGeometry(*geometry, Prefix("geometry", g),
                                            systems, description, mpi_comm_,
                                            partition, num_servers));
        }
        const auto j = static_cast<size_t>(
          std::find(spaces.begin(), spaces.end(), &space) - spaces.begin());
        if (j == spaces.size()) {
          const auto& layout = *space.GetLayout();
          if (layout_overlap_masks_.count(layout.GetName()) != 0) {
            throw pcms_error("Application::Setup: described functions take "
                             "no overlap mask");
          }
          spaces.push_back(&space);
          layouts.push_back(&layout);
          const auto prefix = Prefix("space", j);
          description[prefix + ".geometry"] = std::to_string(g);
          description[prefix + ".order"] =
            std::to_string(detail::LagrangeOrder(layout));
          description[prefix + ".components"] =
            std::to_string(layout.GetNumComponents());
          // the first rank sends the values of a replicated layout
          counts.push_back(
            layout.IsDistributed() || rank == 0 ? layout.GetNumDofHolder() : 0);
        }
        function_spaces.push_back(j);
        const auto prefix = Prefix("function", i);
        detail::DescribeMetadata(function.GetMetadata(), prefix, description);
        description[prefix + ".space"] = std::to_string(j);
        description[prefix + ".type"] =
          TypeNames[static_cast<size_t>(detail::TypeEnumFromType<T>())];
        detail::DescribeValueBasis(function.GetData().GetValueBasis(),
                                   prefix + ".basis", systems, description);
      },
      functions_.at(described_functions_[i]));
  }
  description["geometries"] = std::to_string(geometries.size());
  description["spaces"] = std::to_string(spaces.size());
  description["functions"] = std::to_string(described_functions_.size());

  // to each server rank: [description length, description (from rank 0),
  // holders this rank sends per space, per geometry: [size, data]]
  const auto text =
    rank == 0 ? detail::SerializeDescription(description) : std::string{};
  std::vector<std::vector<GO>> messages(num_servers);
  for (size_t s = 0; s < num_servers; ++s) {
    auto& message = messages[s];
    message.push_back(static_cast<GO>(text.size()));
    message.insert(message.end(), text.begin(), text.end());
    message.insert(message.end(), counts.begin(), counts.end());
    for (const auto& geometry_blocks : blocks) {
      message.push_back(static_cast<GO>(geometry_blocks[s].size()));
      message.insert(message.end(), geometry_blocks[s].begin(),
                     geometry_blocks[s].end());
    }
  }
  auto comm = channel_.CreateComm<GO>("pcms_setup", mpi_comm_);
  Send(channel_, comm, messages);

  // from each server rank, per space: [n, the owned indices of the n holders
  // it receives from this rank, in message order]
  const auto replies = Receive(channel_, comm, mpi_comm_);
  std::vector<std::vector<std::vector<LO>>> lists(
    spaces.size(), std::vector<std::vector<LO>>(replies.size()));
  for (size_t s = 0; s < replies.size(); ++s) {
    auto it = replies[s].begin();
    for (auto& list : lists) {
      const auto n = *it++;
      list[s].assign(it, it + n);
      it += n;
    }
  }
  AddDescribedCommunicators(layouts, function_spaces, lists);
}

void ApplicationComm::SetupServer()
{
  int rank = 0;
  MPI_Comm_rank(mpi_comm_, &rank);
  auto comm = channel_.CreateComm<GO>("pcms_setup", mpi_comm_);
  const auto messages = Receive(channel_, comm, mpi_comm_);
  const auto num_apps = messages.size();
  const auto& first = messages.front();
  const auto description = detail::ParseDescription(
    std::string(first.begin() + 1, first.begin() + 1 + first[0]));
  const auto integer = [&](const std::string& key) {
    return static_cast<size_t>(
      detail::ParseInteger(detail::Require(description, key)));
  };
  const auto systems = detail::SystemsFromDescription(description);
  const auto num_geometries = integer("geometries");
  const auto num_spaces = integer("spaces");
  const auto num_functions = integer("functions");
  std::vector<std::string> names;
  for (size_t i = 0; i < num_functions; ++i) {
    names.push_back(
      detail::Require(description, Prefix("function", i) + ".name"));
  }
  for (const auto& expected : expected_functions_) {
    if (std::find(names.begin(), names.end(), expected) == names.end()) {
      throw pcms_error("Application::Setup: the application does not "
                       "describe the expected function '" +
                       expected + "'");
    }
  }

  // per space, the first server id of each application rank's holders; per
  // geometry, each application rank's data
  std::vector<std::vector<GO>> offsets(num_spaces, std::vector<GO>{0});
  std::vector<std::vector<std::vector<GO>>> blocks(
    num_geometries, std::vector<std::vector<GO>>(num_apps));
  for (size_t r = 0; r < num_apps; ++r) {
    auto it = messages[r].begin() + 1 + messages[r][0];
    for (auto& space_offsets : offsets) {
      space_offsets.push_back(space_offsets.back() + *it++);
    }
    for (auto& geometry_blocks : blocks) {
      const auto n = *it++;
      geometry_blocks[r].assign(it, it + n);
      it += n;
    }
  }

  std::vector<std::shared_ptr<const Geometry>> geometries;
  for (size_t g = 0; g < num_geometries; ++g) {
    const auto prefix = Prefix("geometry", g);
    const auto& kind = detail::Require(description, prefix + ".kind");
    std::shared_ptr<const Geometry> geometry;
    if (kind == "uniform_grid") {
      geometry = detail::RebuildUniformGrid(description, prefix, systems);
    }
    if (kind == "embedded") {
      geometry =
        detail::RebuildEmbedded(description, prefix, systems, blocks[g]);
    }
#ifdef PCMS_ENABLE_OMEGA_H
    if (kind == "omega_h") {
      geometry = detail::RebuildOmegaH(description, prefix, systems, blocks[g],
                                       mpi_comm_);
    }
#endif
    if (geometry == nullptr) {
      throw pcms_error("Application::Setup: unknown geometry kind '" + kind +
                       "'");
    }
    geometries.push_back(std::move(geometry));
  }

  // the holders this rank receives from each application rank, and their
  // owned indices there
  std::vector<std::shared_ptr<const FunctionSpace>> spaces;
  std::vector<const FieldLayout*> layouts;
  std::vector<std::vector<std::vector<LO>>> holders(
    num_spaces, std::vector<std::vector<LO>>(num_apps));
  std::vector<std::vector<GO>> replies(num_apps);
  for (size_t j = 0; j < num_spaces; ++j) {
    const auto prefix = Prefix("space", j);
    const auto& geometry = geometries.at(integer(prefix + ".geometry"));
    const auto embedded =
      std::dynamic_pointer_cast<const EmbeddedGeometry>(geometry);
    auto space = CreateLagrangeSpace(
      embedded ? embedded->GetParameterGeometry() : geometry,
      static_cast<int>(integer(prefix + ".order")),
      static_cast<int>(integer(prefix + ".components")));
    spaces.push_back(embedded ? CreateEmbeddedSpace(embedded, std::move(space))
                              : std::move(space));
    const auto& layout = *spaces.back()->GetLayout();
    layouts.push_back(&layout);
    const auto gids = layout.GetGidsHost();
    const auto received = Received(layout, redev_.GetPartition(), rank);
    std::vector<std::vector<GO>> app_indices(num_apps);
    for (LO h = 0; h < layout.GetNumDofHolder(); ++h) {
      const auto app = static_cast<size_t>(
        std::upper_bound(offsets[j].begin(), offsets[j].end(), gids[h]) -
        offsets[j].begin() - 1);
      if (received[h]) {
        holders[j][app].push_back(h);
        app_indices[app].push_back(gids[h] - offsets[j][app]);
      }
    }
    for (size_t r = 0; r < num_apps; ++r) {
      replies[r].push_back(static_cast<GO>(app_indices[r].size()));
      replies[r].insert(replies[r].end(), app_indices[r].begin(),
                        app_indices[r].end());
    }
  }

  std::vector<size_t> function_spaces;
  for (size_t i = 0; i < num_functions; ++i) {
    const auto prefix = Prefix("function", i);
    const auto basis = detail::ValueBasisFromDescription(
      description, prefix + ".basis", systems);
    if (basis.parity != Parity::Even) {
      throw pcms_error("Application::Setup: functions of odd parity cannot "
                       "be rebuilt yet");
    }
    function_spaces.push_back(integer(prefix + ".space"));
    const auto type =
      std::find(TypeNames.begin(), TypeNames.end(),
                detail::Require(description, prefix + ".type")) -
      TypeNames.begin();
    detail::apply_to_type(static_cast<Type>(type), [&](auto tag) {
      using T = typename decltype(tag)::type;
      functions_.emplace(
        names[i],
        spaces.at(function_spaces.back())
          ->CreateFunction<T>(
            detail::MetadataFromDescription(description, prefix), basis));
    });
    described_functions_.push_back(names[i]);
  }
  Send(channel_, comm, replies);
  AddDescribedCommunicators(layouts, function_spaces, holders);
}

} // namespace pcms
