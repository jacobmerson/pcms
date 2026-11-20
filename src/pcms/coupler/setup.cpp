// ApplicationComm::Setup: clients send the descriptions of the functions they
// described and the data of their geometries; the server rebuilds them and
// tells each application rank which of its DOF holders' values it receives,
// in which order. The exchange plans of both sides follow from that reply.
//
// Messages (each on its own redev communicator, used once):
//   client -> server  pcms_setup_description  the description, from rank 0
//                     pcms_setup_ints/_reals  per server rank, per geometry:
//                                             [#ints, #reals, ints...], reals
//   server -> client  pcms_setup_claims       per application rank, per space:
//                                             [n, owned indices of the n
//                                             holders it sends this server
//                                             rank, in send order]

#include "description.hpp"
#include "pcms/coupler/coupler.hpp"
#include "pcms/field/gid_permutation.hpp"
#include "pcms/utility/detail/types.hpp"
#include <algorithm>

namespace pcms
{

namespace
{

template <typename F>
struct ValueTypeOf;

template <typename T>
struct ValueTypeOf<Function<T>>
{
  using type = T;
};

/// Each sender's [begin, end) in the data this rank received.
std::vector<std::pair<size_t, size_t>> SenderSegments(
  const redev::InMessageLayout& in, int rank, int nproc, size_t total)
{
  const size_t senders = in.srcRanks.size() / static_cast<size_t>(nproc);
  std::vector<std::pair<size_t, size_t>> segments(senders);
  for (size_t i = 0; i < senders; ++i) {
    const auto begin = static_cast<size_t>(in.srcRanks[i * nproc + rank]);
    const size_t end =
      i + 1 < senders ? static_cast<size_t>(in.srcRanks[(i + 1) * nproc + rank])
                      : total;
    segments[i] = {begin, end};
  }
  return segments;
}

int NumServerRanks(const redev::Partition& partition)
{
  const auto ranks = std::visit(
    [](const auto& ptn) -> std::vector<redev::LO> { return ptn.GetRanks(); },
    partition);
  return ranks.empty() ? 0 : *std::max_element(ranks.begin(), ranks.end()) + 1;
}

/// The plan of a layout whose holders are sent, per peer rank, in the order
/// of lists[peer]: holder lists[peer][k] goes to position k of that peer's
/// part of the message.
detail::ExchangePlan PlanFromLists(const std::vector<std::vector<LO>>& lists,
                                   LO num_holders)
{
  detail::ExchangePlan plan;
  plan.permutation = field::MakeUnmappedPermutation(num_holders);
  plan.offsets.push_back(0);
  LO position = 0;
  for (size_t peer = 0; peer < lists.size(); ++peer) {
    if (lists[peer].empty()) {
      continue;
    }
    plan.dest_ranks.push_back(static_cast<redev::LO>(peer));
    for (const LO holder : lists[peer]) {
      if (holder < 0 || holder >= num_holders ||
          plan.permutation[holder] >= 0) {
        throw pcms_error("Setup: a DOF holder is sent twice or does not exist");
      }
      plan.permutation[holder] = position++;
    }
    plan.offsets.push_back(position);
  }
  plan.msg_size = static_cast<size_t>(position);
  return plan;
}

std::string TypeName(Type type)
{
  switch (type) {
    case Type::Real: return "float64";
    case Type::Float: return "float32";
    case Type::LO: return "int32";
    case Type::GO: return "int64";
    case Type::Int8: return "int8";
  }
  throw pcms_error("Setup: unknown value type");
}

Type TypeFromName(const std::string& name)
{
  for (const Type type :
       {Type::Real, Type::Float, Type::LO, Type::GO, Type::Int8}) {
    if (TypeName(type) == name) {
      return type;
    }
  }
  throw pcms_error("Setup: unknown value type '" + name + "'");
}

std::string Key(std::string_view group, size_t index, std::string_view field)
{
  return std::string(group) + "." + std::to_string(index) + "." +
         std::string(field);
}

} // namespace

void ApplicationComm::Expect(std::vector<std::string> names)
{
  expected_functions_.insert(expected_functions_.end(),
                             std::make_move_iterator(names.begin()),
                             std::make_move_iterator(names.end()));
}

void ApplicationComm::Setup()
{
  PCMS_FUNCTION_TIMER;
  if (set_up_) {
    throw pcms_error("Application::Setup: already set up");
  }
  if (InSendPhase() || InReceivePhase()) {
    throw pcms_error("Application::Setup: called in a send or receive phase");
  }
  if (redev_.GetProcessType() == redev::ProcessType::Client) {
    SetupClient();
  } else {
    SetupServer();
  }
  set_up_ = true;
}

void ApplicationComm::AddDescribedCommunicators(
  const std::vector<std::shared_ptr<const FieldLayout>>& layouts,
  const std::vector<std::size_t>& function_spaces,
  std::vector<detail::ExchangePlan> plans)
{
  std::vector<detail::FieldLayoutCommunicator*> layout_communicators;
  for (size_t j = 0; j < layouts.size(); ++j) {
    owned_field_layout_communicators_.push_back(
      std::make_unique<detail::FieldLayoutCommunicator>(
        "pcms_described_space_" + std::to_string(j), mpi_comm_, redev_,
        channel_, *layouts[j], std::move(plans[j])));
    layout_communicators.push_back(
      owned_field_layout_communicators_.back().get());
  }
  for (size_t i = 0; i < described_functions_.size(); ++i) {
    const auto& name = described_functions_[i];
    auto& layout_communicator = *layout_communicators[function_spaces[i]];
    std::visit(
      [&](auto& function) {
        using T = typename ValueTypeOf<std::decay_t<decltype(function)>>::type;
        detail::FieldCommunicator2Ptr communicator =
          std::make_unique<detail::FieldCommunicator<T>>(
            name, layout_communicator, function);
        if (!field_communicators_.emplace(name, std::move(communicator))
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
  int size = 1;
  MPI_Comm_rank(mpi_comm_, &rank);
  MPI_Comm_size(mpi_comm_, &size);
  const auto& partition = redev_.GetPartition();
  const detail::SetupContext context{mpi_comm_, rank, size,
                                     NumServerRanks(partition), partition};
  const auto num_servers = static_cast<size_t>(context.peer_size);

  // describe the geometries, spaces and functions, each geometry and space
  // once
  detail::Description description;
  description["pcms_schema"] = std::string(detail::DescriptionSchema);
  std::vector<const Geometry*> geometries;
  std::vector<std::vector<detail::SetupBlock>> geometry_blocks;
  std::vector<const FunctionSpace*> spaces;
  std::vector<std::shared_ptr<const FieldLayout>> layouts;
  std::vector<std::size_t> function_spaces;
  for (size_t i = 0; i < described_functions_.size(); ++i) {
    const auto& name = described_functions_[i];
    std::visit(
      [&](const auto& function) {
        using T = typename ValueTypeOf<std::decay_t<decltype(function)>>::type;
        const FunctionSpace& space = function.GetSpace();
        const auto& kind = detail::FindSpaceKind(space);
        if (layout_overlap_masks_.count(space.GetLayout()->GetName()) != 0) {
          throw pcms_error("Application::Setup: the overlap mask of layout '" +
                           space.GetLayout()->GetName() +
                           "' cannot be applied to a described function");
        }
        const Geometry* geometry = space.GetGeometry().get();
        auto g = static_cast<size_t>(
          std::find(geometries.begin(), geometries.end(), geometry) -
          geometries.begin());
        if (g == geometries.size()) {
          geometries.push_back(geometry);
          description[Key("geometry", g, "kind")] = std::string(kind.Name());
          geometry_blocks.push_back(kind.DescribeGeometry(
            *geometry, "geometry." + std::to_string(g), description, context));
        }
        auto j = static_cast<size_t>(
          std::find(spaces.begin(), spaces.end(), &space) - spaces.begin());
        if (j == spaces.size()) {
          spaces.push_back(&space);
          layouts.push_back(space.GetLayout());
          description[Key("space", j, "geometry")] = std::to_string(g);
          kind.DescribeSpace(space, "space." + std::to_string(j), description);
        }
        function_spaces.push_back(j);
        description[Key("function", i, "name")] = name;
        description[Key("function", i, "space")] = std::to_string(j);
        description[Key("function", i, "type")] =
          TypeName(detail::TypeEnumFromType<T>());
        detail::DescribeValueBasis(function.GetData().GetValueBasis(),
                                   Key("function", i, "basis"), description);
      },
      functions_.at(name));
  }
  description["geometries"] = std::to_string(geometries.size());
  description["spaces"] = std::to_string(spaces.size());
  description["functions"] = std::to_string(described_functions_.size());

  // the description from rank 0 to every server rank; the geometry blocks
  const auto text = detail::SerializeDescription(description);
  std::vector<int8_t> text_message;
  redev::LOs text_dest;
  redev::LOs text_offsets{0};
  if (rank == 0) {
    for (size_t s = 0; s < num_servers; ++s) {
      text_message.insert(text_message.end(), text.begin(), text.end());
      text_dest.push_back(static_cast<redev::LO>(s));
      text_offsets.push_back(static_cast<redev::LO>(text_message.size()));
    }
  }
  std::vector<GO> ints;
  std::vector<Real> reals;
  redev::LOs dest;
  redev::LOs int_offsets{0};
  redev::LOs real_offsets{0};
  for (size_t s = 0; s < num_servers; ++s) {
    for (const auto& blocks : geometry_blocks) {
      ints.push_back(static_cast<GO>(blocks[s].ints.size()));
      ints.push_back(static_cast<GO>(blocks[s].reals.size()));
      ints.insert(ints.end(), blocks[s].ints.begin(), blocks[s].ints.end());
      reals.insert(reals.end(), blocks[s].reals.begin(), blocks[s].reals.end());
    }
    dest.push_back(static_cast<redev::LO>(s));
    int_offsets.push_back(static_cast<redev::LO>(ints.size()));
    real_offsets.push_back(static_cast<redev::LO>(reals.size()));
  }
  auto text_comm =
    channel_.CreateComm<int8_t>("pcms_setup_description", mpi_comm_);
  auto int_comm = channel_.CreateComm<GO>("pcms_setup_ints", mpi_comm_);
  auto real_comm = channel_.CreateComm<Real>("pcms_setup_reals", mpi_comm_);
  auto claim_comm = channel_.CreateComm<GO>("pcms_setup_claims", mpi_comm_);
  text_comm.SetOutMessageLayout(text_dest, text_offsets);
  int_comm.SetOutMessageLayout(dest, int_offsets);
  real_comm.SetOutMessageLayout(dest, real_offsets);
  channel_.BeginSendCommunicationPhase();
  text_comm.Send(text_message.data(), redev::Mode::Synchronous);
  int_comm.Send(ints.data(), redev::Mode::Synchronous);
  real_comm.Send(reals.data(), redev::Mode::Synchronous);
  channel_.EndSendCommunicationPhase();

  // the holders each server rank receives, per space
  channel_.BeginReceiveCommunicationPhase();
  const auto claims = claim_comm.Recv(redev::Mode::Synchronous);
  channel_.EndReceiveCommunicationPhase();
  const auto segments =
    SenderSegments(claim_comm.GetInMessageLayout(), rank, size, claims.size());
  std::vector<std::vector<std::vector<LO>>> lists(
    spaces.size(), std::vector<std::vector<LO>>(segments.size()));
  for (size_t s = 0; s < segments.size(); ++s) {
    size_t position = segments[s].first;
    for (size_t j = 0; j < spaces.size(); ++j) {
      PCMS_ALWAYS_ASSERT(position < segments[s].second);
      const auto n = static_cast<size_t>(claims[position++]);
      PCMS_ALWAYS_ASSERT(position + n <= segments[s].second);
      for (size_t k = 0; k < n; ++k) {
        lists[j][s].push_back(static_cast<LO>(claims[position++]));
      }
    }
  }
  std::vector<detail::ExchangePlan> plans;
  for (size_t j = 0; j < spaces.size(); ++j) {
    plans.push_back(PlanFromLists(lists[j], layouts[j]->GetNumDofHolder()));
  }
  AddDescribedCommunicators(layouts, function_spaces, std::move(plans));
}

void ApplicationComm::SetupServer()
{
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(mpi_comm_, &rank);
  MPI_Comm_size(mpi_comm_, &size);
  auto text_comm =
    channel_.CreateComm<int8_t>("pcms_setup_description", mpi_comm_);
  auto int_comm = channel_.CreateComm<GO>("pcms_setup_ints", mpi_comm_);
  auto real_comm = channel_.CreateComm<Real>("pcms_setup_reals", mpi_comm_);
  auto claim_comm = channel_.CreateComm<GO>("pcms_setup_claims", mpi_comm_);
  channel_.BeginReceiveCommunicationPhase();
  const auto text_message = text_comm.Recv(redev::Mode::Synchronous);
  const auto ints = int_comm.Recv(redev::Mode::Synchronous);
  const auto reals = real_comm.Recv(redev::Mode::Synchronous);
  channel_.EndReceiveCommunicationPhase();

  const auto int_segments =
    SenderSegments(int_comm.GetInMessageLayout(), rank, size, ints.size());
  const auto real_segments =
    SenderSegments(real_comm.GetInMessageLayout(), rank, size, reals.size());
  const auto text_segments = SenderSegments(text_comm.GetInMessageLayout(),
                                            rank, size, text_message.size());
  const auto num_apps = int_segments.size();
  PCMS_ALWAYS_ASSERT(num_apps > 0 && real_segments.size() == num_apps);
  const auto description = detail::ParseDescription(std::string_view(
    reinterpret_cast<const char*>(text_message.data()) +
      text_segments.front().first,
    text_segments.front().second - text_segments.front().first));
  const auto num_geometries = static_cast<size_t>(
    detail::ParseInteger(detail::Require(description, "geometries")));
  const auto num_spaces = static_cast<size_t>(
    detail::ParseInteger(detail::Require(description, "spaces")));
  const auto num_functions = static_cast<size_t>(
    detail::ParseInteger(detail::Require(description, "functions")));

  std::vector<std::string> names;
  for (size_t i = 0; i < num_functions; ++i) {
    names.push_back(detail::Require(description, Key("function", i, "name")));
  }
  for (const auto& expected : expected_functions_) {
    if (std::find(names.begin(), names.end(), expected) == names.end()) {
      throw pcms_error("Application::Setup: the application does not "
                       "describe the expected function '" +
                       expected + "'");
    }
  }

  // split the blocks per geometry and application rank
  std::vector<std::vector<detail::ReceivedBlock>> blocks(
    num_geometries, std::vector<detail::ReceivedBlock>(num_apps));
  for (size_t r = 0; r < num_apps; ++r) {
    size_t int_position = int_segments[r].first;
    size_t real_position = real_segments[r].first;
    for (size_t g = 0; g < num_geometries; ++g) {
      PCMS_ALWAYS_ASSERT(int_position + 2 <= int_segments[r].second);
      const auto num_ints = static_cast<size_t>(ints[int_position]);
      const auto num_reals = static_cast<size_t>(ints[int_position + 1]);
      int_position += 2;
      PCMS_ALWAYS_ASSERT(int_position + num_ints <= int_segments[r].second &&
                         real_position + num_reals <= real_segments[r].second);
      blocks[g][r] = {
        std::span<const GO>(ints.data() + int_position, num_ints),
        std::span<const Real>(reals.data() + real_position, num_reals)};
      int_position += num_ints;
      real_position += num_reals;
    }
  }

  const auto& partition = redev_.GetPartition();
  const detail::SetupContext context{mpi_comm_, rank, size,
                                     static_cast<int>(num_apps), partition};
  std::vector<detail::RebuiltGeometry> geometries;
  for (size_t g = 0; g < num_geometries; ++g) {
    const auto& kind = detail::FindSpaceKind(
      detail::Require(description, Key("geometry", g, "kind")));
    geometries.push_back(kind.RebuildGeometry(
      description, "geometry." + std::to_string(g), blocks[g], context));
  }
  std::vector<std::shared_ptr<const FunctionSpace>> spaces;
  std::vector<std::shared_ptr<const FieldLayout>> layouts;
  std::vector<std::vector<detail::Claim>> claims;
  for (size_t j = 0; j < num_spaces; ++j) {
    const auto g = static_cast<size_t>(detail::ParseInteger(
      detail::Require(description, Key("space", j, "geometry"))));
    const auto& kind = detail::FindSpaceKind(
      detail::Require(description, Key("geometry", g, "kind")));
    spaces.push_back(kind.RebuildSpace(geometries.at(g), description,
                                       "space." + std::to_string(j)));
    layouts.push_back(spaces.back()->GetLayout());
    claims.push_back(geometries[g].claims(*layouts.back()));
  }

  // the server's functions, owned by this application
  std::vector<std::size_t> function_spaces;
  for (size_t i = 0; i < num_functions; ++i) {
    const auto j = static_cast<size_t>(detail::ParseInteger(
      detail::Require(description, Key("function", i, "space"))));
    const auto basis = detail::ValueBasisFromDescription(
      description, Key("function", i, "basis"));
    if (basis.parity != Parity::Even) {
      throw pcms_error("Application::Setup: functions of odd parity cannot "
                       "be rebuilt yet");
    }
    const auto& space = *spaces.at(j);
    const auto type =
      TypeFromName(detail::Require(description, Key("function", i, "type")));
    detail::apply_to_type(type, [&](auto tag) {
      using T = typename decltype(tag)::type;
      auto function = basis.Rank() == 0
                        ? space.CreateFunction<T>(names[i])
                        : space.CreateFunction<T>(names[i], basis.variance,
                                                  basis.system, basis.scaling);
      if (fields_.count(names[i]) != 0 ||
          !functions_.emplace(names[i], std::move(function)).second) {
        throw pcms_error("Field with this name already exists");
      }
    });
    described_functions_.push_back(names[i]);
    function_spaces.push_back(j);
  }

  // tell each application rank the holders it sends this server rank
  std::vector<std::vector<std::vector<LO>>> by_app(
    num_spaces, std::vector<std::vector<LO>>(num_apps));
  std::vector<std::vector<std::vector<LO>>> app_indices(
    num_spaces, std::vector<std::vector<LO>>(num_apps));
  for (size_t j = 0; j < num_spaces; ++j) {
    for (const auto& claim : claims[j]) {
      by_app[j].at(static_cast<size_t>(claim.app_rank)).push_back(claim.holder);
      app_indices[j][static_cast<size_t>(claim.app_rank)].push_back(
        claim.app_index);
    }
  }
  std::vector<GO> reply;
  redev::LOs reply_dest;
  redev::LOs reply_offsets{0};
  for (size_t r = 0; r < num_apps; ++r) {
    for (size_t j = 0; j < num_spaces; ++j) {
      reply.push_back(static_cast<GO>(app_indices[j][r].size()));
      reply.insert(reply.end(), app_indices[j][r].begin(),
                   app_indices[j][r].end());
    }
    reply_dest.push_back(static_cast<redev::LO>(r));
    reply_offsets.push_back(static_cast<redev::LO>(reply.size()));
  }
  claim_comm.SetOutMessageLayout(reply_dest, reply_offsets);
  channel_.BeginSendCommunicationPhase();
  claim_comm.Send(reply.data(), redev::Mode::Synchronous);
  channel_.EndSendCommunicationPhase();

  std::vector<detail::ExchangePlan> plans;
  for (size_t j = 0; j < num_spaces; ++j) {
    plans.push_back(PlanFromLists(by_app[j], layouts[j]->GetNumDofHolder()));
  }
  AddDescribedCommunicators(layouts, function_spaces, std::move(plans));
}

} // namespace pcms
