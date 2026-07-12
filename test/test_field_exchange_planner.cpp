#include <catch2/catch_test_macros.hpp>

#include <pcms/coupler/field_exchange_planner.h>
#include <pcms/field/layout/point_cloud.h>

TEST_CASE("Field exchange plans exclude GID message headers",
          "[field_exchange]")
{
  Kokkos::View<pcms::Real**> coords("coords", 4, 2);
  pcms::PointCloudLayout layout(2, coords, pcms::CoordinateSystem::Cartesian);

  // Two incoming messages. Each contains a five-entry entity-offset header
  // followed by two vertex GIDs.
  Kokkos::View<pcms::GO*, pcms::HostMemorySpace> received(
    "received_gids", 2 * (pcms::ent_offsets_len + 2));
  const pcms::GO message[] = {
    0, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 3, 1,
  };
  for (size_t i = 0; i < received.size(); ++i)
    received(i) = message[i];

  redev::InMessageLayout in;
  in.srcRanks = {0, 7};
  in.offset = {0, 14};

  pcms::GenericFieldExchangePlanner planner;
  const auto plan = planner.BuildReceivePlan(
    layout,
    pcms::GlobalIDView<pcms::HostMemorySpace>(received.data(), received.size()),
    0, 1, in);

  REQUIRE(plan.dest_ranks == redev::LOs{0, 1});
  REQUIRE(plan.offsets == redev::LOs{0, 2, 4});
  REQUIRE(plan.permutation == std::vector<pcms::LO>{1, 3, 0, 2});
  REQUIRE(plan.msg_size == 4);
}

TEST_CASE("GID messages insert headers around compact field payloads",
          "[field_exchange]")
{
  Kokkos::View<pcms::Real**> coords("coords", 4, 2);
  pcms::PointCloudLayout layout(2, coords, pcms::CoordinateSystem::Cartesian);

  pcms::ExchangePlan plan;
  plan.dest_ranks = {0, 1};
  plan.offsets = {0, 2, 4};
  plan.permutation = {0, 2, 1, 3};
  plan.msg_size = 4;

  Kokkos::View<pcms::GO*, pcms::HostMemorySpace> message(
    "gid_message", plan.msg_size + 2 * pcms::ent_offsets_len);
  pcms::GenericFieldExchangePlanner planner;
  planner.FillGidMessage(layout, plan,
                         pcms::Rank1View<pcms::GO, pcms::HostMemorySpace>(
                           message.data(), message.size()));

  const pcms::GO expected[] = {
    0, 2, 2, 2, 2, 0, 2, 0, 2, 2, 2, 2, 1, 3,
  };
  for (size_t i = 0; i < message.size(); ++i) {
    CAPTURE(i);
    REQUIRE(message(i) == expected[i]);
  }
}
