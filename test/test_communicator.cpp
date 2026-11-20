#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "pcms/configuration.hpp"
#include "pcms/utility/communicator.hpp"
#ifdef PCMS_ENABLE_MPI
#include "pcms/utility/mpi.hpp"
#endif

using pcms::Communicator;
using pcms::GO;
using pcms::LO;
using pcms::Real;

TEST_CASE("Self is one rank and its reductions are the identity",
          "[communicator]")
{
  const auto& self = Communicator::Self();
  REQUIRE(self != nullptr);
  REQUIRE(self == Communicator::Self());
  REQUIRE(self->Rank() == 0);
  REQUIRE(self->Size() == 1);
  REQUIRE(self->SumAll(Real{2.5}) == Catch::Approx(2.5));
  REQUIRE(self->MaxAll(Real{-1.5}) == Catch::Approx(-1.5));
  REQUIRE(self->SumAll(LO{3}) == 3);
  REQUIRE(self->MaxAll(GO{1} << 40) == (GO{1} << 40));
}

TEST_CASE("Split of Self", "[communicator]")
{
  const auto& self = Communicator::Self();
  const auto same = self->Split(0, 0);
  REQUIRE(same != nullptr);
  REQUIRE(same->Size() == 1);
  REQUIRE(self->Split(-1, 0) == nullptr);
}

#ifdef PCMS_ENABLE_MPI
// run on any number of ranks (the 2-rank ctest runs it on two)
TEST_CASE("MPI communicator", "[communicator]")
{
  int world_rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  auto comm = pcms::CreateCommunicator(MPI_COMM_WORLD);
  REQUIRE(comm->Rank() == world_rank);
  REQUIRE(comm->Size() == world_size);

  SECTION("a duplicate: same ranks, another message space")
  {
    int relation = MPI_UNEQUAL;
    MPI_Comm_compare(MPI_COMM_WORLD, pcms::GetMPIComm(*comm), &relation);
    REQUIRE(relation == MPI_CONGRUENT);
    REQUIRE(pcms::GetMPIComm(*Communicator::Self()) == MPI_COMM_SELF);
  }

  SECTION("reductions")
  {
    const Real n = world_size;
    REQUIRE(comm->SumAll(Real{world_rank + 1}) ==
            Catch::Approx(n * (n + 1) / 2));
    REQUIRE(comm->MaxAll(LO{world_rank}) == world_size - 1);
    REQUIRE(comm->SumAll(GO{1}) == world_size);
  }

  SECTION("Split by parity keeps the parent alive")
  {
    auto half = comm->Split(world_rank % 2, world_rank);
    const std::weak_ptr<const Communicator> parent = comm;
    comm.reset();
    REQUIRE_FALSE(parent.expired());
    const int expected = (world_size + 1 - world_rank % 2) / 2;
    REQUIRE(half->Size() == expected);
    REQUIRE(half->Rank() == world_rank / 2);
    REQUIRE(half->SumAll(LO{1}) == expected);
    half.reset();
    REQUIRE(parent.expired());
  }

  SECTION("a negative color leaves the rank out")
  {
    REQUIRE(comm->Split(-1, 0) == nullptr);
  }
}
#endif
