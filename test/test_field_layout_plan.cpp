#include <catch2/catch_test_macros.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_library.hpp>
#include <redev.h>
#include "pcms/create_field.h"
#include "pcms/field_layout.h"
#include <vector>

TEST_CASE("omega_h field layout builds a plan", "[field_layout]")
{
  auto lib = Omega_h::Library{};
  auto world = lib.world();
  auto mesh_storage =
    Omega_h::build_box(world, OMEGA_H_SIMPLEX, 1, 1, 0, 1, 1, 0, false);
  const auto& mesh = mesh_storage;
  auto layout = pcms::CreateLagrangeLayout(mesh_storage, 1, 1,
                                           pcms::CoordinateSystem::Cartesian);

  auto class_ids = mesh.get_array<Omega_h::ClassId>(0, "class_id");
  auto class_dims = mesh.get_array<Omega_h::I8>(0, "class_dim");
  auto class_ids_h = Omega_h::HostRead(class_ids);
  auto class_dims_h = Omega_h::HostRead(class_dims);

  redev::LOs ranks;
  ranks.reserve(class_ids_h.size());
  redev::ClassPtn::ModelEntVec ents;
  ents.reserve(class_ids_h.size());
  for (int i = 0; i < class_ids_h.size(); ++i) {
    ranks.push_back(0);
    ents.emplace_back(redev::ClassPtn::ModelEnt({class_dims_h[i], class_ids_h[i]}));
  }

  auto partition =
    redev::Partition{redev::ClassPtn(world->get_impl(), ranks, ents)};
  const auto plan = layout->BuildClientPlan(partition);

  REQUIRE(plan.destinations.size() == 1);
  REQUIRE(plan.offsets.size() == 2);
  REQUIRE(plan.permutation.size() ==
          static_cast<size_t>(layout->GetNumOwnedDofHolder()));
  REQUIRE(plan.gid_payload.size() >= pcms::ent_offsets_len);

  auto gids = layout->GetGids();
  std::vector<pcms::GO> recovered(gids.size());
  for (size_t i = 0; i < gids.size(); ++i) {
    const auto slot = static_cast<size_t>(plan.permutation[i]);
    REQUIRE(slot < plan.gid_payload.size());
    recovered[i] = plan.gid_payload[slot];
  }
  for (size_t i = 0; i < gids.size(); ++i) {
    REQUIRE(recovered[i] == gids[i]);
  }

  const auto ent_offsets = layout->GetEntOffsets();
  for (int i = 0; i < pcms::ent_offsets_len; ++i) {
    REQUIRE(plan.gid_payload[i] == static_cast<pcms::GO>(ent_offsets[i]));
  }
}
