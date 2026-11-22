#include <catch2/catch_test_macros.hpp>
#include "pcms/backends/xgc/field_data.hpp"
#include "pcms/coupler/overlap_mask.hpp"
#include "pcms/backends/xgc/field_factory.hpp"
#include <numeric>

namespace pcms
{
using xgc::XGCFieldData;
using xgc::XGCFieldFactory;
using xgc::XGCFieldLayout;
} // namespace pcms

namespace
{

pcms::xgc::ReverseClassificationVertex create_dummy_rc(int size)
{
  pcms::xgc::ReverseClassificationVertex rc;
  for (int i = 0; i < size; ++i) {
    if (i % 4 == 0) {
      pcms::xgc::detail::ReverseClassificationAccess::Insert(rc, {0, 0}, i);
    } else {
      pcms::xgc::detail::ReverseClassificationAccess::Insert(rc, {0, 1}, i);
    }
  }
  return rc;
}

bool in_overlap(int, int id)
{
  return id == 0;
}

} // namespace

TEST_CASE("XGC FieldLayout owns and classifies every entry")
{
  static constexpr int data_size = 16;
  auto rc = create_dummy_rc(data_size);
  pcms::XGCFieldLayout layout(rc, data_size);

  Kokkos::View<bool*, pcms::HostMemorySpace> owned("owned", data_size);
  Kokkos::View<pcms::LO*, pcms::HostMemorySpace> class_dims("class_dims",
                                                            data_size);
  Kokkos::View<pcms::LO*, pcms::HostMemorySpace> class_ids("class_ids",
                                                           data_size);
  pcms::detail::CopyDeviceRank1ViewToHostView<bool>(owned, layout.GetOwned());
  pcms::detail::CopyDeviceRank1ViewToHostView<pcms::LO>(
    class_dims, layout.GetDOFHolderClassificationDimensions());
  pcms::detail::CopyDeviceRank1ViewToHostView<pcms::LO>(
    class_ids, layout.GetDOFHolderClassificationIds());
  auto gids = layout.GetGidsHost();
  const pcms::OverlapMask overlap_mask(in_overlap);
  auto overlap = overlap_mask.GetMask(layout);

  for (int i = 0; i < data_size; ++i) {
    REQUIRE(gids[i] == i + 1);
    REQUIRE(owned[i]);
    REQUIRE(class_dims[i] == 0);
    REQUIRE(class_ids[i] == (i % 4 == 0 ? 0 : 1));
    REQUIRE(overlap[i] == (i % 4 == 0));
  }
}

TEST_CASE("XGCFieldFactory creates fields and rejects evaluator access")
{
  static constexpr int data_size = 16;
  auto rc = create_dummy_rc(data_size);
  pcms::XGCFieldFactory function_space(rc, data_size);

  std::vector<pcms::Real> data(data_size);
  std::iota(data.begin(), data.end(), 0.0);
  auto field = function_space.CreateField<pcms::Real>(
    "", std::make_unique<pcms::XGCFieldData<pcms::Real>>(
          function_space.GetXGCLayout(), pcms::ValueBasis{},
          pcms::detail::make_array_view(data)));

  REQUIRE(&field.GetLayout() == function_space.GetLayout().get());
}
