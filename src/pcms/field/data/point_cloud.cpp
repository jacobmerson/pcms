#include "pcms/field/data/point_cloud.h"
#include "pcms/field/layout/point_cloud.h"
#include "pcms/utility/profile.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/arrays.h"

namespace pcms
{

PointCloud::PointCloud(std::shared_ptr<const PointCloudLayout> layout)
  : layout_(std::move(layout)),
    metadata_{},
    device_data_("",
                 layout_->GetDOFHolderCoordinates().GetValues().extent(0)),
    data_host_("",
               layout_->GetDOFHolderCoordinates().GetValues().extent(0))
{
}

const FieldMetadata& PointCloud::GetMetadata() const
{
  return metadata_;
}

Rank1View<const Real, HostMemorySpace> PointCloud::GetDOFHolderDataHost() const
{
  Kokkos::deep_copy(data_host_, device_data_);
  return make_const_array_view(data_host_);
}

void PointCloud::SetDOFHolderDataHost(
  Rank1View<const Real, HostMemorySpace> data)
{
  PCMS_FUNCTION_TIMER;
  PCMS_ALWAYS_ASSERT(data.size() == device_data_.size());
  CopyHostRank1ViewToDeviceView(device_data_, data);
}

Rank1View<const Real, DeviceMemorySpace> PointCloud::GetDOFHolderData() const
{
  return make_const_array_view(device_data_);
}

void PointCloud::SetDOFHolderData(Rank1View<const Real, DeviceMemorySpace> data)
{
  PCMS_FUNCTION_TIMER;
  PCMS_ALWAYS_ASSERT(data.size() == device_data_.size());
  CopyDeviceRank1ViewToDeviceView(device_data_, data);
}

} // namespace pcms
