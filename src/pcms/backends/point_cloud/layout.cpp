#include "pcms/backends/point_cloud/layout.hpp"
#include "pcms/utility/arrays.hpp"
#include "pcms/utility/entity_types.hpp"
#include <memory>
#include <Kokkos_StdAlgorithms.hpp>

namespace pcms::detail
{

namespace
{

void InitializePointCloudClassification(
  Kokkos::View<LO*, DeviceMemorySpace> classification_dims,
  Kokkos::View<LO*, DeviceMemorySpace> classification_ids, LO n,
  LO classification_entity_dim)
{
  Kokkos::deep_copy(classification_dims, classification_entity_dim);
  // for (LO i = 0; i < n; ++i) {
  //   classification_ids_host(i) = i;
  // }
  Kokkos::parallel_for(n, KOKKOS_LAMBDA(LO i) { classification_ids(i) = i; });
}

} // namespace

PointCloudLayout::PointCloudLayout(
  int dim, Kokkos::View<Real**> coords,
  std::shared_ptr<const CoordinateSystem> coordinate_system)
  : dim_(dim),
    coords_(coords),
    owned_("", coords.extent(0)),
    gids_("", coords.extent(0)),
    gids_host_("", coords.extent(0))
{
  SetCoordinateSystem(pcms::detail::ResolveCoordinateSystem(
    std::move(coordinate_system), static_cast<int>(coords_.extent(1))));
  components_ = 1;

  namespace KE = Kokkos::Experimental;
  KE::fill(Kokkos::DefaultExecutionSpace(), owned_, true);
  pcms::detail::iota_view(gids_);

  LO n = static_cast<LO>(coords.extent(0));
  classification_dims_ =
    Kokkos::View<LO*, DeviceMemorySpace>("classification_dims", n);
  classification_ids_ =
    Kokkos::View<LO*, DeviceMemorySpace>("classification_ids", n);
  InitializePointCloudClassification(classification_dims_, classification_ids_,
                                     n, Vertex);

  Kokkos::deep_copy(gids_host_, gids_);
}

int PointCloudLayout::GetNumComponents() const
{
  return components_;
}

LO PointCloudLayout::GetNumDofHolder() const
{
  return coords_.extent(0);
}

GO PointCloudLayout::GetNumGlobalDofHolder() const
{
  return coords_.extent(0);
}

Rank1View<const bool, DeviceMemorySpace> PointCloudLayout::GetOwned() const
{
  return pcms::detail::make_const_array_view(owned_);
}

GlobalIDView<HostMemorySpace> PointCloudLayout::GetGidsHost() const
{
  return GlobalIDView<HostMemorySpace>(gids_host_.data(), gids_host_.size());
}

CoordinateView<DeviceMemorySpace> PointCloudLayout::GetDOFHolderCoordinates()
  const
{
  auto coords_view = MakeConstRank2View(coords_);
  return CoordinateView<DeviceMemorySpace>{GetCoordinateSystem(), coords_view};
}

bool PointCloudLayout::IsDistributed() const
{
  return false;
}

size_t PointCloudLayout::GetNumEnts() const
{
  return coords_.extent(0);
}

EntOffsetsArray PointCloudLayout::GetEntOffsets() const
{
  EntOffsetsArray offsets{};
  for (size_t i = 0; i < offsets.size(); ++i)
    offsets[i] = coords_.extent(0);
  offsets[0] = 0;
  return offsets;
}

std::array<int, 4> PointCloudLayout::GetNodesPerDim() const
{
  std::array<int, 4> nodes{};
  for (size_t i = 0; i < nodes.size(); ++i)
    nodes[i] = 0;
  nodes[0] = 1;
  return nodes;
}

Kokkos::View<const Real**, DeviceMemorySpace> PointCloudLayout::GetCoordinates()
  const
{
  return coords_;
}

int PointCloudLayout::GetDimension() const
{
  return dim_;
}

Rank1View<const LO, DeviceMemorySpace>
PointCloudLayout::GetDOFHolderClassificationDimensions() const
{
  return pcms::detail::make_const_array_view(classification_dims_);
}

Rank1View<const LO, DeviceMemorySpace>
PointCloudLayout::GetDOFHolderClassificationIds() const
{
  return pcms::detail::make_const_array_view(classification_ids_);
}

} // namespace pcms::detail
