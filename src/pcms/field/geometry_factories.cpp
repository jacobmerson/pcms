#include "pcms/configuration.hpp"
#include "pcms/geometry/geometry.hpp"
#include "pcms/backends/point_cloud/geometry.hpp"
#include "pcms/backends/uniform_grid/geometry.hpp"
#ifdef PCMS_ENABLE_OMEGA_H
#include "pcms/backends/omega_h/discretization.hpp"
#include "pcms/backends/omega_h/entity_layout.hpp"
#include "pcms/backends/omega_h/point_cloud_geometry.hpp"
#endif

namespace pcms
{
using backend::point_cloud::PointCloudGeometry;
using backend::point_cloud::PointCloudLayout;
using backend::uniform_grid::UniformGridGeometry;
} // namespace pcms

namespace pcms
{

/// The geometry of a 2D uniform grid.
/// @param grid the grid
/// @param system the grid's coordinate system, or null for Cartesian
std::shared_ptr<Geometry> CreateUniformGridGeometry(
  UniformGrid<2> grid, std::shared_ptr<const CoordinateSystem> system)
{
  return std::make_shared<UniformGridGeometry<2>>(grid, std::move(system));
}

/// The geometry of a 3D uniform grid.
/// @param grid the grid
/// @param system the grid's coordinate system, or null for Cartesian
std::shared_ptr<Geometry> CreateUniformGridGeometry(
  UniformGrid<3> grid, std::shared_ptr<const CoordinateSystem> system)
{
  return std::make_shared<UniformGridGeometry<3>>(grid, std::move(system));
}

/// A point-cloud geometry whose sites are `coords` (copied to the device).
/// @param coords the sites
/// @param options the support search of the geometry
/// @throws pcms_error if the coordinate system does not have the identity
///         metric (MLS distances are Euclidean)
std::shared_ptr<Geometry> CreatePointCloudGeometry(
  CoordinateView<HostMemorySpace> coords, NeighbourhoodOptions options)
{
  const auto values = coords.GetValues();
  const int dim = static_cast<int>(values.extent(1));
  Kokkos::View<const Real**, Kokkos::HostSpace> host_view(
    values.data_handle(), values.extent(0), values.extent(1));
  auto device_view = Kokkos::View<Real**>("device_view", host_view.extent(0),
                                          host_view.extent(1));
  DeepCopyMismatchLayouts(device_view, host_view);
  auto layout = std::make_shared<PointCloudLayout>(
    dim, device_view, coords.GetCoordinateSystem());
  return std::make_shared<PointCloudGeometry>(std::move(layout),
                                              std::move(options));
}

/// A point-cloud geometry whose sites are the DOF holders of `sites` (e.g. the
/// entities of one dimension of a mesh, through OmegaHEntityLayout, which the
/// adjacency search then serves).
/// @param sites the layout of the sites
/// @param options the support search of the geometry
/// @throws pcms_error if `sites` is null or its coordinate system does not
///         have the identity metric
std::shared_ptr<Geometry> CreatePointCloudGeometry(
  std::shared_ptr<const FieldLayout> sites, NeighbourhoodOptions options)
{
  if (sites == nullptr) {
    throw pcms_error("CreatePointCloudGeometry: the sites must not be null");
  }
#ifdef PCMS_ENABLE_OMEGA_H
  // the entities of a mesh: the mesh-sited variant, whose supports use the
  // mesh adjacency
  if (dynamic_cast<const backend::omega_h::OmegaHEntityLayout*>(sites.get()) !=
      nullptr) {
    const auto discretization =
      std::dynamic_pointer_cast<const backend::omega_h::OmegaHDiscretization>(
        detail::DOFHolderDiscretization(*sites));
    PCMS_ALWAYS_ASSERT(discretization != nullptr);
    const int site_entity_dim = sites->GetDOFHolderEntityDim();
    return std::make_shared<backend::omega_h::OmegaHPointCloudGeometry>(
      std::move(sites), discretization->GetMesh(), site_entity_dim,
      std::move(options));
  }
#endif
  return std::make_shared<PointCloudGeometry>(std::move(sites),
                                              std::move(options));
}

} // namespace pcms
