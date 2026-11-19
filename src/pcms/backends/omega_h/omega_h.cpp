#include "pcms/backends/omega_h/omega_h.hpp"
#include "pcms/backends/omega_h/entity_layout.hpp"
#include "pcms/backends/omega_h/omega_h_geometry.hpp"
#include "pcms/backends/point_cloud/geometry.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/detail/memory_spaces.hpp"

namespace pcms::omega_h
{

namespace
{

template <int Dim>
Kokkos::View<const LO*, DeviceMemorySpace> LocatedElementsOf(
  const pcms::detail::OmegaHMeshBinding& binding, Omega_h::Mesh& mesh)
{
  const auto found = binding.FoundPoints();
  const auto results = binding.Results<Dim>();
  const auto up = detail::UpAdjacencies<Dim>(mesh);
  Kokkos::View<LO*, DeviceMemorySpace> elements(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "located_elements"),
    found.extent(0));
  pcms::detail::ForEach(
    "pcms::omega_h::LocatedElements", static_cast<LO>(found.extent(0)),
    KOKKOS_LAMBDA(const LO k) {
      elements(k) = detail::ElementOfResult<Dim>(up, results(found(k)));
    });
  return elements;
}

} // namespace

Kokkos::View<const LO*, DeviceMemorySpace> LocatedElements(
  const BoundPoints& bound)
{
  const auto* binding =
    dynamic_cast<const pcms::detail::OmegaHMeshBinding*>(&bound);
  const auto* geometry =
    dynamic_cast<const OmegaHGeometry*>(bound.GetGeometry().get());
  if (binding == nullptr || geometry == nullptr) {
    throw pcms_error("pcms::omega_h::LocatedElements: the points were not "
                     "bound on an Omega_h geometry");
  }
  Omega_h::Mesh& mesh = geometry->GetMesh();
  return mesh.dim() == 2 ? LocatedElementsOf<2>(*binding, mesh)
                         : LocatedElementsOf<3>(*binding, mesh);
}

std::shared_ptr<const Geometry> CreateGeometry(
  Omega_h::Mesh& mesh, std::shared_ptr<const CoordinateSystem> system,
  OmegaHGeometryOptions options)
{
  return std::make_shared<OmegaHGeometry>(mesh, std::move(system),
                                          std::move(options));
}

std::shared_ptr<const Geometry> CreateGeometry(
  std::shared_ptr<Omega_h::Mesh> mesh,
  std::shared_ptr<Omega_h::Library> library,
  std::shared_ptr<const CoordinateSystem> system, OmegaHGeometryOptions options)
{
  return std::make_shared<OmegaHGeometry>(
    std::move(mesh), std::move(library), std::move(system), std::move(options));
}

std::shared_ptr<const Geometry> CreatePointCloudGeometry(
  std::shared_ptr<const Geometry> geometry, int entity_dim,
  NeighbourhoodOptions options)
{
  if (geometry == nullptr) {
    throw pcms_error("pcms::omega_h::CreatePointCloudGeometry: the "
                     "geometry must not be null");
  }
  auto mesh_geometry =
    std::dynamic_pointer_cast<const OmegaHGeometry>(std::move(geometry));
  if (mesh_geometry == nullptr) {
    throw pcms_error("pcms::omega_h::CreatePointCloudGeometry: the "
                     "geometry must be an Omega_h geometry");
  }
  if (entity_dim != 0 && entity_dim != mesh_geometry->GetMesh().dim()) {
    throw pcms_error("pcms::omega_h::CreatePointCloudGeometry: "
                     "entity_dim must be 0 (vertices) or the mesh dimension "
                     "(element centroids)");
  }
  return point_cloud::CreateGeometry(std::make_shared<OmegaHEntityLayout>(
                                       std::move(mesh_geometry), entity_dim, 1),
                                     std::move(options));
}

} // namespace pcms::omega_h
