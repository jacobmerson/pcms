#include "pcms/backends/omega_h/omega_h.hpp"
#include "pcms/backends/omega_h/entity_layout.hpp"
#include "pcms/backends/omega_h/geometry.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/coordinate_systems/cartesian.hpp"

namespace pcms::backend::omega_h
{

std::shared_ptr<Geometry> CreateGeometry(
  Omega_h::Mesh& mesh, std::shared_ptr<const CoordinateSystem> system,
  OmegaHGeometryOptions options)
{
  return std::make_shared<OmegaHGeometry>(mesh, std::move(system),
                                          std::move(options));
}

Omega_h::Mesh& GetMesh(const Geometry& geometry)
{
  const auto* omega_h = dynamic_cast<const OmegaHGeometry*>(&geometry);
  if (omega_h == nullptr) {
    throw pcms_error("pcms::backend::omega_h::GetMesh: the geometry is not an "
                     "Omega_h geometry");
  }
  return omega_h->GetMesh();
}

std::shared_ptr<Geometry> CreatePointCloudGeometry(
  Omega_h::Mesh& mesh, int entity_dim,
  std::shared_ptr<const CoordinateSystem> system, NeighbourhoodOptions options)
{
  if (entity_dim < 0 || entity_dim > mesh.dim()) {
    throw pcms_error(
      "pcms::backend::omega_h::CreatePointCloudGeometry: entity_dim is out of "
      "range");
  }
  if (system == nullptr) {
    system = csys::Cartesian::Deferred();
  }
  return pcms::CreatePointCloudGeometry(
    std::make_shared<OmegaHEntityLayout>(mesh, entity_dim, 1,
                                         std::move(system)),
    std::move(options));
}

} // namespace pcms::backend::omega_h
