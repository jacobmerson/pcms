#ifndef PCMS_BACKENDS_OMEGA_H_POINT_CLOUD_GEOMETRY_HPP
#define PCMS_BACKENDS_OMEGA_H_POINT_CLOUD_GEOMETRY_HPP

#include "pcms/backends/omega_h/adj_search.hpp"
#include "pcms/backends/omega_h/geometry.hpp"
#include "pcms/backends/point_cloud/geometry.hpp"
#include "pcms/utility/entity_types.hpp"
#include "pcms/utility/omega_h_array_utils.hpp"

#include <Omega_h_mesh.hpp>

#include <memory>

namespace pcms::backend::omega_h
{

/// The point-cloud geometry of the entities of one dimension of an Omega_h
/// mesh (its vertices or its element centroids, the DOF holders of an
/// OmegaHEntityLayout). Neighbourhood is the mesh-adjacency search when
/// `owners` gives the cell or vertex of the mesh at each point, and for
/// vertex sites, which otherwise locate the points first with the mesh's
/// OmegaHGeometry; the distance search otherwise. Created by
/// CreatePointCloudGeometry for mesh-entity sites.
class OmegaHPointCloudGeometry final : public point_cloud::PointCloudGeometry
{
public:
  /// @param mesh the mesh whose entities of dimension `site_entity_dim` are
  ///        the sites
  OmegaHPointCloudGeometry(std::shared_ptr<const FieldLayout> site_layout,
                           Omega_h::Mesh& mesh, int site_entity_dim,
                           NeighbourhoodOptions options)
    : PointCloudGeometry(std::move(site_layout), std::move(options)),
      mesh_(&mesh),
      site_entity_dim_(site_entity_dim),
      vertex_geometry_(site_entity_dim == Vertex
                         ? std::make_shared<OmegaHGeometry>(
                             mesh, GetSiteLayout()->GetCoordinateSystem())
                         : nullptr)
  {
  }

  [[nodiscard]] SupportResults Neighbourhood(
    CoordinateView<DeviceMemorySpace> coords,
    const NeighbourhoodOptions& options,
    const KnownOwners& owners = {}) const override
  {
    pcms::detail::CheckNeighbourhoodArguments(
      *this, coords, options, "PointCloudGeometry::Neighbourhood");
    const auto n = static_cast<LO>(coords.GetValues().extent(0));
    const auto mesh_entity = [this](int dim) {
      return dim == Vertex || dim == mesh_->dim();
    };
    // the supports are entity ids, which are site indices only without ghosts
    if (mesh_entity(site_entity_dim_) && !GetSiteLayout()->HasGhosts() &&
        owners.Hints(*this, n) && mesh_entity(owners.entity_dim)) {
      pcms::detail::CheckEntityIds(
        owners.entities, mesh_->nents(owners.entity_dim),
        "PointCloudGeometry::Neighbourhood",
        owners.entity_dim == Vertex ? "vertex" : "element");
      return searchNeighbors(
        *mesh_, site_entity_dim_,
        utility::flatten_to_omega_h_reals(coords.GetValues()),
        owners.entity_dim, owners.entities, options.radius * options.radius,
        options.min_supports, 3 * options.min_supports, options.adapt_radius);
    }
    if (vertex_geometry_ != nullptr) {
      return vertex_geometry_->Neighbourhood(std::move(coords), options);
    }
    return DistanceNeighbourhood(coords, options);
  }

private:
  Omega_h::Mesh* mesh_;
  int site_entity_dim_;
  // for mesh-vertex sites, the mesh's geometry: its Neighbourhood is the
  // adjacency search from the element its search finds for each point
  std::shared_ptr<const OmegaHGeometry> vertex_geometry_;
};

} // namespace pcms::backend::omega_h

#endif // PCMS_BACKENDS_OMEGA_H_POINT_CLOUD_GEOMETRY_HPP
