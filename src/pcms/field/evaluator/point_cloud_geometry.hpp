#ifndef PCMS_FIELD_EVALUATOR_POINT_CLOUD_GEOMETRY_HPP
#define PCMS_FIELD_EVALUATOR_POINT_CLOUD_GEOMETRY_HPP

#include "pcms/field/coordinate_system.hpp"
#include "pcms/field/evaluator/mls_point_cloud.h"
#include "pcms/field/evaluator/omega_h_geometry.hpp"
#include "pcms/field/field_layout.h"
#include "pcms/field/layout/omega_h_entity.h"
#include "pcms/field/layout/point_cloud.h"
#include "pcms/geometry/geometry.hpp"
#include "pcms/localization/adj_search.hpp"
#include "pcms/localization/mls_support_helpers.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/entity_types.h"
#include "pcms/utility/omega_h_array_utils.h"
#include "pcms/utility/profile.h"

#include <Omega_h_array.hpp>
#include <Omega_h_mesh.hpp>
#include <memory>
#include <string>

namespace pcms
{

/// Geometry of a set of sites (the DOF holders of a layout: a point cloud or
/// the entities of one dimension of a mesh): the identity map, located by the
/// MLS supports of each query point. Bind finds the supports with
/// Neighbourhood, using the options the geometry was created with. For the
/// vertices or the cells of a mesh, construction builds the mesh's
/// OmegaHGeometry, whose search starts the adjacency search of each point.
/// @throws pcms_error on construction if the sites' coordinate system does not
///         have the identity metric
class PointCloudGeometry final : public Geometry
{
public:
  /// Own it by std::shared_ptr, as CreatePointCloudGeometry does: Bind uses
  /// shared_from_this().
  /// @param mesh the mesh whose entities of dimension `site_entity_dim` are
  ///        the sites, or null for a point cloud
  PointCloudGeometry(std::shared_ptr<const FieldLayout> site_layout,
                     Omega_h::Mesh* mesh, int site_entity_dim,
                     NeighbourhoodOptions options)
    : site_layout_(std::move(site_layout)),
      mesh_(mesh),
      site_entity_dim_(site_entity_dim),
      options_(std::move(options)),
      mesh_geometry_(mesh != nullptr && (site_entity_dim == Vertex ||
                                         site_entity_dim == mesh->dim())
                       ? std::make_shared<OmegaHGeometry>(
                           *mesh, site_layout_->GetCoordinateSystem())
                       : nullptr)
  {
    // MLS distances are based on the unweighted Euclidean measure, which is
    // valid only with the identity metric; this almost certainly should be
    // generalized to take the metric for distances
    if (!HasIdentityMetric(*site_layout_->GetCoordinateSystem())) {
      throw pcms_error(
        "PointCloudGeometry: MLS point-cloud evaluation requires a "
        "coordinate system whose metric is the identity; got '" +
        std::string(site_layout_->GetCoordinateSystem()->Kind()) + "'");
    }
  }

  [[nodiscard]] std::shared_ptr<const Discretization> GetDiscretization()
    const noexcept override
  {
    return site_layout_->GetDiscretization();
  }

  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>&
  GetCoordinateSystem() const noexcept override
  {
    return site_layout_->GetCoordinateSystem();
  }

  [[nodiscard]] const std::shared_ptr<const FieldLayout>& GetSiteLayout()
    const noexcept
  {
    return site_layout_;
  }

  /// The mesh-adjacency search for the sites of a mesh (its vertices or its
  /// element centroids) when `owners` claims the cell or vertex of the mesh at
  /// each point, and for mesh-vertex sites, which otherwise locate the points
  /// first; the distance search otherwise. The search starts from the element
  /// the mesh geometry's search finds for each point, starting from the claim
  /// (verified, see OwnerHints).
  [[nodiscard]] SupportResults Neighbourhood(
    CoordinateView<DeviceMemorySpace> coords,
    const NeighbourhoodOptions& options,
    const OwnerHints& owners = {}) const override
  {
    detail::CheckNeighbourhoodArguments(*this, coords, options,
                                        "PointCloudGeometry::Neighbourhood");
    const auto n = static_cast<LO>(coords.GetValues().extent(0));
    // the supports are entity ids, which are site indices only without ghosts
    if (mesh_geometry_ != nullptr && !site_layout_->HasGhosts() &&
        owners.Names(*this, n) &&
        (owners.entity_dim == Vertex || owners.entity_dim == mesh_->dim())) {
      const int dim = mesh_->dim();
      const auto flat_points = flatten_to_omega_h_reals(coords.GetValues());
      const auto seeds = mesh_geometry_->FindElementSeeds(
        ConvertCoordsTo2D(flat_points, n, dim),
        detail::ElementGuesses(*mesh_, owners,
                               "PointCloudGeometry::Neighbourhood"));
      return searchNeighbors(*mesh_, site_entity_dim_, flat_points, dim, seeds,
                             options.radius * options.radius,
                             options.min_supports, 3 * options.min_supports,
                             options.adapt_radius);
    }
    if (mesh_geometry_ != nullptr && site_entity_dim_ == Vertex) {
      return mesh_geometry_->Neighbourhood(std::move(coords), options);
    }
    const auto sites = site_layout_->GetDOFHolderCoordinates().GetValues();
    return BuildPointCloudSupports(
      flatten_to_omega_h_reals(sites, "src_coords"),
      flatten_to_omega_h_reals(coords.GetValues(), "tgt_coords"),
      static_cast<int>(sites.extent(1)), options.radius, options.min_supports,
      options.adapt_radius);
  }

protected:
  std::shared_ptr<BoundSource> BindImpl(
    const BindRequest& request) const override
  {
    PCMS_FUNCTION_TIMER;
    // Extract source coordinates for the MLS solve.
    const auto src_view = site_layout_->GetDOFHolderCoordinates().GetValues();
    Omega_h::Reals source_coords =
      flatten_to_omega_h_reals(src_view, "src_coords");

    SupportResults supports =
      Neighbourhood(request.coords, options_, request.owners);

    return std::make_shared<MLSBinding>(shared_from_this(), request,
                                        std::move(source_coords),
                                        std::move(supports));
  }

private:
  std::shared_ptr<const FieldLayout> site_layout_;
  Omega_h::Mesh* mesh_;
  int site_entity_dim_;
  NeighbourhoodOptions options_;
  // for mesh-vertex or cell sites, the mesh's geometry: its search finds the
  // element the adjacency search of each point starts from
  std::shared_ptr<const OmegaHGeometry> mesh_geometry_;
};

/// A point-cloud geometry whose sites are `coords` (copied to the device).
/// @param coords the sites
/// @param options the support search of the geometry
/// @throws pcms_error if the coordinate system does not have the identity
///         metric (MLS distances are Euclidean)
inline std::shared_ptr<Geometry> CreatePointCloudGeometry(
  CoordinateView<HostMemorySpace> coords, NeighbourhoodOptions options = {})
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
  return std::make_shared<PointCloudGeometry>(std::move(layout), nullptr, -1,
                                              std::move(options));
}

/// A point-cloud geometry whose sites are the DOF holders of `sites` (e.g. the
/// entities of one dimension of a mesh, through OmegaHEntityLayout, which the
/// adjacency search then serves).
/// @param sites the layout of the sites
/// @param options the support search of the geometry
/// @throws pcms_error if `sites` is null or its coordinate system does not
///         have the identity metric
inline std::shared_ptr<Geometry> CreatePointCloudGeometry(
  std::shared_ptr<const FieldLayout> sites, NeighbourhoodOptions options = {})
{
  if (sites == nullptr) {
    throw pcms_error("CreatePointCloudGeometry: the sites must not be null");
  }
  Omega_h::Mesh* mesh = nullptr;
  int site_entity_dim = -1;
  if (dynamic_cast<const OmegaHEntityLayout*>(sites.get()) != nullptr) {
    const auto discretization =
      std::dynamic_pointer_cast<const OmegaHDiscretization>(
        sites->GetDiscretization());
    PCMS_ALWAYS_ASSERT(discretization != nullptr);
    mesh = &discretization->GetMesh();
    site_entity_dim = sites->GetDOFHolderEntityDim();
  }
  return std::make_shared<PointCloudGeometry>(
    std::move(sites), mesh, site_entity_dim, std::move(options));
}

} // namespace pcms

#endif // PCMS_FIELD_EVALUATOR_POINT_CLOUD_GEOMETRY_HPP
