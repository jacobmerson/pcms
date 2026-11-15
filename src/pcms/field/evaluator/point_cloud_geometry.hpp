#ifndef PCMS_FIELD_EVALUATOR_POINT_CLOUD_GEOMETRY_HPP
#define PCMS_FIELD_EVALUATOR_POINT_CLOUD_GEOMETRY_HPP

#include "pcms/field/coordinate_system.hpp"
#include "pcms/field/evaluator/mls_options.h"
#include "pcms/field/evaluator/mls_point_cloud.h"
#include "pcms/field/field_layout.h"
#include "pcms/geometry/geometry.hpp"
#include "pcms/localization/localization_path_selection.h"
#include "pcms/localization/mesh_localization.h"
#include "pcms/localization/mls_support_helpers.h"
#include "pcms/utility/assert.h"
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
/// Neighbourhood, using the Localization() of the MLS options the geometry
/// was created with.
/// @throws pcms_error on construction if the sites' coordinate system does not
///         have the identity metric
class PointCloudGeometry final : public Geometry
{
public:
  [[nodiscard]] static std::shared_ptr<PointCloudGeometry> Create(
    std::shared_ptr<const FieldLayout> site_layout, MLSOptions options = {})
  {
    return std::make_shared<PointCloudGeometry>(std::move(site_layout),
                                                options);
  }

  [[nodiscard]] static std::shared_ptr<PointCloudGeometry> Create(
    std::shared_ptr<const FieldLayout> site_layout, Omega_h::Mesh& mesh,
    int site_entity_dim, MLSOptions options = {})
  {
    return std::make_shared<PointCloudGeometry>(std::move(site_layout), mesh,
                                                site_entity_dim, options);
  }

  /// Use Create or std::make_shared: Bind uses shared_from_this().
  PointCloudGeometry(std::shared_ptr<const FieldLayout> site_layout,
                     MLSOptions options)
    : PointCloudGeometry(std::move(site_layout), nullptr, -1, options)
  {
  }

  PointCloudGeometry(std::shared_ptr<const FieldLayout> site_layout,
                     Omega_h::Mesh& mesh, int site_entity_dim,
                     MLSOptions options)
    : PointCloudGeometry(std::move(site_layout), &mesh, site_entity_dim,
                         options)
  {
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

  [[nodiscard]] const NeighbourhoodOptions& GetNeighbourhoodOptions()
    const noexcept
  {
    return options_;
  }

  /// The mesh-adjacency search for mesh-entity sites, the distance search
  /// for a point cloud. When `options.query_layout` holds the vertices of the
  /// sites' mesh and the sites are its element centroids, the supports are
  /// found from the mesh adjacency alone and `coords` is taken to be those
  /// vertices (it is not read).
  [[nodiscard]] SupportResults Neighbourhood(
    CoordinateView<DeviceMemorySpace> coords,
    const NeighbourhoodOptions& options) const override
  {
    detail::CheckNeighbourhoodArguments(*this, coords, options,
                                        "PointCloudGeometry::Neighbourhood");
    const MLSOptions search = MLSOptions::FromLocalization(options);
    if (mesh_ != nullptr) {
      AdjacencyLocalizationFactory adjacency(*mesh_, site_entity_dim_, search);
      if (options.query_layout != nullptr &&
          detail::SelectLocalizationPath(*site_layout_,
                                         options.query_layout.get()) ==
            detail::LocalizationPath::CentroidToVertexAdjacencySearch) {
        return adjacency.BuildSameMeshCentroidToVertex();
      }
      return adjacency.Build(std::move(coords));
    }
    const auto sites = site_layout_->GetDOFHolderCoordinates().GetValues();
    const int dim = static_cast<int>(sites.extent(1));
    const auto query = coords.GetValues();
    PCMS_ALWAYS_ASSERT(static_cast<int>(query.extent(1)) == dim);
    return BuildPointCloudSupports(
      flatten_to_omega_h_reals(sites, "src_coords"),
      flatten_to_omega_h_reals(query, "tgt_coords"), dim, search.radius,
      search.min_req_supports, search.adapt_radius);
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

    NeighbourhoodOptions options = options_;
    options.query_layout = request.query_layout;
    SupportResults supports = Neighbourhood(request.coords, options);

    return std::make_shared<MLSBinding>(shared_from_this(), request,
                                        std::move(source_coords),
                                        std::move(supports));
  }

  std::shared_ptr<BoundSource> BindParametricImpl(
    const ParametricPoints& /*points*/,
    OutOfBoundsPolicy /*policy*/) const override
  {
    throw pcms_error(
      "PointCloudGeometry::Bind: a point cloud has no parametric coordinates");
  }

private:
  PointCloudGeometry(std::shared_ptr<const FieldLayout> site_layout,
                     Omega_h::Mesh* mesh, int site_entity_dim,
                     MLSOptions options)
    : site_layout_(std::move(site_layout)),
      mesh_(mesh),
      site_entity_dim_(site_entity_dim),
      options_(options.Localization())
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

  std::shared_ptr<const FieldLayout> site_layout_;
  Omega_h::Mesh* mesh_;
  int site_entity_dim_;
  NeighbourhoodOptions options_;
};

} // namespace pcms

#endif // PCMS_FIELD_EVALUATOR_POINT_CLOUD_GEOMETRY_HPP
