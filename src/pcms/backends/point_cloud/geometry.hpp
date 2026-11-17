#ifndef PCMS_BACKENDS_POINT_CLOUD_GEOMETRY_HPP
#define PCMS_BACKENDS_POINT_CLOUD_GEOMETRY_HPP

#include "pcms/utility/coordinate_system.hpp"
#include "pcms/backends/point_cloud/mls_evaluator.hpp"
#include "pcms/field/field_layout.hpp"
#include "pcms/backends/point_cloud/layout.hpp"
#include "pcms/geometry/geometry.hpp"
#include "pcms/localization/mls_support_helpers.hpp"
#include "pcms/utility/arrays.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/entity_types.hpp"
#include "pcms/utility/detail/profile.hpp"

#include <memory>
#include <string>

namespace pcms::backend::point_cloud
{

/// Geometry of a set of sites (the DOF holders of a layout): the identity
/// map, located by the MLS supports of each query point. Bind finds the
/// supports with Neighbourhood, using the options the geometry was created
/// with; the supports are found by distance. Sites that are the entities of
/// an Omega_h mesh use the mesh-sited variant of the Omega_h backend
/// (backend::omega_h::OmegaHPointCloudGeometry), whose Neighbourhood is the
/// mesh-adjacency search. Create it with CreatePointCloudGeometry
/// (pcms/geometry/geometry.hpp).
/// @throws pcms_error on construction if the sites' coordinate system does not
///         have the identity metric
class PointCloudGeometry : public Geometry
{
public:
  /// Own it by std::shared_ptr, as CreatePointCloudGeometry does: Bind uses
  /// shared_from_this().
  PointCloudGeometry(std::shared_ptr<const FieldLayout> site_layout,
                     NeighbourhoodOptions options)
    : site_layout_(std::move(site_layout)), options_(std::move(options))
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
    return pcms::detail::DOFHolderDiscretization(*site_layout_);
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

  /// The supports by distance (BuildPointCloudSupports); `owners` is
  /// ignored.
  [[nodiscard]] SupportResults Neighbourhood(
    CoordinateView<DeviceMemorySpace> coords,
    const NeighbourhoodOptions& options,
    const KnownOwners& /*owners*/ = {}) const override
  {
    pcms::detail::CheckNeighbourhoodArguments(
      *this, coords, options, "PointCloudGeometry::Neighbourhood");
    return DistanceNeighbourhood(coords, options);
  }

protected:
  /// The supports of the points `coords` among the sites by distance.
  [[nodiscard]] SupportResults DistanceNeighbourhood(
    const CoordinateView<DeviceMemorySpace>& coords,
    const NeighbourhoodOptions& options) const
  {
    const auto sites = site_layout_->GetDOFHolderCoordinates().GetValues();
    return localization::BuildPointCloudSupports(
      detail::FlattenRows(sites, "src_coords"),
      detail::FlattenRows(coords.GetValues(), "tgt_coords"),
      static_cast<int>(sites.extent(1)), options.radius, options.min_supports,
      options.adapt_radius);
  }

  std::shared_ptr<BoundSource> BindImpl(
    const BindRequest& request) const override
  {
    PCMS_FUNCTION_TIMER;
    // Extract source coordinates for the MLS solve.
    const auto src_view = site_layout_->GetDOFHolderCoordinates().GetValues();
    Kokkos::View<const Real*, DeviceMemorySpace> source_coords =
      detail::FlattenRows(src_view, "src_coords");

    SupportResults supports =
      Neighbourhood(request.coords, options_, request.owners);

    return std::make_shared<MLSBinding>(shared_from_this(), request,
                                        std::move(source_coords),
                                        std::move(supports), options_);
  }

private:
  std::shared_ptr<const FieldLayout> site_layout_;
  NeighbourhoodOptions options_;
};

} // namespace pcms::backend::point_cloud

#endif // PCMS_BACKENDS_POINT_CLOUD_GEOMETRY_HPP
