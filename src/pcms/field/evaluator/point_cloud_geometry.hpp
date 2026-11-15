#ifndef PCMS_FIELD_EVALUATOR_POINT_CLOUD_GEOMETRY_HPP
#define PCMS_FIELD_EVALUATOR_POINT_CLOUD_GEOMETRY_HPP

#include "pcms/field/coordinate_system.hpp"
#include "pcms/field/evaluator/mls_options.h"
#include "pcms/field/evaluator/mls_point_cloud.h"
#include "pcms/field/field_layout.h"
#include "pcms/geometry/geometry.hpp"
#include "pcms/localization/localization_factory.h"
#include "pcms/localization/localization_path_selection.h"
#include "pcms/localization/mesh_localization.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/omega_h_array_utils.h"
#include "pcms/utility/profile.h"

#include <Omega_h_array.hpp>
#include <memory>
#include <string>

namespace pcms
{

/// @brief Geometry of a set of sites (a point cloud or mesh entities): the
///        identity map, located by the MLS supports of each query point.
class PointCloudGeometry final : public Geometry
{
public:
  /// @brief Creates the geometry owned by a std::shared_ptr, as Bind needs.
  /// @param site_layout layout whose DOF holders are the sites
  /// @param localization search for the MLS supports of query points
  [[nodiscard]] static std::shared_ptr<PointCloudGeometry> Create(
    std::shared_ptr<const FieldLayout> site_layout,
    std::shared_ptr<LocalizationFactory> localization)
  {
    return std::make_shared<PointCloudGeometry>(std::move(site_layout),
                                                std::move(localization));
  }

  /// @brief Create it with Create or std::make_shared: Bind uses
  ///        shared_from_this().
  /// @param site_layout layout whose DOF holders are the sites
  /// @param localization search for the MLS supports of query points
  PointCloudGeometry(std::shared_ptr<const FieldLayout> site_layout,
                     std::shared_ptr<LocalizationFactory> localization)
    : site_layout_(std::move(site_layout)),
      localization_(std::move(localization))
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

protected:
  std::shared_ptr<BoundSource> BindImpl(
    const BindRequest& request) const override
  {
    PCMS_FUNCTION_TIMER;
    // MLS distances are based on the unweighted Euclidean measure
    // this is only valid with identity metric
    // this almost certainly should be generalized to take metric for distances
    // for now, we just check and fail so at least the user gets a warning
    if (!HasIdentityMetric(*site_layout_->GetCoordinateSystem())) {
      throw pcms_error(
        "PointCloudGeometry: MLS point-cloud evaluation requires a "
        "coordinate system whose metric is the identity; got '" +
        std::string(site_layout_->GetCoordinateSystem()->Kind()) + "'");
    }

    // Extract source coordinates for the MLS solve.
    const auto src_view = site_layout_->GetDOFHolderCoordinates().GetValues();
    Omega_h::Reals source_coords =
      flatten_to_omega_h_reals(src_view, "src_coords");

    SupportResults supports;
    auto path =
      detail::SelectLocalizationPath(*site_layout_, request.query_layout.get());
    if (path == detail::LocalizationPath::CentroidToVertexAdjacencySearch) {
      auto* adjacency =
        dynamic_cast<const AdjacencyLocalizationFactory*>(localization_.get());
      PCMS_ALWAYS_ASSERT(adjacency != nullptr);
      supports = adjacency->BuildSameMeshCentroidToVertex();
    } else {
      supports = localization_->Build(request.coords);
    }

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
  std::shared_ptr<const FieldLayout> site_layout_;
  std::shared_ptr<LocalizationFactory> localization_;
};

} // namespace pcms

#endif // PCMS_FIELD_EVALUATOR_POINT_CLOUD_GEOMETRY_HPP
