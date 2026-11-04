#ifndef PCMS_POINT_CLOUD_EVALUATOR_FACTORY_H
#define PCMS_POINT_CLOUD_EVALUATOR_FACTORY_H

#include "pcms/field/point_evaluator_factory.hpp"
#include "pcms/field/point_evaluator.hpp"
#include "pcms/field/field_data.hpp"
#include "pcms/field/evaluator/mls_options.hpp"
#include "pcms/field/evaluator/mls_point_cloud.hpp"
#include "pcms/localization/localization_path_selection.hpp"
#include "pcms/localization/mesh_localization.hpp"
#include "pcms/localization/localization_factory.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/omega_h_array_utils.hpp"

#include <Omega_h_array.hpp>
#include <memory>
#include <string>

namespace pcms
{

/// @brief Point evaluation of point-cloud fields by moving least squares.
class PointCloudEvaluatorFactory
  : public TypedPointEvaluatorFactory<Real, MLSBinding>
{
  using Bound = MLSBinding;

public:
  /// @param layout point-cloud layout of the evaluated fields
  /// @param localization search for the MLS supports of query points
  /// @param options MLS options
  PointCloudEvaluatorFactory(std::shared_ptr<const FieldLayout> layout,
                             std::shared_ptr<LocalizationFactory> localization,
                             MLSOptions options = {})
    : TypedPointEvaluatorFactory<Real, Bound>(layout->GetCoordinateSystem()),
      layout_(std::move(layout)),
      localization_(std::move(localization)),
      options_(options)
  {
  }

protected:
  std::shared_ptr<Bound> BindPoints(const BindRequest& request) const override
  {
    const auto coords = request.coords;
    // MLS distances are based on the unweighted Euclidean measure
    // this is only valid with identity metric
    // this almost certainly should be generalized to take metric for distances
    // for now, we just check and fail so at least the user gets a warning
    if (!HasIdentityMetric(*layout_->GetCoordinateSystem())) {
      throw pcms_error(
        "PointCloudEvaluatorFactory: MLS point-cloud evaluation requires a "
        "coordinate system whose metric is the identity; got '" +
        std::string(layout_->GetCoordinateSystem()->Kind()) + "'");
    }

    // Extract source coordinates for the MLS solve.
    const auto src_view = layout_->GetDOFHolderCoordinates().GetValues();
    Omega_h::Reals source_coords =
      flatten_to_omega_h_reals(src_view, "src_coords");

    SupportResults supports;
    auto path =
      detail::SelectLocalizationPath(*layout_, request.query_layout.get());
    if (path == detail::LocalizationPath::CentroidToVertexAdjacencySearch) {
      auto* adjacency =
        dynamic_cast<const AdjacencyLocalizationFactory*>(localization_.get());
      PCMS_ALWAYS_ASSERT(adjacency != nullptr);
      supports = adjacency->BuildSameMeshCentroidToVertex();
    } else {
      supports = localization_->Build(coords);
    }

    return std::make_shared<Bound>(request, std::move(source_coords),
                                   std::move(supports));
  }

  std::unique_ptr<PointEvaluator<Real>> CreateEvaluator(
    std::shared_ptr<const Bound> bound, int /*derivative_order*/) const override
  {
    return std::make_unique<MLSPointEvaluator<>>(
      std::move(bound), layout_->GetDimension(), options_);
  }

private:
  std::shared_ptr<const FieldLayout> layout_;
  std::shared_ptr<LocalizationFactory> localization_;
  MLSOptions options_;
};

} // namespace pcms

#endif // PCMS_POINT_CLOUD_EVALUATOR_FACTORY_H
