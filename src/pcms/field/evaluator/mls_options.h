#ifndef PCMS_FIELD_EVALUATOR_MLS_OPTIONS_H
#define PCMS_FIELD_EVALUATOR_MLS_OPTIONS_H

#include "pcms/field/evaluator/mls_interpolation.hpp"
#include "pcms/geometry/geometry.hpp"

namespace pcms
{

/// Configuration for Moving Least Squares evaluation.
/// Passed to PolynomialReconstructionFunctionSpace::Create and
/// PointCloudEvaluatorFactory.
struct MLSOptions
{
  double radius = 0.5;
  unsigned min_req_supports = 10;
  unsigned degree = 3;
  bool adapt_radius = true;
  double lambda = 0.0;
  double tol = 1e-6;
  double decay_factor = 5.0;
  RadialBasisFunction basis = RadialBasisFunction::RBF_GAUSSIAN;

  /// @return the options of the support search these options ask for:
  ///         radius, min_req_supports, at most 3 * min_req_supports, and
  ///         adapt_radius
  [[nodiscard]] NeighbourhoodOptions Localization() const
  {
    NeighbourhoodOptions options;
    options.radius = radius;
    options.min_supports = static_cast<int>(min_req_supports);
    options.max_supports = 3 * static_cast<int>(min_req_supports);
    options.adapt_radius = adapt_radius;
    return options;
  }

  /// @return MLS options whose support search is `localization` (the other
  ///         members keep their defaults)
  [[nodiscard]] static MLSOptions FromLocalization(
    const NeighbourhoodOptions& localization)
  {
    MLSOptions options;
    options.radius = localization.radius;
    options.min_req_supports = static_cast<unsigned>(localization.min_supports);
    options.adapt_radius = localization.adapt_radius;
    return options;
  }
};

} // namespace pcms

#endif // PCMS_FIELD_EVALUATOR_MLS_OPTIONS_H
