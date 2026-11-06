#ifndef PCMS_FIELD_EVALUATOR_MLS_POINT_CLOUD_H
#define PCMS_FIELD_EVALUATOR_MLS_POINT_CLOUD_H

#include "pcms/field/point_evaluator.hpp"
#include "pcms/field/field_data.hpp"
#include "pcms/localization/mls_options.hpp"
#include "pcms/field/evaluator/mls_interpolation.hpp"
#include "pcms/backends/omega_h/adj_search.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/omega_h_array_utils.hpp"

#include <Omega_h_array.hpp>
#include <Omega_h_array_ops.hpp>

namespace pcms::backend::point_cloud
{

/// @brief MLS bound source: the supports of the query points and the
///        coordinates the solve uses. Every query point is found.
class MLSBinding final : public BoundSource
{
public:
  /// @param request query points and out-of-bounds policy
  /// @param source_coords flattened coordinates of the source points
  /// @param supports MLS supports of the query points
  MLSBinding(const BindRequest& request, Omega_h::Reals source_coords,
             localization::SupportResults supports)
    : BoundSource(request),
      source_coords_(std::move(source_coords)),
      target_coords_(
        utility::flatten_to_omega_h_reals(PointValues(), "mls_target_coords")),
      supports_(std::move(supports))
  {
  }

  /// @return flattened coordinates of the source points
  [[nodiscard]] const Omega_h::Reals& SourceCoordinates() const noexcept
  {
    return source_coords_;
  }

  /// @return flattened coordinates of the query points
  [[nodiscard]] const Omega_h::Reals& TargetCoordinates() const noexcept
  {
    return target_coords_;
  }

  /// @return MLS supports of the query points
  [[nodiscard]] const localization::SupportResults& Supports() const noexcept
  {
    return supports_;
  }

private:
  Omega_h::Reals source_coords_;
  Omega_h::Reals target_coords_;
  localization::SupportResults supports_;
};

/// @brief Evaluates a scalar point-cloud field at the bound points by MLS.
template <typename LayoutPolicy =
            detail::default_layout_for_memory_space_t<DeviceMemorySpace>>
class MLSPointEvaluator : public PointEvaluator<Real, LayoutPolicy>
{
public:
  /// @param binding MLS supports of the query points
  /// @param dim spatial dimension
  /// @param options MLS options
  MLSPointEvaluator(std::shared_ptr<const MLSBinding> binding, int dim,
                    MLSOptions options)
    : PointEvaluator<Real, LayoutPolicy>(binding),
      binding_(std::move(binding)),
      dim_(dim),
      options_(options)
  {
  }

  // Public because CUDA forbids extended lambdas in non-public member
  // functions. User call Evaluate.
  // TODO: add passkey or other mechanism to protect this
  void EvaluateImpl(
    const Field<Real>& field,
    ValueView<Real, DeviceMemorySpace, LayoutPolicy> out) const override
  {
    const auto values = out.GetValues();
    if (values.extent(1) != 1) {
      throw pcms_error(
        "MLSPointEvaluator: only scalar (num_components==1) evaluation is "
        "supported in this phase");
    }

    const auto device_data = field.GetDOFHolderData().GetValues();
    const int n_sources = static_cast<int>(device_data.size());

    Omega_h::Write<Omega_h::Real> src_w(n_sources, "mls_source_values");
    Kokkos::parallel_for(
      "CopyDeviceDataToOmegaHWrite",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, n_sources),
      KOKKOS_LAMBDA(int i) { src_w[i] = device_data(i, 0); });
    Omega_h::Reals source_values(src_w);

    auto result = field::mls_interpolation(
      source_values, binding_->SourceCoordinates(),
      binding_->TargetCoordinates(), binding_->Supports(), dim_,
      static_cast<Omega_h::LO>(options_.degree), options_.basis,
      options_.lambda, options_.tol, options_.decay_factor);

    const int n_targets = result.size();
    Kokkos::parallel_for(
      "CopyMLSResultToValues",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, n_targets),
      KOKKOS_LAMBDA(int i) { values(i, 0) = result[i]; });
  }

private:
  std::shared_ptr<const MLSBinding> binding_;
  int dim_;
  MLSOptions options_;
};

} // namespace pcms::backend::point_cloud

#endif // PCMS_FIELD_EVALUATOR_MLS_POINT_CLOUD_H
