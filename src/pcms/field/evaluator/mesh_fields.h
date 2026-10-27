#ifndef PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_EVALUATOR_FACTORY_H
#define PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_EVALUATOR_FACTORY_H

#include "pcms/field/evaluator/mesh_fields_backend.h"
#include "pcms/field/data/mesh_fields.h"
#include "pcms/field/field_evaluator_factory.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/field/field_data.h"
#include "pcms/utility/profile.h"
#include "pcms/utility/arrays.h"
#include <memory>

namespace pcms
{

// MeshFieldsPointEvaluator<T> implements PointEvaluator<T> for
// MeshFields-backed simplex meshes. Each Evaluate call loads DOF data from the
// FieldData argument into the MeshFieldBackend's shape_field_ via SetData, then
// calls evaluate().
template <typename T,
          typename LayoutPolicy =
            detail::default_layout_for_memory_space_t<DeviceMemorySpace>>
class MeshFieldsPointEvaluator : public PointEvaluator<T, LayoutPolicy>
{
public:
  MeshFieldsPointEvaluator(
    std::shared_ptr<const MeshFieldsAdapterLayout> layout,
    MeshFieldsAdapter2LocalizationHint hint, Real fill_value,
    int derivative_order = 0)
    : layout_(std::move(layout)),
      hint_(std::move(hint)),
      fill_value_(fill_value),
      derivative_order_(derivative_order)
  {
  }

  [[nodiscard]] ValueBasis OutputBasis(const ValueBasis& stored) const override
  {
    if (derivative_order_ == 0) {
      return stored;
    }
    return FirstDerivativeBasis(stored, layout_->GetCoordinateSystem());
  }

  void Evaluate(
    const Field<T>& field,
    ValueView<T, DeviceMemorySpace, LayoutPolicy> out) const override
  {
    PCMS_FUNCTION_TIMER;
    this->CheckEvaluateWriteTag("MeshFieldsPointEvaluator::Evaluate", field,
                                out);
    const auto values = out.GetValues();
    PCMS_ALWAYS_ASSERT(values.extent(0) ==
                       hint_.coordinates_.extent(0) + hint_.num_missing_);
    // Components, or with derivative order 1 the derivative along each mesh
    // coordinate of each component.
    const int ncomp = layout_->GetNumComponents() *
                      (derivative_order_ == 0 ? 1 : layout_->GetMesh().dim());
    PCMS_ALWAYS_ASSERT(values.extent(1) == static_cast<size_t>(ncomp));
    auto const* mesh_field_data =
      dynamic_cast<const MeshFieldsFieldData<T>*>(&field.GetData());
    if (!mesh_field_data) {
      throw pcms_error(
        "MeshFieldsPointEvaluator::Evaluate: incompatible FieldData type");
    }

    const auto& backend = *mesh_field_data->GetMeshFieldBackend();
    auto eval_results =
      derivative_order_ == 0
        ? backend.evaluate(hint_.coordinates_d_, hint_.offsets_d_)
        : backend.evaluateGradient(hint_.coordinates_d_, hint_.offsets_d_);

    int num_eval = static_cast<int>(eval_results.extent(0));
    Kokkos::parallel_for(
      "CopyEvalResultsToValues",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_eval, ncomp}),
      KOKKOS_CLASS_LAMBDA(LO i, int c) {
        values(hint_.indices_d_(i), c) = eval_results(i, c);
      });

    if (hint_.num_missing_ > 0 && hint_.mode_ == OutOfBoundsMode::FILL) {
      T fill_val = static_cast<T>(fill_value_);
      int num_missing = static_cast<int>(hint_.num_missing_);
      Kokkos::parallel_for(
        "FillMissingValues",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_missing, ncomp}),
        KOKKOS_CLASS_LAMBDA(LO i, int c) {
          values(hint_.missing_indices_d_(i), c) = fill_val;
        });
    }
  }

  [[nodiscard]] Kokkos::View<const LO*, DeviceMemorySpace> FilledPoints()
    const override
  {
    if (hint_.mode_ != OutOfBoundsMode::FILL) {
      return {};
    }
    return hint_.missing_indices_d_;
  }

private:
  std::shared_ptr<const MeshFieldsAdapterLayout> layout_;
  MeshFieldsAdapter2LocalizationHint hint_;
  Real fill_value_;
  int derivative_order_;
};

// MeshFieldsEvaluatorFactory<T> implements FieldEvaluatorFactory<T> for
// MeshFields-backed simplex meshes. It owns the spatial search structure and
// the MeshFieldBackend (which holds the internal shape field).
template <typename T>
class MeshFieldsEvaluatorFactory : public FieldEvaluatorFactory<T>
{
public:
  explicit MeshFieldsEvaluatorFactory(
    std::shared_ptr<const MeshFieldsAdapterLayout> layout)
    : layout_(std::move(layout)),
      mesh_(layout_->GetMesh()),
      search_(mesh_, 10, 10)
  {
    if (mesh_.dim() == 3) {
      throw pcms_error("MeshFieldsEvaluatorFactory does not support 3D meshes");
    }
  }

  const FieldLayout& GetLayout() const override { return *layout_; }

  bool HasDOFHolderCoordinates() const override { return true; }

  bool SupportsNearestBoundary() const override { return false; }

  std::shared_ptr<BoundSource> Bind(
    const EvaluationRequest& request) const override
  {
    PCMS_FUNCTION_TIMER;
    const auto coords = request.coords;
    const auto policy = request.policy;
    if (policy.mode == OutOfBoundsMode::NEAREST_BOUNDARY) {
      throw pcms_error(
        "MeshFieldsEvaluatorFactory: NearestBoundary is not supported");
    }

    auto coordinates = coords.GetValues();
    Kokkos::View<Real* [2], DeviceMemorySpace> coords_search(
      "coords_search", coordinates.extent(0));
    Kokkos::parallel_for(
      "copy_coords",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(
        0, coordinates.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        coords_search(i, 0) = coordinates(i, 0);
        coords_search(i, 1) = coordinates(i, 1);
      });
    auto results = search_(coords_search);
    auto owning_ids = search_.GetOwningElementIds(results);

    MeshFieldsAdapter2LocalizationHint hint(mesh_, results, coords_search,
                                            owning_ids, policy.mode);

    return std::make_shared<
      detail::HintBoundSource<MeshFieldsAdapter2LocalizationHint>>(
      request, std::move(hint));
  }

  std::unique_ptr<PointEvaluator<T>> CreatePointEvaluator(
    std::shared_ptr<const BoundSource> bound,
    int derivative_order) const override
  {
    if (derivative_order > 1) {
      throw pcms_error("MeshFieldsEvaluatorFactory: derivative order " +
                       std::to_string(derivative_order) + " is not supported");
    }
    if (derivative_order == 1 && layout_->GetNumComponents() != 1) {
      throw pcms_error("MeshFieldsEvaluatorFactory: derivatives of "
                       "multi-component fields are not supported yet");
    }
    const auto& hint =
      detail::BoundSourceAs<
        detail::HintBoundSource<MeshFieldsAdapter2LocalizationHint>>(
        *bound, "MeshFieldsEvaluatorFactory")
        .GetHint();
    return std::make_unique<MeshFieldsPointEvaluator<T>>(
      layout_, hint, bound->Policy().fill_value, derivative_order);
  }

  CoordinateView<DeviceMemorySpace> GetDOFHolderCoordinates() const override
  {
    return layout_->GetDOFHolderCoordinates();
  }

private:
  std::shared_ptr<const MeshFieldsAdapterLayout> layout_;
  Omega_h::Mesh& mesh_;
  mutable GridPointSearch2D search_;
};

} // namespace pcms

#endif // PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_EVALUATOR_FACTORY_H
