#ifndef PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_EVALUATOR_FACTORY_H
#define PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_EVALUATOR_FACTORY_H

#include "pcms/field/evaluator/mesh_fields_backend.h"
#include "pcms/field/data/mesh_fields.h"
#include "pcms/field/point_evaluator_factory.hpp"
#include "pcms/field/point_evaluator.h"
#include "pcms/field/field_data.h"
#include "pcms/localization/point_search.h"
#include "pcms/utility/profile.h"
#include "pcms/utility/arrays.h"
#include <memory>

namespace pcms
{

/// @brief MeshFields bound source: the found query points grouped by owning
///        element, with their coordinates in that element.
class MeshFieldsBinding final : public BoundSource
{
public:
  /// @param request query points and out-of-bounds policy
  /// @param search containing-element search over `mesh`
  /// @param mesh 2D simplex mesh of the evaluated fields
  MeshFieldsBinding(const BindRequest& request, const GridPointSearch2D& search,
                    Omega_h::Mesh& mesh)
    : BoundSource(request)
  {
    const auto points = PointValues();
    auto [found, owning_ids] = FindOwningElements(search, points);
    SetFound(found);
    const auto found_points = FoundPoints();
    const auto num_found = static_cast<LO>(found_points.extent(0));
    const LO nelems = mesh.nelems();
    using Range = Kokkos::RangePolicy<DeviceMemorySpace::execution_space>;

    Kokkos::View<LO*> elem_counts("elem_counts", nelems);
    Kokkos::parallel_for(
      "CountPerElement", Range(0, num_found),
      CountPerElementFunctor(found_points, owning_ids, elem_counts));
    offsets_ = Kokkos::View<LO*>("offsets", nelems + 1);
    LO total = 0;
    Kokkos::parallel_scan("ComputeOffsets", Range(0, nelems),
                          ComputeOffsetsDeviceFunctor(offsets_, elem_counts),
                          total);
    Kokkos::deep_copy(Kokkos::subview(offsets_, nelems), total);

    local_coords_ = Kokkos::View<Real**>("local_coords", num_found, mesh.dim());
    indices_ = Kokkos::View<LO*>("indices", num_found);
    Kokkos::parallel_for("FillLocalCoordinates", Range(0, num_found),
                         FillCoordinatesDeviceFunctor(
                           found_points, owning_ids, elem_counts, offsets_,
                           local_coords_, indices_, mesh.ask_elem_verts(),
                           mesh.coords(), points, mesh.dim()));
  }

  /// @return offsets into LocalCoordinates() of each element's points
  ///         (number of elements + 1 entries)
  [[nodiscard]] Kokkos::View<LO*> Offsets() const noexcept { return offsets_; }

  /// @return coordinates of the found points in their owning elements, grouped
  ///         by element
  [[nodiscard]] Kokkos::View<Real**> LocalCoordinates() const noexcept
  {
    return local_coords_;
  }

  /// @return query point of each row of LocalCoordinates()
  [[nodiscard]] Kokkos::View<const LO*> Indices() const noexcept
  {
    return indices_;
  }

private:
  Kokkos::View<LO*> offsets_;
  Kokkos::View<Real**> local_coords_;
  Kokkos::View<LO*> indices_;
};

/// @brief Evaluates MeshFields fields at the bound points with the field's
///        MeshFields backend.
template <typename T,
          typename LayoutPolicy =
            detail::default_layout_for_memory_space_t<DeviceMemorySpace>>
class MeshFieldsPointEvaluator : public PointEvaluator<T, LayoutPolicy>
{
public:
  /// @param layout MeshFields layout of the evaluated fields
  /// @param binding found query points grouped by element
  MeshFieldsPointEvaluator(
    std::shared_ptr<const MeshFieldsAdapterLayout> layout,
    std::shared_ptr<const MeshFieldsBinding> binding)
    : PointEvaluator<T, LayoutPolicy>(binding),
      layout_(std::move(layout)),
      binding_(std::move(binding))
  {
  }

  // Public because CUDA forbids extended lambdas in non-public member
  // functions. User should call Evaluate
  // TODO: add passkey or other mechanism to avoid user calling this
  void EvaluateImpl(
    const Field<T>& field,
    ValueView<T, DeviceMemorySpace, LayoutPolicy> out) const override
  {
    PCMS_FUNCTION_TIMER;
    const auto values = out.GetValues();
    auto const* mesh_field_data =
      dynamic_cast<const MeshFieldsFieldData<T>*>(&field.GetData());
    if (!mesh_field_data) {
      throw pcms_error(
        "MeshFieldsPointEvaluator::Evaluate: incompatible FieldData type");
    }

    auto eval_results = mesh_field_data->GetMeshFieldBackend()->evaluate(
      binding_->LocalCoordinates(), binding_->Offsets());

    int ncomp = layout_->GetNumComponents();
    int num_eval = static_cast<int>(eval_results.extent(0));
    const auto indices = binding_->Indices();
    Kokkos::parallel_for(
      "CopyEvalResultsToValues",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num_eval, ncomp}),
      KOKKOS_LAMBDA(LO i, int c) {
        values(indices(i), c) = eval_results(i, c);
      });
  }

private:
  std::shared_ptr<const MeshFieldsAdapterLayout> layout_;
  std::shared_ptr<const MeshFieldsBinding> binding_;
};

/// @brief Point evaluation of MeshFields fields on a 2D simplex mesh.
/// @tparam T field value type
template <typename T>
class MeshFieldsEvaluatorFactory
  : public TypedPointEvaluatorFactory<T, MeshFieldsBinding>
{
  using Bound = MeshFieldsBinding;

public:
  /// @param layout MeshFields layout of the evaluated fields
  explicit MeshFieldsEvaluatorFactory(
    std::shared_ptr<const MeshFieldsAdapterLayout> layout)
    : TypedPointEvaluatorFactory<T, Bound>(layout->GetCoordinateSystem()),
      layout_(std::move(layout)),
      mesh_(layout_->GetMesh()),
      search_(mesh_, 10, 10)
  {
    if (mesh_.dim() == 3) {
      throw pcms_error("MeshFieldsEvaluatorFactory does not support 3D meshes");
    }
  }

protected:
  std::shared_ptr<Bound> BindPoints(const BindRequest& request) const override
  {
    PCMS_FUNCTION_TIMER;
    return std::make_shared<Bound>(request, search_, mesh_);
  }

  std::unique_ptr<PointEvaluator<T>> CreateEvaluator(
    std::shared_ptr<const Bound> bound, int /*derivative_order*/) const override
  {
    return std::make_unique<MeshFieldsPointEvaluator<T>>(layout_,
                                                         std::move(bound));
  }

private:
  std::shared_ptr<const MeshFieldsAdapterLayout> layout_;
  Omega_h::Mesh& mesh_;
  mutable GridPointSearch2D search_;
};

} // namespace pcms

#endif // PCMS_ADAPTER_MESHFIELDS_MESH_FIELDS_EVALUATOR_FACTORY_H
