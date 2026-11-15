#include "pcms/field/function_space/polynomial_reconstruction.hpp"
#include "pcms/field/layout/omega_h_entity.h"
#include "pcms/field/layout/point_cloud.h"
#include "pcms/field/evaluator/point_cloud.h"
#include "pcms/field/evaluator/point_cloud_geometry.hpp"
#include "pcms/field/data/simple.h"
#include "pcms/discretization/discretization/omega_h.hpp"
#include "pcms/utility/assert.h"
#include "pcms/utility/common.h"
#include "pcms/utility/mesh_geometry.h"

#include <Kokkos_Core.hpp>
#include <Omega_h_array.hpp>
#include <string>

namespace pcms
{

PolynomialReconstructionFunctionSpace::PolynomialReconstructionFunctionSpace(
  Key, std::shared_ptr<const FieldLayout> layout,
  std::shared_ptr<const PointEvaluatorFactory> evaluator) noexcept
  : FunctionSpace(std::move(evaluator)), layout_(std::move(layout))
{
}

std::shared_ptr<PolynomialReconstructionFunctionSpace>
PolynomialReconstructionFunctionSpace::Create(
  CoordinateView<HostMemorySpace> coords, MLSOptions options)
{
  const auto values = coords.GetValues();
  const int dim = static_cast<int>(values.extent(1));
  Kokkos::View<const Real**, Kokkos::HostSpace> host_view(
    values.data_handle(), values.extent(0), values.extent(1));
  auto device_view = Kokkos::View<Real**>("device_view", host_view.extent(0),
                                          host_view.extent(1));
  DeepCopyMismatchLayouts(device_view, host_view);
  auto pc_layout = std::make_shared<PointCloudLayout>(
    dim, device_view, coords.GetCoordinateSystem());
  return FromGeometry(PointCloudGeometry::Create(pc_layout, options), options);
}

std::shared_ptr<PolynomialReconstructionFunctionSpace>
PolynomialReconstructionFunctionSpace::FromMesh(
  Omega_h::Mesh& mesh, int source_entity_dim,
  std::shared_ptr<const CoordinateSystem> coordinate_system, MLSOptions options)
{
  if (source_entity_dim < 0 || source_entity_dim > mesh.dim()) {
    throw pcms_error(
      "PolynomialReconstructionFunctionSpace::FromMesh: source_entity_dim is "
      "out of range");
  }

  auto mesh_layout = std::make_shared<OmegaHEntityLayout>(
    mesh, source_entity_dim, 1, std::move(coordinate_system));
  return FromGeometry(
    PointCloudGeometry::Create(mesh_layout, mesh, source_entity_dim, options),
    options);
}

std::shared_ptr<PolynomialReconstructionFunctionSpace>
PolynomialReconstructionFunctionSpace::FromGeometry(
  std::shared_ptr<const PointCloudGeometry> geometry, MLSOptions options)
{
  if (geometry == nullptr) {
    throw pcms_error("PolynomialReconstructionFunctionSpace::FromGeometry: "
                     "geometry must not be null");
  }
  auto layout = geometry->GetSiteLayout();
  auto eval_factory = std::make_shared<PointCloudEvaluatorFactory>(
    std::move(geometry), layout, options);
  return std::make_shared<PolynomialReconstructionFunctionSpace>(
    Key{}, std::move(layout), std::move(eval_factory));
}

std::shared_ptr<const FieldLayout>
PolynomialReconstructionFunctionSpace::GetLayout() const noexcept
{
  return layout_;
}

FieldVariant PolynomialReconstructionFunctionSpace::CreateFieldImpl(
  Type storage_type, ValueBasis basis) const
{
  return apply_to_type(storage_type, [&](auto tag) -> FieldVariant {
    using T = typename decltype(tag)::type;
    if constexpr (!std::is_same_v<T, double>) {
      throw pcms_error(
        "PolynomialReconstructionFunctionSpace: only double (Real) is "
        "supported");
    } else {
      return WrapField<double>(
        layout_,
        std::make_unique<SimpleFieldData<double>>(layout_, std::move(basis)));
    }
  });
}

FieldVariant PolynomialReconstructionFunctionSpace::CreateFieldImpl(
  FieldDataVariant data) const
{
  if (!std::holds_alternative<std::unique_ptr<FieldData<double>>>(data)) {
    throw pcms_error(
      "PolynomialReconstructionFunctionSpace: only double (Real) is "
      "supported");
  }
  auto fd = std::move(std::get<std::unique_ptr<FieldData<double>>>(data));
  PCMS_ALWAYS_ASSERT(fd != nullptr);
  if (dynamic_cast<const SimpleFieldData<double>*>(fd.get()) == nullptr) {
    throw pcms_error(
      "PolynomialReconstructionFunctionSpace::CreateField: requires "
      "SimpleFieldData<double>");
  }
  if (fd->GetDOFAccess()->GetValues().size() !=
      detail::ExpectedFlatFieldDataSize(*layout_)) {
    throw pcms_error(
      "PolynomialReconstructionFunctionSpace::CreateField: field data size "
      "does not match layout");
  }
  return WrapField<double>(layout_, std::move(fd));
}

} // namespace pcms
