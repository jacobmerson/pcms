#include "pcms/field/function_space/spline.h"

#include "pcms/field/data/simple.h"
#include "pcms/field/evaluator/uniform_grid_geometry.hpp"
#include "pcms/field/evaluator/uniform_grid_spline.h"
#include "pcms/field/layout/uniform_grid.h"
#include "pcms/utility/common.h"

#include <stdexcept>

namespace pcms
{

SplineFunctionSpace::SplineFunctionSpace(
  Key, std::shared_ptr<const FieldLayout> layout,
  std::shared_ptr<const PointEvaluatorFactory> evaluator) noexcept
  : FunctionSpace(std::move(evaluator)), layout_(std::move(layout))
{
}

std::shared_ptr<SplineFunctionSpace> SplineFunctionSpace::FromUniformGrid(
  const UniformGrid<2>& grid,
  std::shared_ptr<const CoordinateSystem> mesh_coordinate_system)
{
  return FromGeometry(
    UniformGridGeometry<2>::Create(grid, std::move(mesh_coordinate_system)));
}

std::shared_ptr<SplineFunctionSpace> SplineFunctionSpace::FromGeometry(
  std::shared_ptr<const UniformGridGeometry<2>> geometry)
{
  if (geometry == nullptr) {
    throw pcms_error("SplineFunctionSpace::FromGeometry: geometry must not "
                     "be null");
  }
  auto layout = std::make_shared<UniformGridFieldLayout<2>>(
    geometry->GetGrid(), 1, geometry->GetCoordinateSystem(), 1);
  auto evaluator_factory =
    std::make_shared<UniformGridSplineEvaluatorFactory2D>(std::move(geometry),
                                                          layout);
  return std::make_shared<SplineFunctionSpace>(Key{}, layout,
                                               std::move(evaluator_factory));
}

std::shared_ptr<const FieldLayout> SplineFunctionSpace::GetLayout()
  const noexcept
{
  return layout_;
}

FieldVariant SplineFunctionSpace::CreateFieldImpl(Type storage_type,
                                                  ValueBasis basis) const
{
  return apply_to_type(storage_type, [&](auto tag) -> FieldVariant {
    using T = typename decltype(tag)::type;
    if constexpr (!std::is_same_v<T, double>) {
      throw pcms_error("SplineFunctionSpace: only double (Real) is supported");
    } else {
      return WrapField<double>(
        layout_,
        std::make_unique<SimpleFieldData<double>>(layout_, std::move(basis)));
    }
  });
}

FieldVariant SplineFunctionSpace::CreateFieldImpl(FieldDataVariant data) const
{
  if (!std::holds_alternative<std::unique_ptr<FieldData<double>>>(data)) {
    throw pcms_error("SplineFunctionSpace: only double (Real) is supported");
  }
  auto fd = std::move(std::get<std::unique_ptr<FieldData<double>>>(data));
  PCMS_ALWAYS_ASSERT(fd != nullptr);
  if (dynamic_cast<const SimpleFieldData<double>*>(fd.get()) == nullptr) {
    throw pcms_error(
      "SplineFunctionSpace::CreateField: requires SimpleFieldData<double>");
  }
  if (fd->GetDOFAccess()->GetValues().size() !=
      detail::ExpectedFlatFieldDataSize(*layout_)) {
    throw pcms_error(
      "SplineFunctionSpace::CreateField: field data size does not match "
      "layout");
  }
  return WrapField<double>(layout_, std::move(fd));
}

} // namespace pcms
