#include "pcms/field/function_spaces.hpp"

#include "pcms/field/detail/simple_field_data.hpp"
#include "pcms/backends/uniform_grid/uniform_grid_geometry.hpp"
#include "pcms/backends/uniform_grid/spline_evaluator.hpp"
#include "pcms/backends/uniform_grid/layout.hpp"
#include "pcms/utility/detail/common.hpp"

namespace pcms
{
using uniform_grid::UniformGridFieldLayout;
using uniform_grid::UniformGridGeometry;
using uniform_grid::UniformGridSplineEvaluatorFactory;
} // namespace pcms

namespace pcms
{

namespace
{

/// The spline space on a uniform-grid geometry: order-1 single-component
/// layout, one cubic (1D) or bicubic (2D) spline of the vertex values.
template <unsigned Dim>
std::shared_ptr<const FunctionSpace> CreateUniformGridSplineSpace(
  std::shared_ptr<const UniformGridGeometry<Dim>> grid_geometry)
{
  auto layout =
    std::make_shared<UniformGridFieldLayout<Dim>>(grid_geometry, 1, 1);
  auto evaluator_factory =
    std::make_shared<UniformGridSplineEvaluatorFactory<Dim>>(
      std::move(grid_geometry), layout);
  return std::make_shared<FunctionSpace>(
    layout,
    [layout](Type t, ValueBasis basis) -> FieldDataVariant {
      return detail::apply_to_type(t, [&](auto tag) -> FieldDataVariant {
        using T = typename decltype(tag)::type;
        if constexpr (!std::is_same_v<T, double>) {
          throw pcms_error("CreateSplineSpace: only double (Real) is "
                           "supported");
        } else {
          return std::make_unique<pcms::detail::SimpleFieldData<double>>(
            layout, std::move(basis));
        }
      });
    },
    evaluator_factory, evaluator_factory->GetGeometry());
}

} // namespace

std::shared_ptr<const FunctionSpace> CreateSplineSpace(
  std::shared_ptr<const Geometry> geometry)
{
  if (geometry == nullptr) {
    throw pcms_error("CreateSplineSpace: the geometry must not be null");
  }
  if (auto grid =
        std::dynamic_pointer_cast<const UniformGridGeometry<1>>(geometry)) {
    return CreateUniformGridSplineSpace<1>(std::move(grid));
  }
  if (auto grid =
        std::dynamic_pointer_cast<const UniformGridGeometry<2>>(geometry)) {
    return CreateUniformGridSplineSpace<2>(std::move(grid));
  }
  throw pcms_error(
    "CreateSplineSpace: only 1D and 2D uniform-grid geometries are supported");
}

} // namespace pcms
