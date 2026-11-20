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
using uniform_grid::UniformGridSplineEvaluatorFactory2D;
} // namespace pcms

namespace pcms
{

std::shared_ptr<const FunctionSpace> CreateSplineSpace(
  std::shared_ptr<const Geometry> geometry, int num_components)
{
  if (geometry == nullptr) {
    throw pcms_error("CreateSplineSpace: the geometry must not be null");
  }
  auto grid_geometry =
    std::dynamic_pointer_cast<const UniformGridGeometry<2>>(geometry);
  if (grid_geometry == nullptr) {
    throw pcms_error(
      "CreateSplineSpace: only 2D uniform-grid geometries are supported");
  }
  auto layout = std::make_shared<UniformGridFieldLayout<2>>(grid_geometry,
                                                            num_components, 1);
  auto evaluator_factory =
    std::make_shared<UniformGridSplineEvaluatorFactory2D>(
      std::move(grid_geometry), layout);
  return std::make_shared<FunctionSpace>(layout, evaluator_factory,
                                         evaluator_factory->GetGeometry());
}

} // namespace pcms
