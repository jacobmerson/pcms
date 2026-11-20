#include "pcms/field/embedding.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/detail/memory_spaces.hpp"
#include <cmath>
#include <string>

namespace pcms
{

namespace
{

template <typename T>
using DeviceView = Kokkos::View<T, DeviceMemorySpace>;

// jacobian(p, i, j) = d x_j / d xi_i from the gradient of the position, whose
// component j has the partials d x_j / d xi_i at columns j * dp + i.
void ReshapeJacobian(DeviceView<const Real**> gradient,
                     DeviceView<Real***> jacobian)
{
  const int dp = jacobian.extent(1);
  const int dt = jacobian.extent(2);
  detail::ForEach(
    "position_jacobian", static_cast<LO>(jacobian.extent(0)),
    KOKKOS_LAMBDA(const LO p) {
      for (int i = 0; i < dp; ++i) {
        for (int j = 0; j < dt; ++j) {
          jacobian(p, i, j) = gradient(p, j * dp + i);
        }
      }
    });
}

struct Evaluation
{
  DeviceView<Real**> mapped;    // [point][chart dimension]
  DeviceView<Real***> jacobian; // [point][parameter][chart]
};

// A position Function: its value and Jacobian at parameter points.
class Position
{
public:
  explicit Position(std::shared_ptr<const Function<Real>> position)
    : position_(std::move(position))
  {
    if (position_ == nullptr) {
      throw pcms_error("CreateCoordinateMap: the position must not be null");
    }
    if (!position_->GetData().GetValueBasis().coordinates) {
      throw pcms_error("CreateCoordinateMap: the position's values must be "
                       "coordinates (ValueBasis::Coordinates)");
    }
    if (!position_->GetSpace().Supports(DifferentialOperator::Gradient)) {
      throw pcms_error("CreateCoordinateMap: the position's space does not "
                       "evaluate gradients");
    }
    if (ParameterChart()->Dimension() > 3 || Chart()->Dimension() > 3) {
      throw pcms_error("CreateCoordinateMap: dimensions above 3 are not "
                       "supported");
    }
  }

  [[nodiscard]] const std::shared_ptr<const Function<Real>>& Get() const
  {
    return position_;
  }
  /// the chart the position's values are points of
  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>& Chart() const
  {
    return position_->GetData().GetValueBasis().system;
  }
  /// the chart of the parameter geometry
  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>& ParameterChart()
    const
  {
    return position_->GetSpace().GetCoordinateSystem();
  }

  /// The position and its Jacobian at parameter points; points outside the
  /// parameter geometry use its nearest cell.
  [[nodiscard]] Evaluation Evaluate(
    Rank2View<const Real, DeviceMemorySpace> xi) const
  {
    const FunctionSpace& space = position_->GetSpace();
    const auto geometry = space.GetGeometry();
    const auto n = xi.extent(0);
    const int dp = ParameterChart()->Dimension();
    const int dt = Chart()->Dimension();
    const auto bound = geometry->Bind(
      CoordinateView<DeviceMemorySpace>(geometry->GetCoordinateSystem(), xi),
      OutOfBoundsPolicy{OutOfBoundsMode::NEAREST_BOUNDARY});
    const ValueBasis basis = position_->GetData().GetValueBasis();
    Evaluation e{DeviceView<Real**>("position_mapped", n, dt),
                 DeviceView<Real***>("position_jacobian", n, dp, dt)};
    space.CreatePointEvaluator<Real>(bound)->Evaluate(
      *position_,
      ValueView<Real, DeviceMemorySpace>(basis, MakeRank2View(e.mapped)));
    DeviceView<Real**> gradient("position_gradient", n,
                                static_cast<std::size_t>(dt) * dp);
    const auto evaluator =
      space.CreatePointEvaluator<Real>(bound, DifferentialOperator::Gradient);
    evaluator->Evaluate(
      *position_, ValueView<Real, DeviceMemorySpace>(
                    evaluator->OutputBasis(basis), MakeRank2View(gradient)));
    ReshapeJacobian(gradient, e.jacobian);
    return e;
  }

private:
  std::shared_ptr<const Function<Real>> position_;
};

// The coordinate map of a position: parameter chart -> the position's chart.
class PositionMap final : public CoordinateMap
{
public:
  explicit PositionMap(Position position)
    : CoordinateMap(position.ParameterChart(), position.Chart()),
      position_(std::move(position))
  {
  }

protected:
  [[nodiscard]] std::unique_ptr<BoundCoordinateMap> BindImpl(
    const CoordinateView<DeviceMemorySpace>& query_points) const override
  {
    auto e = position_.Evaluate(query_points.GetValues());
    return std::make_unique<detail::BoundJacobianMap>(
      GetTargetCoordinateSystem(), std::move(e.mapped), std::move(e.jacobian));
  }

private:
  Position position_;
};

} // namespace

std::shared_ptr<const CoordinateMap> CreateCoordinateMap(
  std::shared_ptr<const Function<Real>> position)
{
  return std::make_shared<const PositionMap>(Position(std::move(position)));
}

} // namespace pcms
