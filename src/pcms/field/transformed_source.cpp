#include "pcms/field/transformed_source.hpp"
#include "pcms/field/function_space.h"
#include "pcms/utility/profile.h"
#include <string>

namespace pcms
{

namespace
{

class TransformedBoundSource final : public BoundSource
{
public:
  TransformedBoundSource(std::shared_ptr<const Geometry> geometry,
                         const BindRequest& request,
                         std::shared_ptr<const BoundCoordinateMap> map,
                         std::shared_ptr<const BoundSource> inner)
    : BoundSource(std::move(geometry), request),
      map_(std::move(map)),
      inner_(std::move(inner))
  {
    SetFound(*inner_);
  }

  [[nodiscard]] const std::shared_ptr<const BoundCoordinateMap>& Map()
    const noexcept
  {
    return map_;
  }
  [[nodiscard]] const std::shared_ptr<const BoundSource>& Inner() const noexcept
  {
    return inner_;
  }

private:
  std::shared_ptr<const BoundCoordinateMap> map_;
  std::shared_ptr<const BoundSource> inner_;
};

/// The source's geometry seen through a coordinate map: binding maps the
/// query points and binds the source's geometry to the mapped points.
class TransformedGeometry final : public Geometry
{
public:
  TransformedGeometry(std::shared_ptr<const Geometry> inner,
                      std::shared_ptr<const CoordinateMap> to_inner)
    : inner_(std::move(inner)),
      to_inner_(std::move(to_inner)),
      system_(to_inner_->GetSourceCoordinateSystem())
  {
  }

  [[nodiscard]] std::shared_ptr<const Discretization> GetDiscretization()
    const noexcept override
  {
    return inner_->GetDiscretization();
  }

  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>&
  GetCoordinateSystem() const noexcept override
  {
    return system_;
  }

  [[nodiscard]] bool SupportsNearestBoundary() const noexcept override
  {
    return inner_->SupportsNearestBoundary();
  }

protected:
  std::shared_ptr<BoundSource> BindImpl(
    const BindRequest& request) const override
  {
    PCMS_FUNCTION_TIMER;
    std::shared_ptr<const BoundCoordinateMap> map =
      to_inner_->Bind(request.coords);
    auto inner = inner_->Bind(map->MappedPoints(), request.policy);
    return std::make_shared<TransformedBoundSource>(
      shared_from_this(), request, std::move(map), std::move(inner));
  }

  std::shared_ptr<BoundSource> BindParametricImpl(
    const ParametricPoints& /*points*/,
    OutOfBoundsPolicy /*policy*/) const override
  {
    throw pcms_error("TransformedSource: parametric points are not mapped; "
                     "bind them on the source's geometry");
  }

private:
  std::shared_ptr<const Geometry> inner_;
  std::shared_ptr<const CoordinateMap> to_inner_;
  std::shared_ptr<const CoordinateSystem> system_;
};

} // namespace

TransformedSource::TransformedSource(
  std::shared_ptr<const PointEvaluatorFactory> source,
  std::shared_ptr<const CoordinateMap> to_source)
  : source_(std::move(source))
{
  if (source_ == nullptr) {
    throw pcms_error("TransformedSource: source must not be null");
  }
  if (to_source == nullptr) {
    throw pcms_error("TransformedSource: to_source must not be null");
  }
  auto inner = source_->GetGeometry();
  if (inner == nullptr) {
    throw pcms_error("TransformedSource: the source has no geometry");
  }
  if (!SameCoordinateSystem(to_source->GetTargetCoordinateSystem(),
                            inner->GetCoordinateSystem())) {
    throw pcms_error(
      "TransformedSource: to_source maps into '" +
      std::string(to_source->GetTargetCoordinateSystem()->Kind()) +
      "' but the source's coordinate system is '" +
      std::string(inner->GetCoordinateSystem()->Kind()) +
      "'; expected a map ending in '" +
      std::string(inner->GetCoordinateSystem()->Kind()) + "'");
  }
  geometry_ = std::make_shared<TransformedGeometry>(std::move(inner),
                                                    std::move(to_source));
}

template <typename T>
std::unique_ptr<PointEvaluator<T>> TransformedSource::Create(
  std::shared_ptr<const BoundSource> bound, DifferentialOperator op) const
{
  // CreatePointEvaluator accepts only bound sources of geometry_, which binds
  // TransformedBoundSources
  const auto& transformed = static_cast<const TransformedBoundSource&>(*bound);
  auto inner = source_->CreatePointEvaluator<T>(transformed.Inner(), op);
  return std::make_unique<TransformedPointEvaluator<T>>(
    std::move(bound), transformed.Map(), std::move(inner));
}

PointEvaluatorVariant TransformedSource::CreatePointEvaluatorImpl(
  Type value_type, std::shared_ptr<const BoundSource> bound,
  DifferentialOperator op) const
{
  return apply_to_type(value_type, [&](auto type) -> PointEvaluatorVariant {
    return Create<typename decltype(type)::type>(bound, op);
  });
}

} // namespace pcms
