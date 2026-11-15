#include "pcms/field/transformed_source.hpp"
#include "pcms/field/function_space.h"
#include <string>

namespace pcms
{

namespace
{

class TransformedBoundSource final : public BoundSource
{
public:
  TransformedBoundSource(const BindRequest& request,
                         std::shared_ptr<const BoundCoordinateMap> map,
                         std::shared_ptr<const BoundSource> inner)
    : BoundSource(request), map_(std::move(map)), inner_(std::move(inner))
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

} // namespace

TransformedSource::TransformedSource(
  std::shared_ptr<const PointEvaluatorFactory> source,
  std::shared_ptr<const CoordinateMap> to_source)
  : source_(std::move(source)), to_source_(std::move(to_source))
{
  if (source_ == nullptr) {
    throw pcms_error("TransformedSource: source must not be null");
  }
  if (to_source_ == nullptr) {
    throw pcms_error("TransformedSource: to_source must not be null");
  }
  if (!SameCoordinateSystem(to_source_->GetTargetCoordinateSystem(),
                            source_->GetCoordinateSystem())) {
    throw pcms_error(
      "TransformedSource: to_source maps into '" +
      std::string(to_source_->GetTargetCoordinateSystem()->Kind()) +
      "' but the source's coordinate system is '" +
      std::string(source_->GetCoordinateSystem()->Kind()) +
      "'; expected a map ending in '" +
      std::string(source_->GetCoordinateSystem()->Kind()) + "'");
  }
  system_ = to_source_->GetSourceCoordinateSystem();
}

std::shared_ptr<BoundSource> TransformedSource::BindImpl(
  const BindRequest& request) const
{
  PCMS_FUNCTION_TIMER;
  std::shared_ptr<const BoundCoordinateMap> map =
    to_source_->Bind(request.coords);
  auto inner = source_->Bind(map->MappedPoints(), request.policy);
  return std::make_shared<TransformedBoundSource>(request, std::move(map),
                                                  std::move(inner));
}

template <typename T>
std::unique_ptr<PointEvaluator<T>> TransformedSource::Create(
  std::shared_ptr<const BoundSource> bound, DifferentialOperator op) const
{
  // CreatePointEvaluator accepts only bound sources this source created
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
