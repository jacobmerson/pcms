#include "pcms/field/transformed_source.hpp"
#include "pcms/field/function_space.h"
#include <string>

namespace pcms
{

namespace detail
{

void FillRows(Rank2View<Real, DeviceMemorySpace> values,
              const Kokkos::View<const LO*, DeviceMemorySpace>& rows,
              Real fill_value)
{
  const auto num_cols = static_cast<int>(values.extent(1));
  Kokkos::parallel_for(
    "transformed_source_fill_rows",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(
      0, static_cast<LO>(rows.extent(0))),
    KOKKOS_LAMBDA(const LO k) {
      for (int c = 0; c < num_cols; ++c) {
        values(rows(k), c) = fill_value;
      }
    });
}

} // namespace detail

namespace
{

class TransformedBoundSource final : public BoundSource
{
public:
  TransformedBoundSource(const EvaluationRequest& request,
                         std::shared_ptr<const BoundCoordinateMap> map,
                         std::shared_ptr<const BoundSource> inner)
    : BoundSource(request), map_(std::move(map)), inner_(std::move(inner))
  {
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
  const EvaluationRequest& request) const
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
  const BoundSource& bound, int derivative_order) const
{
  const auto& transformed =
    detail::BoundSourceAs<TransformedBoundSource>(bound, "TransformedSource");
  auto inner =
    source_->CreatePointEvaluator<T>(transformed.Inner(), derivative_order);
  return std::make_unique<TransformedPointEvaluator<T>>(
    transformed.Map(), std::move(inner), bound.Policy());
}

PointEvaluatorVariant TransformedSource::CreatePointEvaluatorImpl(
  Type value_type, std::shared_ptr<const BoundSource> bound,
  int derivative_order) const
{
  switch (value_type) {
    case Type::Real: return Create<Real>(*bound, derivative_order);
    case Type::LO: return Create<LO>(*bound, derivative_order);
    case Type::GO: return Create<GO>(*bound, derivative_order);
    case Type::Int8: return Create<int8_t>(*bound, derivative_order);
    case Type::Float: return Create<float>(*bound, derivative_order);
  }
  throw pcms_error("TransformedSource::CreatePointEvaluator: unsupported "
                   "value type");
}

} // namespace pcms
