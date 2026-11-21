#include "pcms/utility/coordinate_system.hpp"
#include "pcms/utility/coordinate_systems/cartesian.hpp"
#include "pcms/utility/coordinate_systems/cylindrical.hpp"
#include "pcms/utility/coordinate_systems/euclidean.hpp"
#include "pcms/utility/assert.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace pcms
{

namespace
{

class Cartesian final : public CoordinateSystem
{
public:
  explicit Cartesian(int dim) noexcept : dim_(dim) {}

  [[nodiscard]] std::string_view Kind() const noexcept override
  {
    return dim_ == 0 ? "cartesian (dimension deferred)" : "cartesian";
  }
  [[nodiscard]] int Dimension() const noexcept override { return dim_; }
  [[nodiscard]] bool HasOrthogonalBasis() const noexcept override
  {
    return true;
  }
  [[nodiscard]] bool HasUnitScaleFactors() const noexcept override
  {
    return true;
  }
  [[nodiscard]] bool operator==(
    const CoordinateSystem& other) const noexcept override
  {
    const auto* p = dynamic_cast<const Cartesian*>(&other);
    return p != nullptr && p->dim_ == dim_;
  }

private:
  int dim_;
};

class Euclidean final : public CoordinateSystem
{
public:
  explicit Euclidean(std::vector<csys::Axis> axes) : axes_(std::move(axes)) {}

  [[nodiscard]] std::string_view Kind() const noexcept override
  {
    return "euclidean";
  }
  [[nodiscard]] int Dimension() const noexcept override
  {
    return static_cast<int>(axes_.size());
  }
  [[nodiscard]] bool HasOrthogonalBasis() const noexcept override
  {
    return true;
  }
  [[nodiscard]] bool HasUnitScaleFactors() const noexcept override
  {
    return true;
  }
  [[nodiscard]] bool operator==(
    const CoordinateSystem& other) const noexcept override
  {
    const auto* p = dynamic_cast<const Euclidean*>(&other);
    return p != nullptr && p->axes_ == axes_;
  }
  const std::vector<csys::Axis> axes_;
};

class CylindricalRThetaZ final : public CoordinateSystem
{
public:
  [[nodiscard]] std::string_view Kind() const noexcept override
  {
    return "cylindrical_rthetaz";
  }
  [[nodiscard]] int Dimension() const noexcept override { return 3; }
  [[nodiscard]] bool HasOrthogonalBasis() const noexcept override
  {
    return true;
  }
  [[nodiscard]] bool HasUnitScaleFactors() const noexcept override
  {
    return false;
  }
  [[nodiscard]] bool operator==(
    const CoordinateSystem& other) const noexcept override
  {
    return dynamic_cast<const CylindricalRThetaZ*>(&other) != nullptr;
  }
};

class CylindricalRZ final : public CoordinateSystem
{
public:
  [[nodiscard]] std::string_view Kind() const noexcept override
  {
    return "cylindrical_rz";
  }
  [[nodiscard]] int Dimension() const noexcept override { return 2; }
  [[nodiscard]] bool HasOrthogonalBasis() const noexcept override
  {
    return true;
  }
  [[nodiscard]] bool HasUnitScaleFactors() const noexcept override
  {
    return true;
  }
  [[nodiscard]] bool operator==(
    const CoordinateSystem& other) const noexcept override
  {
    return dynamic_cast<const CylindricalRZ*>(&other) != nullptr;
  }
};

} // namespace

namespace csys
{

std::shared_ptr<const CoordinateSystem> CreateCartesian(int dim)
{
  if (dim < 1 || dim > 3) {
    throw pcms_error("csys::CreateCartesian: dimension must be 1..3, got " +
                     std::to_string(dim));
  }
  return std::make_shared<const Cartesian>(dim);
}

std::shared_ptr<const CoordinateSystem> CreateEuclidean(std::vector<Axis> axes)
{
  const auto valid = [](Real x) { return x > 0 && std::isfinite(x); };
  bool ok = !axes.empty() && axes.size() <= 6;
  for (const auto& axis : axes) {
    ok = ok && valid(axis.scale) && (!axis.period || valid(*axis.period));
  }
  if (!ok) {
    throw pcms_error("csys::CreateEuclidean: need 1..6 axes with positive, "
                     "finite periods and scales");
  }
  return std::make_shared<const Euclidean>(std::move(axes));
}

std::shared_ptr<const CoordinateSystem> CreateEuclidean(int dim)
{
  return CreateEuclidean(std::vector<Axis>(std::max(dim, 0)));
}

std::vector<Axis> GetAxes(const CoordinateSystem& system)
{
  const auto* p = dynamic_cast<const Euclidean*>(&system);
  return p == nullptr ? std::vector<Axis>{} : p->axes_;
}

std::shared_ptr<const CoordinateSystem> CreateCylindricalRThetaZ()
{
  return std::make_shared<const CylindricalRThetaZ>();
}

std::shared_ptr<const CoordinateSystem> CreateCylindricalRZ()
{
  return std::make_shared<const CylindricalRZ>();
}

} // namespace csys

namespace detail
{

std::shared_ptr<const CoordinateSystem> DeferredCartesian()
{
  return std::make_shared<const Cartesian>(0);
}

bool IsCartesian(const CoordinateSystem& system) noexcept
{
  return dynamic_cast<const Cartesian*>(&system) != nullptr;
}

std::shared_ptr<const CoordinateSystem> ResolveCoordinateSystem(
  std::shared_ptr<const CoordinateSystem> system, int dim)
{
  if (system == nullptr) {
    throw pcms_error(
      "ResolveCoordinateSystem: coordinate system must not be null");
  }
  // To avoid requiring the user specifying the dimension twice in
  // FunctionSpace constructors, we have a default
  // Cartesian constructor that constructs with dimension 0.
  if (IsCartesian(*system) && system->Dimension() == 0) {
    return csys::CreateCartesian(dim);
  }
  if (system->Dimension() != dim) {
    throw pcms_error("ResolveCoordinateSystem: coordinate system '" +
                     std::string(system->Kind()) + "' has " +
                     std::to_string(system->Dimension()) +
                     " coordinate columns but the data has " +
                     std::to_string(dim));
  }
  return system;
}

} // namespace detail
} // namespace pcms
