#ifndef PCMS_FIELD_FUNCTION_SPACE_PRODUCT_BINDING_HPP
#define PCMS_FIELD_FUNCTION_SPACE_PRODUCT_BINDING_HPP

// Private: the product geometry class and its binding.

#include "pcms/geometry/geometry.hpp"
#include "pcms/geometry/product_geometry.hpp"
#include "pcms/utility/detail/memory_spaces.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <vector>

namespace pcms
{

class ProductGeometry;

/// BoundPoints of a product geometry: the factors' bound points of the factor
/// parts of the query points, or of the factors' own points for a tensor
/// product of them.
class ProductBinding final : public BoundPoints
{
public:
  /// @brief Each factor binds its part of every point
  ProductBinding(std::shared_ptr<const Geometry> geometry,
                 const BindRequest& request);
  /// @brief The tensor product of the factors' bound points: point (i, k) at
  ///        i * (points of b) + k
  ProductBinding(std::shared_ptr<const Geometry> geometry,
                 std::shared_ptr<const BoundPoints> a,
                 std::shared_ptr<const BoundPoints> b);

  [[nodiscard]] const std::shared_ptr<const BoundPoints>& FactorA()
    const noexcept
  {
    return a_;
  }
  [[nodiscard]] const std::shared_ptr<const BoundPoints>& FactorB()
    const noexcept
  {
    return b_;
  }
  /// @return whether the points are the tensor product of FactorA's and
  ///         FactorB's
  [[nodiscard]] bool IsTensor() const noexcept { return tensor_; }

protected:
  void RebindImpl() override;

private:
  ProductBinding(std::shared_ptr<const Geometry> geometry,
                 std::shared_ptr<const BoundPoints> a,
                 std::shared_ptr<const BoundPoints> b,
                 Kokkos::View<const Real**, DeviceMemorySpace> points);

  void Locate(bool rebind);

  std::shared_ptr<const BoundPoints> a_;
  std::shared_ptr<const BoundPoints> b_;
  bool tensor_ = false;
};

class ProductGeometry final : public Geometry
{
public:
  ProductGeometry(std::shared_ptr<const Geometry> a,
                  std::shared_ptr<const Geometry> b,
                  std::shared_ptr<const CoordinateSystem> system);

  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>&
  GetCoordinateSystem() const noexcept override
  {
    return system_;
  }
  [[nodiscard]] bool SupportsNearestBoundary() const noexcept override
  {
    return a_->SupportsNearestBoundary() && b_->SupportsNearestBoundary();
  }
  [[nodiscard]] const std::shared_ptr<const Geometry>& FactorA() const noexcept
  {
    return a_;
  }
  [[nodiscard]] const std::shared_ptr<const Geometry>& FactorB() const noexcept
  {
    return b_;
  }

  /// The tensor product of a rule of each factor: the point (i, k) is the
  /// i-th point of `qa` with the k-th of `qb`, at index i * qb points + k,
  /// with the product of the weights.
  [[nodiscard]] Quadrature TensorQuadrature(const Quadrature& qa,
                                            const Quadrature& qb) const;

protected:
  [[nodiscard]] std::shared_ptr<BoundPoints> BindImpl(
    const BindRequest& request) const override;

  /// The tensor product of the factors' rules.
  [[nodiscard]] Quadrature CreateQuadrature(
    const detail::QuadratureRule& rule) const override;

private:
  std::shared_ptr<const Geometry> a_;
  std::shared_ptr<const Geometry> b_;
  std::shared_ptr<const CoordinateSystem> system_;
};

namespace detail
{

/// @brief The most leaf factors of a product: their dimensions add up to 6 at
///        most
inline constexpr int MaxFactors = 6;

/// @brief The bound points of the leaf factors of a product binding
struct LeafBinding
{
  std::vector<std::shared_ptr<const BoundPoints>> factors;
  /// for a tensor binding, the points of each leaf (the first leaf slowest)
  Kokkos::Array<LO, MaxFactors> extent = {};
  /// whether the points are the tensor product of the leaves' points
  bool tensor = true;
};

/// @return the bound points under `binding` of the leaves on `geometries`, in
///         coordinate order; a binding that is a tensor product at some
///         levels only is bound again, every leaf at every point
LeafBinding Flatten(const ProductBinding& binding,
                    const std::vector<std::shared_ptr<const Geometry>>& leaves);

} // namespace detail

} // namespace pcms

#endif // PCMS_FIELD_FUNCTION_SPACE_PRODUCT_BINDING_HPP
