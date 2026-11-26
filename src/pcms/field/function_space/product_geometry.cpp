#include "pcms/geometry/product_geometry.hpp"
#include "pcms/utility/arrays.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/coordinate_systems/parameter.hpp"
#include "product_binding.hpp"
#include <algorithm>
#include <vector>

namespace pcms
{

namespace
{

/// The columns [first, first + count) of `points` as a device view
/// (a free function: CUDA does not allow device lambdas in members).
Kokkos::View<Real**, DeviceMemorySpace> Columns(
  Kokkos::View<const Real**, DeviceMemorySpace> points, int first, int count)
{
  Kokkos::View<Real**, DeviceMemorySpace> out("product_factor_points",
                                              points.extent(0), count);
  pcms::detail::ForEach(
    "ProductColumns", static_cast<LO>(points.extent(0)),
    KOKKOS_LAMBDA(const LO p) {
      for (int d = 0; d < count; ++d) {
        out(p, d) = points(p, first + d);
      }
    });
  return out;
}

CoordinateView<DeviceMemorySpace> FactorView(
  const Geometry& g, const Kokkos::View<Real**, DeviceMemorySpace>& points)
{
  return CoordinateView<DeviceMemorySpace>(g.GetCoordinateSystem(),
                                           MakeConstRank2View(points));
}

/// The parameter system of the product of `a` and `b`, with their
/// periodic axes.
std::shared_ptr<const CoordinateSystem> ProductSystem(const Geometry& a,
                                                      const Geometry& b)
{
  const int da = a.GetCoordinateSystem()->Dimension();
  std::vector<PeriodicAxis> periodic_axes;
  for (const auto& id : a.GetCoordinateSystem()->PeriodicAxes()) {
    periodic_axes.push_back(id);
  }
  for (auto id : b.GetCoordinateSystem()->PeriodicAxes()) {
    if (id.transform != nullptr) {
      throw pcms_error("CreateProductGeometry: the second factor has a "
                       "twist-shift periodic axis; pass the product's "
                       "coordinate system");
    }
    id.axis += da;
    periodic_axes.push_back(id);
  }
  return csys::CreateParameterSystem("product",
                                     da + b.GetCoordinateSystem()->Dimension(),
                                     std::move(periodic_axes));
}

/// The points (i, k) of the located points of `a` and `b`, at i * nb + k.
Kokkos::View<Real**, DeviceMemorySpace> TensorPoints(const BoundPoints& a,
                                                     const BoundPoints& b)
{
  const auto pa = a.LocatedPoints().GetValues();
  const auto pb = b.LocatedPoints().GetValues();
  const auto nb = static_cast<LO>(pb.extent(0));
  const int da = static_cast<int>(pa.extent(1));
  const int db = static_cast<int>(pb.extent(1));
  Kokkos::View<Real**, DeviceMemorySpace> points(
    "tensor_points", static_cast<std::size_t>(pa.extent(0)) * pb.extent(0),
    da + db);
  pcms::detail::ForEach(
    "TensorPoints", static_cast<LO>(points.extent(0)),
    KOKKOS_LAMBDA(const LO i) {
      for (int d = 0; d < da; ++d) {
        points(i, d) = pa(i / nb, d);
      }
      for (int d = 0; d < db; ++d) {
        points(i, da + d) = pb(i % nb, d);
      }
    });
  return points;
}

/// The request to bind `points` on `product`, which must outlive it.
BindRequest TensorRequest(
  const Geometry& product,
  const Kokkos::View<const Real**, DeviceMemorySpace>& points,
  OutOfBoundsPolicy policy)
{
  const CoordinateView<DeviceMemorySpace> view(product.GetCoordinateSystem(),
                                               MakeConstRank2View(points));
  return BindRequest{view, policy, {}, nullptr, view};
}

/// Point (i, k) is found when i is found in `a` and k in `b`.
Kokkos::View<bool*, DeviceMemorySpace> TensorFound(const BoundPoints& a,
                                                   const BoundPoints& b)
{
  const auto fa = pcms::detail::FoundMask(a.NumPoints(), {a.MissingPoints()});
  const auto fb = pcms::detail::FoundMask(b.NumPoints(), {b.MissingPoints()});
  const LO nb = b.NumPoints();
  Kokkos::View<bool*, DeviceMemorySpace> found("tensor_found",
                                               fa.extent(0) * nb);
  pcms::detail::ForEach(
    "TensorFound", static_cast<LO>(found.extent(0)),
    KOKKOS_LAMBDA(const LO i) { found(i) = fa(i / nb) && fb(i % nb); });
  return found;
}

/// Appends the leaves under `bound` to `out`; false if a product level
/// is not a tensor product.
bool AppendLeaves(const std::shared_ptr<const BoundPoints>& bound,
                  const std::vector<std::shared_ptr<const Geometry>>& leaves,
                  pcms::detail::LeafBinding& out)
{
  const auto k = out.factors.size();
  PCMS_ALWAYS_ASSERT(k < leaves.size());
  if (bound->GetGeometry() == leaves[k]) {
    out.factors.push_back(bound);
    out.extent[k] = bound->NumPoints();
    return true;
  }
  const auto& product = dynamic_cast<const ProductBinding&>(*bound);
  const bool a = AppendLeaves(product.FactorA(), leaves, out);
  const bool b = AppendLeaves(product.FactorB(), leaves, out);
  return a && b && product.IsTensor();
}

} // namespace

ProductBinding::ProductBinding(std::shared_ptr<const Geometry> geometry,
                               const BindRequest& request)
  : BoundPoints(std::move(geometry), request)
{
  Locate(false);
}

ProductBinding::ProductBinding(std::shared_ptr<const Geometry> geometry,
                               std::shared_ptr<const BoundPoints> a,
                               std::shared_ptr<const BoundPoints> b)
  : ProductBinding(geometry, a, b, TensorPoints(*a, *b))
{
}

ProductBinding::ProductBinding(
  std::shared_ptr<const Geometry> geometry,
  std::shared_ptr<const BoundPoints> a, std::shared_ptr<const BoundPoints> b,
  Kokkos::View<const Real**, DeviceMemorySpace> points)
  : BoundPoints(geometry, TensorRequest(*geometry, points, a->Policy())),
    a_(std::move(a)),
    b_(std::move(b)),
    tensor_(true)
{
  if (a_->MissingPoints().extent(0) + b_->MissingPoints().extent(0) > 0) {
    SetFound(TensorFound(*a_, *b_));
  }
}

void ProductBinding::RebindImpl()
{
  tensor_ = false;
  Locate(true);
}

pcms::detail::LeafBinding pcms::detail::Flatten(
  const ProductBinding& binding,
  const std::vector<std::shared_ptr<const Geometry>>& leaves)
{
  LeafBinding out;
  const bool a = AppendLeaves(binding.FactorA(), leaves, out);
  const bool b = AppendLeaves(binding.FactorB(), leaves, out);
  out.tensor = a && b && binding.IsTensor();
  PCMS_ALWAYS_ASSERT(out.factors.size() == leaves.size());
  if (!out.tensor && binding.IsTensor()) {
    const auto scattered =
      binding.GetGeometry()->Bind(binding.LocatedPoints(), binding.Policy());
    return Flatten(static_cast<const ProductBinding&>(*scattered), leaves);
  }
  return out;
}

void ProductBinding::Locate(bool rebind)
{
  const auto& product = static_cast<const ProductGeometry&>(*GetGeometry());
  const auto points = LocatedPointValues();
  const int da = product.FactorA()->GetCoordinateSystem()->Dimension();
  const int db = product.FactorB()->GetCoordinateSystem()->Dimension();
  const auto xa = Columns(points, 0, da);
  const auto xb = Columns(points, da, db);
  const auto view_a = FactorView(*product.FactorA(), xa);
  const auto view_b = FactorView(*product.FactorB(), xb);
  if (rebind) {
    a_ = product.FactorA()->Rebind(std::move(a_), view_a);
    b_ = product.FactorB()->Rebind(std::move(b_), view_b);
  } else {
    a_ = product.FactorA()->Bind(view_a, Policy());
    b_ = product.FactorB()->Bind(view_b, Policy());
  }
  if (a_->MissingPoints().extent(0) + b_->MissingPoints().extent(0) > 0) {
    SetFound(pcms::detail::FoundMask(
      NumPoints(), {a_->MissingPoints(), b_->MissingPoints()}));
  }
}

ProductGeometry::ProductGeometry(std::shared_ptr<const Geometry> a,
                                 std::shared_ptr<const Geometry> b,
                                 std::shared_ptr<const CoordinateSystem> system)
  : a_(std::move(a)), b_(std::move(b)), system_(std::move(system))
{
  if (a_ == nullptr || b_ == nullptr) {
    throw pcms_error("CreateProductGeometry: the factors must not be null");
  }
  if (system_ == nullptr) {
    system_ = ProductSystem(*a_, *b_);
  }
  if (system_->HasMetric() ||
      system_->Dimension() != a_->GetCoordinateSystem()->Dimension() +
                                b_->GetCoordinateSystem()->Dimension()) {
    throw pcms_error("CreateProductGeometry: the product's coordinate system "
                     "must have no metric and the summed dimension of the "
                     "factors");
  }
}

std::shared_ptr<BoundPoints> ProductGeometry::BindImpl(
  const BindRequest& request) const
{
  return std::make_shared<ProductBinding>(shared_from_this(), request);
}
Quadrature ProductGeometry::TensorQuadrature(const Quadrature& qa,
                                             const Quadrature& qb) const
{
  const auto wa = qa.weights;
  const auto wb = qb.weights;
  const auto nb = static_cast<LO>(wb.extent(0));
  Kokkos::View<Real*, DeviceMemorySpace> weights("tensor_weights",
                                                 wa.extent(0) * wb.extent(0));
  pcms::detail::ForEach(
    "TensorQuadrature", static_cast<LO>(weights.extent(0)),
    KOKKOS_LAMBDA(const LO i) { weights(i) = wa(i / nb) * wb(i % nb); });
  return Quadrature{
    std::make_shared<ProductBinding>(shared_from_this(), qa.points, qb.points),
    weights,
    {},
    std::min(qa.degree, qb.degree)};
}

Quadrature ProductGeometry::CreateQuadrature(
  const detail::QuadratureRule& rule) const
{
  return TensorQuadrature(detail::CreateQuadrature(*a_, rule),
                          detail::CreateQuadrature(*b_, rule));
}

std::shared_ptr<const Geometry> CreateProductGeometry(
  std::shared_ptr<const Geometry> a, std::shared_ptr<const Geometry> b,
  std::shared_ptr<const CoordinateSystem> system)
{
  return std::make_shared<ProductGeometry>(std::move(a), std::move(b),
                                           std::move(system));
}

} // namespace pcms
