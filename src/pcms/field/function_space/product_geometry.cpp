#include "pcms/geometry/product_geometry.hpp"
#include "pcms/utility/arrays.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/coordinate_systems/cartesian.hpp"
#include "product_binding.hpp"

namespace pcms
{

namespace
{

/// The columns [first, first + count) of `points` as a device view
/// (a free function: CUDA does not allow device lambdas in members).
Kokkos::View<Real**, DeviceMemorySpace> Columns(
  Rank2View<const Real, DeviceMemorySpace> points, int first, int count)
{
  Kokkos::View<Real**, DeviceMemorySpace> out(
    "product_factor_points", points.extent(0), count);
  pcms::detail::ForEach(
    "ProductColumns", static_cast<LO>(points.extent(0)),
    KOKKOS_LAMBDA(const LO p) {
      for (int d = 0; d < count; ++d) {
        out(p, d) = points(p, first + d);
      }
    });
  return out;
}

/// found = found in both factors
Kokkos::View<bool*, DeviceMemorySpace> BothFound(
  LO n, Kokkos::View<const LO*, DeviceMemorySpace> missing_a,
  Kokkos::View<const LO*, DeviceMemorySpace> missing_b)
{
  Kokkos::View<bool*, DeviceMemorySpace> found("product_found", n);
  Kokkos::deep_copy(found, true);
  pcms::detail::ForEach(
    "ProductMissingA", static_cast<LO>(missing_a.extent(0)),
    KOKKOS_LAMBDA(const LO i) { found(missing_a(i)) = false; });
  pcms::detail::ForEach(
    "ProductMissingB", static_cast<LO>(missing_b.extent(0)),
    KOKKOS_LAMBDA(const LO i) { found(missing_b(i)) = false; });
  return found;
}

CoordinateView<DeviceMemorySpace> FactorView(
  const Geometry& g, const Kokkos::View<Real**, DeviceMemorySpace>& points)
{
  return CoordinateView<DeviceMemorySpace>(g.GetCoordinateSystem(),
                                           MakeConstRank2View(points));
}

} // namespace

ProductBinding::ProductBinding(std::shared_ptr<const Geometry> geometry,
                               const BindRequest& request)
  : BoundPoints(std::move(geometry), request)
{
  Locate(false);
}

void ProductBinding::RebindImpl()
{
  Locate(true);
}

void ProductBinding::Locate(bool rebind)
{
  const auto& product = static_cast<const ProductGeometry&>(*GetGeometry());
  const auto points = LocatedPointValues();
  const int da = product.FactorA()->GetCoordinateSystem()->Dimension();
  const int db = product.FactorB()->GetCoordinateSystem()->Dimension();
  const auto xa = Columns(MakeConstRank2View(points), 0, da);
  const auto xb = Columns(MakeConstRank2View(points), da, db);
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
    SetFound(BothFound(NumPoints(), a_->MissingPoints(), b_->MissingPoints()));
  }
}

ProductGeometry::ProductGeometry(std::shared_ptr<const Geometry> a,
                                 std::shared_ptr<const Geometry> b)
  : a_(std::move(a)), b_(std::move(b))
{
  if (a_ == nullptr || b_ == nullptr) {
    throw pcms_error("CreateProductGeometry: the factors must not be null");
  }
  // moves to the composition of the factors' systems (S4/S5)
  if (!pcms::detail::IsCartesian(*a_->GetCoordinateSystem()) ||
      !pcms::detail::IsCartesian(*b_->GetCoordinateSystem())) {
    throw pcms_error("CreateProductGeometry: the factors must be Cartesian");
  }
  system_ = csys::CreateCartesian(a_->GetCoordinateSystem()->Dimension() +
                                  b_->GetCoordinateSystem()->Dimension());
}

std::shared_ptr<BoundPoints> ProductGeometry::BindImpl(
  const BindRequest& request) const
{
  return std::make_shared<ProductBinding>(shared_from_this(), request);
}

DifferentialView ProductGeometry::Differential(const BoundPoints& bound) const
{
  const auto* binding = dynamic_cast<const ProductBinding*>(&bound);
  if (binding == nullptr || bound.GetGeometry().get() != this) {
    throw pcms_error("ProductGeometry::Differential: the points belong to "
                     "another geometry");
  }
  const DifferentialView ja = a_->Differential(*binding->FactorA());
  const DifferentialView jb = b_->Differential(*binding->FactorB());
  // per-element differentials need the element of each point, which the bound
  // points do not give until Differential moves to them (S4/S5)
  if (ja.Empty() || jb.Empty() || ja.per_element || jb.per_element ||
      bound.MissingPoints().extent(0) > 0) {
    throw pcms_error("ProductGeometry::Differential: needs per-point "
                     "differentials of both factors and no missing points");
  }
  const auto n = static_cast<std::size_t>(bound.NumPoints());
  const int da = a_->GetCoordinateSystem()->Dimension();
  const int db = b_->GetCoordinateSystem()->Dimension();
  Kokkos::View<Real***, DeviceMemorySpace> jacobian("product_jacobian", n,
                                                    da + db, da + db);
  Kokkos::View<Real*, DeviceMemorySpace> determinant("product_determinant", n);
  const auto jac_a = ja.jacobian;
  const auto jac_b = jb.jacobian;
  const auto det_a = ja.determinant;
  const auto det_b = jb.determinant;
  pcms::detail::ForEach(
    "ProductDifferential", static_cast<LO>(n), KOKKOS_LAMBDA(const LO p) {
      for (int i = 0; i < da; ++i) {
        for (int j = 0; j < da; ++j) {
          jacobian(p, i, j) = jac_a(p, i, j);
        }
      }
      for (int i = 0; i < db; ++i) {
        for (int j = 0; j < db; ++j) {
          jacobian(p, da + i, da + j) = jac_b(p, i, j);
        }
      }
      determinant(p) = det_a(p) * det_b(p);
    });
  return {jacobian, determinant, false};
}

Quadrature ProductGeometry::TensorQuadrature(const Quadrature& qa,
                                             const Quadrature& qb) const
{
  const auto pa = qa.points->LocatedPointValues();
  const auto pb = qb.points->LocatedPointValues();
  const auto na = static_cast<LO>(pa.extent(0));
  const auto nb = static_cast<LO>(pb.extent(0));
  const int da = static_cast<int>(pa.extent(1));
  const int db = static_cast<int>(pb.extent(1));
  const auto wa = qa.weights;
  const auto wb = qb.weights;
  Kokkos::View<Real**, DeviceMemorySpace> points("tensor_points",
                                                 static_cast<std::size_t>(na) * nb,
                                                 da + db);
  Kokkos::View<Real*, DeviceMemorySpace> weights(
    "tensor_weights", static_cast<std::size_t>(na) * nb);
  pcms::detail::ForEach(
    "TensorQuadrature", na * nb, KOKKOS_LAMBDA(const LO i) {
      const LO ia = i / nb;
      const LO ib = i % nb;
      for (int d = 0; d < da; ++d) {
        points(i, d) = pa(ia, d);
      }
      for (int d = 0; d < db; ++d) {
        points(i, da + d) = pb(ib, d);
      }
      weights(i) = wa(ia) * wb(ib);
    });
  return Quadrature{Bind(CoordinateView<DeviceMemorySpace>(
                      system_, MakeConstRank2View(points))),
                    weights,
                    {}};
}

Quadrature ProductGeometry::CreateQuadrature(
  const QuadratureOptions& options) const
{
  return TensorQuadrature(a_->CreateQuadrature(options),
                          b_->CreateQuadrature(options));
}

std::shared_ptr<const Geometry> CreateProductGeometry(
  std::shared_ptr<const Geometry> a, std::shared_ptr<const Geometry> b)
{
  return std::make_shared<ProductGeometry>(std::move(a), std::move(b));
}

} // namespace pcms
