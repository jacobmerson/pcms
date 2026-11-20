#include "pcms/geometry/product_geometry.hpp"
#include "pcms/field/function_space.hpp"
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

/// The twist-shift: moves the wrap axis of `xa` into the interval and shifts
/// the first column of `xb` by the number of periods times `shift`, evaluated
/// at the other columns of `xa`.
void ApplyIdentification(const ProductIdentification& id,
                         Kokkos::View<Real**, DeviceMemorySpace> xa,
                         Kokkos::View<Real**, DeviceMemorySpace> xb)
{
  const int da = static_cast<int>(xa.extent(1));
  const int axis = id.wrap_axis;
  Kokkos::View<Real**, DeviceMemorySpace> other("identification_other",
                                                xa.extent(0), da - 1);
  pcms::detail::ForEach(
    "IdentificationOther", static_cast<LO>(xa.extent(0)),
    KOKKOS_LAMBDA(const LO p) {
      for (int d = 0, k = 0; d < da; ++d) {
        if (d != axis) {
          other(p, k++) = xa(p, d);
        }
      }
    });
  const FunctionSpace& space = id.shift->GetSpace();
  const auto geometry = space.GetGeometry();
  const auto evaluator = space.CreatePointEvaluator<Real>(
    geometry->Bind(CoordinateView<DeviceMemorySpace>(
      geometry->GetCoordinateSystem(), MakeConstRank2View(other))));
  Kokkos::View<Real**, DeviceMemorySpace> shift("identification_shift",
                                                xa.extent(0), 1);
  evaluator->Evaluate(
    *id.shift, ValueView<Real, DeviceMemorySpace>(
                 evaluator->OutputBasis(id.shift->GetData().GetValueBasis()),
                 MakeRank2View(shift)));
  const Real lower = id.lower;
  const Real period = id.period;
  pcms::detail::ForEach(
    "Identification", static_cast<LO>(xa.extent(0)), KOKKOS_LAMBDA(const LO p) {
      const Real periods = Kokkos::floor((xa(p, axis) - lower) / period);
      xa(p, axis) -= periods * period;
      xb(p, 0) += periods * shift(p, 0);
    });
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
  if (product.Identification()) {
    ApplyIdentification(*product.Identification(), xa, xb);
  }
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

ProductGeometry::ProductGeometry(
  std::shared_ptr<const Geometry> a, std::shared_ptr<const Geometry> b,
  std::optional<ProductIdentification> identification)
  : a_(std::move(a)),
    b_(std::move(b)),
    identification_(std::move(identification))
{
  if (a_ == nullptr || b_ == nullptr) {
    throw pcms_error("CreateProductGeometry: the factors must not be null");
  }
  // the product's system is the Cartesian of the summed dimension: the factors'
  // systems are not composed (a product of systems, with its metric and
  // periods, is not implemented)
  if (!pcms::detail::IsCartesian(*a_->GetCoordinateSystem()) ||
      !pcms::detail::IsCartesian(*b_->GetCoordinateSystem())) {
    throw pcms_error("CreateProductGeometry: the factors must be Cartesian");
  }
  system_ = csys::CreateCartesian(a_->GetCoordinateSystem()->Dimension() +
                                  b_->GetCoordinateSystem()->Dimension());
  if (identification_) {
    const int da = a_->GetCoordinateSystem()->Dimension();
    auto& id = *identification_;
    // a system that declares the wrap axis periodic gives the period, which
    // must be the one given
    if (id.wrap_axis >= 0 && id.wrap_axis < da) {
      const auto declared = a_->GetCoordinateSystem()->Period(id.wrap_axis);
      if (declared && id.period != 0 && id.period != *declared) {
        throw pcms_error("CreateProductGeometry: the identification's period "
                         "differs from the period of the coordinate system");
      }
      if (declared) {
        id.period = *declared;
      }
    }
    if (id.wrap_axis < 0 || id.wrap_axis >= da || da < 2 || !(id.period > 0) ||
        id.shift == nullptr ||
        id.shift->GetSpace()
            .GetGeometry()
            ->GetCoordinateSystem()
            ->Dimension() != da - 1) {
      throw pcms_error("CreateProductGeometry: the identification needs a wrap "
                       "axis of a (of dimension 2 or more), a positive period "
                       "and a shift function of the other coordinates of a");
    }
  }
}

std::shared_ptr<BoundPoints> ProductGeometry::BindImpl(
  const BindRequest& request) const
{
  return std::make_shared<ProductBinding>(shared_from_this(), request);
}

DifferentialView ProductBinding::Differential() const
{
  const auto& product = static_cast<const ProductGeometry&>(*GetGeometry());
  const DifferentialView ja = a_->Differential();
  const DifferentialView jb = b_->Differential();
  // a per-element differential is indexed by the element of each point,
  // which the factor's bound points do not give
  if (ja.Empty() || jb.Empty() || ja.per_element || jb.per_element ||
      MissingPoints().extent(0) > 0) {
    throw pcms_error("ProductBinding::Differential: needs per-point "
                     "differentials of both factors and no missing points");
  }
  const auto n = static_cast<std::size_t>(NumPoints());
  const int da = product.FactorA()->GetCoordinateSystem()->Dimension();
  const int db = product.FactorB()->GetCoordinateSystem()->Dimension();
  jacobian_ = Kokkos::View<Real***, DeviceMemorySpace>("product_jacobian", n,
                                                       da + db, da + db);
  determinant_ =
    Kokkos::View<Real*, DeviceMemorySpace>("product_determinant", n);
  const auto jacobian = jacobian_;
  const auto determinant = determinant_;
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
  return DifferentialView{View<3, const Real, DeviceMemorySpace>(
                            jacobian_.data(), n, da + db, da + db),
                          MakeRank1View(determinant_), /*per_element=*/false};
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
  Kokkos::View<Real**, DeviceMemorySpace> points(
    "tensor_points", static_cast<std::size_t>(na) * nb, da + db);
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
  std::shared_ptr<const Geometry> a, std::shared_ptr<const Geometry> b,
  std::optional<ProductIdentification> identification)
{
  return std::make_shared<ProductGeometry>(std::move(a), std::move(b),
                                           std::move(identification));
}

} // namespace pcms
