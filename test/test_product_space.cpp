#include "periodic_test_utils.hpp"
#include "field_test_utils.hpp"
#include "pcms/field/point_rows.hpp"
#include <Omega_h_library.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <pcms/backends/uniform_grid/geometry.hpp>
#include <pcms/field/periodic_axis.hpp>
#include <pcms/geometry/product_geometry.hpp>
#include <pcms/utility/coordinate_systems/parameter.hpp>
#include <pcms/configuration.hpp>
#include <pcms/transfer/forms.hpp>
#include <pcms/transfer/l2_inner_product.hpp>
#include <pcms/transfer/projection.hpp>
#include <pcms/transfer/detail/projection.hpp>
#include <pcms/utility/coordinate_systems/cartesian.hpp>
#include <chrono>
#include <cmath>
#include <numbers>

namespace
{

using pcms::LO;
using pcms::Real;
using pcms::test::EvaluateAt;
using pcms::test::SetDofs;
constexpr Real TwoPi = 2 * std::numbers::pi_v<Real>;

std::shared_ptr<const pcms::Geometry> Periodic(int cells, Real period = TwoPi)
{
  return pcms::test::PeriodicGrid(cells, period);
}

std::vector<std::vector<Real>> Points(int n, int dim, Real lo, Real hi)
{
  std::vector<std::vector<Real>> points;
  for (int i = 0; i < n; ++i) {
    std::vector<Real> x;
    for (int d = 0; d < dim; ++d) {
      x.push_back(
        lo +
        (hi - lo) *
          std::fmod(0.1234 + 0.6180339 * (i + 1) * (d + 1) + 0.31 * d, 1.0));
    }
    points.push_back(x);
  }
  return points;
}

// the DOFs (i, k) = a_i b_k of the product of two spaces
std::vector<Real> Outer(const std::vector<Real>& a, const std::vector<Real>& b)
{
  std::vector<Real> c;
  for (const Real x : a) {
    for (const Real y : b) {
      c.push_back(x * y);
    }
  }
  return c;
}

// f(z) = 1 + 0.5 cos z - 0.3 sin 2 z, in the DOFs of modes {0, 1, 2}
const std::vector<Real> ZCoefficients = {1.0, 0.5, 0.0, 0.0, -0.3};
Real Z(Real z)
{
  return 1 + 0.5 * std::cos(z) - 0.3 * std::sin(2 * z);
}
Real DZ(Real z)
{
  return -0.5 * std::sin(z) - 0.6 * std::cos(2 * z);
}

// The nodal values of f on the DOF holders of a layout with coordinates
template <typename F>
std::vector<Real> NodalValues(const pcms::FieldLayout& layout, F f)
{
  const auto device = layout.GetDOFHolderCoordinates().GetValues();
  Kokkos::View<Real**, pcms::DeviceMemorySpace> copy("x", device.extent(0),
                                                     device.extent(1));
  pcms::detail::CopyDeviceRank2ViewToDeviceView(copy, device);
  const auto x =
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace{}, copy);
  std::vector<Real> values;
  for (std::size_t i = 0; i < x.extent(0); ++i) {
    values.push_back(f(x(i, 0), x.extent(1) > 1 ? x(i, 1) : 0.0));
  }
  return values;
}

// An evaluation-only source: an analytic function of the 3D points
struct Source
{
  KOKKOS_INLINE_FUNCTION Real operator()(Real x, Real y, Real z) const
  {
    return (x + 2 * y) * (1 + 0.5 * Kokkos::cos(z) - 0.3 * Kokkos::sin(2 * z));
  }
};

class SourceEvaluator final : public pcms::PointEvaluator<Real>
{
public:
  explicit SourceEvaluator(std::shared_ptr<const pcms::BoundPoints> bound)
    : PointEvaluator(std::move(bound))
  {
  }
  void EvaluateImpl(
    const pcms::Field<Real>&,
    pcms::ValueView<Real, pcms::DeviceMemorySpace> out) const override
  {
    const auto values = out.GetValues();
    const auto points = Bound().LocatedPoints().GetValues();
    pcms::detail::ForEach(
      "source", static_cast<pcms::LO>(points.extent(0)),
      KOKKOS_LAMBDA(const pcms::LO p) {
        values(p, 0) = Source{}(points(p, 0), points(p, 1), points(p, 2));
      });
  }
};

class SourceFactory final : public pcms::PointEvaluatorFactory
{
public:
  explicit SourceFactory(std::shared_ptr<const pcms::Geometry> geometry)
    : geometry_(std::move(geometry))
  {
  }
  std::shared_ptr<const pcms::Geometry> GetGeometry() const noexcept override
  {
    return geometry_;
  }
  bool Supports(pcms::DifferentialOperator op) const noexcept override
  {
    return op == pcms::DifferentialOperator::Value;
  }

protected:
  pcms::PointEvaluatorVariant CreatePointEvaluatorImpl(
    pcms::Type, std::shared_ptr<const pcms::BoundPoints> bound,
    pcms::DifferentialOperator) const override
  {
    return std::unique_ptr<pcms::PointEvaluator<Real>>(
      std::make_unique<SourceEvaluator>(std::move(bound)));
  }

private:
  std::shared_ptr<const pcms::Geometry> geometry_;
};

// The values of `field` at `bound`, one row per point.
std::vector<std::vector<Real>> EvaluateBound(
  const pcms::FunctionSpace& space, const pcms::Field<Real>& field,
  std::shared_ptr<const pcms::BoundPoints> bound,
  pcms::DifferentialOperator op = pcms::DifferentialOperator::Value)
{
  const auto evaluator = space.CreatePointEvaluator<Real>(bound, op);
  const auto columns =
    op == pcms::DifferentialOperator::Value
      ? 1
      : space.GetGeometry()->GetCoordinateSystem()->Dimension();
  Kokkos::View<Real**, pcms::DeviceMemorySpace> out("out", bound->NumPoints(),
                                                    columns);
  evaluator->Evaluate(field,
                      pcms::ValueView<Real, pcms::DeviceMemorySpace>(
                        evaluator->OutputBasis(field.GetData().GetValueBasis()),
                        pcms::MakeRank2View(out)));
  const auto host =
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace{}, out);
  std::vector<std::vector<Real>> result(host.extent(0));
  for (std::size_t p = 0; p < host.extent(0); ++p) {
    for (int c = 0; c < columns; ++c) {
      result[p].push_back(host(p, c));
    }
  }
  return result;
}

// The DOFs of E^T y at `bound`.
std::vector<Real> TransposeBound(const pcms::FunctionSpace& space,
                                 std::shared_ptr<const pcms::BoundPoints> bound,
                                 const std::vector<Real>& y)
{
  const auto evaluator = space.CreatePointEvaluator<Real>(bound);
  auto functional = space.CreateFunction<Real>();
  const auto device = pcms::test::DevicePoints([&] {
    std::vector<std::vector<Real>> rows;
    for (const Real v : y) {
      rows.push_back({v});
    }
    return rows;
  }());
  evaluator->EvaluateTranspose(
    pcms::ValueView<const Real, pcms::DeviceMemorySpace>(
      functional.GetData().GetValueBasis(), pcms::MakeConstRank2View(device)),
    functional);
  return pcms::test::GetDofs(functional);
}

// The DOFs of the product of the factors' DOFs, the first factor slowest
std::vector<Real> OuterAll(const std::vector<std::vector<Real>>& factors)
{
  std::vector<Real> c = {1.0};
  for (const auto& f : factors) {
    c = Outer(c, f);
  }
  return c;
}

void CheckClose(const std::vector<std::vector<Real>>& a,
                const std::vector<std::vector<Real>>& b, Real margin)
{
  REQUIRE(a.size() == b.size());
  for (std::size_t p = 0; p < a.size(); ++p) {
    REQUIRE(a[p].size() == b[p].size());
    for (std::size_t c = 0; c < a[p].size(); ++c) {
      CHECK(a[p][c] == Catch::Approx(b[p][c]).margin(margin));
    }
  }
}

// The Kronecker rows of two factors' rows: the explicit evaluation of a
// product (the rows the product space built before it was applied factor by
// factor), here as the reference.
struct FactorMatrices
{
  pcms::detail::PointRows va, vb, ga, gb;
};

struct RowSelect
{
  int num_out, da, db;
  bool gradient;
  KOKKOS_INLINE_FUNCTION void operator()(const LO r, bool& a_gradient,
                                         bool& b_gradient, LO& row_a,
                                         LO& row_b) const
  {
    const LO p = r / num_out;
    const int c = r % num_out;
    a_gradient = gradient && c < da;
    b_gradient = gradient && c >= da;
    row_a = a_gradient ? p * da + c : p;
    row_b = b_gradient ? p * db + (c - da) : p;
  }
};

pcms::detail::PointRows KroneckerRows(const FactorMatrices& m, LO num_points,
                                      int da, int db, bool gradient,
                                      LO holders_b)
{
  const int num_out = gradient ? da + db : 1;
  const LO num_rows = num_points * num_out;
  const RowSelect select{num_out, da, db, gradient};
  Kokkos::View<LO*, pcms::DeviceMemorySpace> offsets("product_offsets",
                                                     num_rows + 1);
  Kokkos::parallel_scan(
    "ProductRowOffsets",
    Kokkos::RangePolicy<pcms::DefaultExecutionSpace>(0, num_rows + 1),
    KOKKOS_LAMBDA(const LO r, LO& partial, const bool final) {
      if (final) {
        offsets(r) = partial;
      }
      if (r < num_rows) {
        bool ag, bg;
        LO ra, rb;
        select(r, ag, bg, ra, rb);
        const auto& a = ag ? m.ga : m.va;
        const auto& b = bg ? m.gb : m.vb;
        partial += (a.offsets(ra + 1) - a.offsets(ra)) *
                   (b.offsets(rb + 1) - b.offsets(rb));
      }
    });
  LO nnz = 0;
  Kokkos::deep_copy(nnz, Kokkos::subview(offsets, num_rows));
  Kokkos::View<LO*, pcms::DeviceMemorySpace> holders("product_holders", nnz);
  Kokkos::View<Real*, pcms::DeviceMemorySpace> values("product_values", nnz);
  Kokkos::parallel_for(
    "ProductRows",
    Kokkos::RangePolicy<pcms::DefaultExecutionSpace>(0, num_rows),
    KOKKOS_LAMBDA(const LO r) {
      bool ag, bg;
      LO ra, rb;
      select(r, ag, bg, ra, rb);
      const auto& a = ag ? m.ga : m.va;
      const auto& b = bg ? m.gb : m.vb;
      LO k = offsets(r);
      for (LO i = a.offsets(ra); i < a.offsets(ra + 1); ++i) {
        for (LO j = b.offsets(rb); j < b.offsets(rb + 1); ++j, ++k) {
          holders(k) = a.holders(i) * holders_b + b.holders(j);
          values(k) = a.values(i) * b.values(j);
        }
      }
    });
  return {offsets, holders, values, num_out};
}

// The value and gradient rows of a space at points of its geometry
struct Rows
{
  pcms::detail::PointRows value, gradient;
  int dimension;
  LO holders;
};

Rows FactorRowsAt(const pcms::FunctionSpace& space,
                  Kokkos::View<Real**, pcms::DeviceMemorySpace> points,
                  bool gradient)
{
  const auto bound =
    space.GetGeometry()->Bind(pcms::CoordinateView<pcms::DeviceMemorySpace>(
      space.GetGeometry()->GetCoordinateSystem(),
      pcms::MakeConstRank2View(points)));
  const auto rows_of = [](const pcms::PointEvaluator<Real>& evaluator) {
    const auto* source = pcms::detail::AsRowsSource(evaluator);
    REQUIRE(source != nullptr);
    return source->Rows();
  };
  Rows rows;
  rows.dimension = static_cast<int>(points.extent(1));
  rows.holders = space.GetLayout()->GetNumDofHolder();
  rows.value = rows_of(*space.CreatePointEvaluator<Real>(bound));
  if (gradient) {
    rows.gradient = rows_of(*space.CreatePointEvaluator<Real>(
      bound, pcms::DifferentialOperator::Gradient));
  }
  return rows;
}

// The columns [first, first + count) of the points
Kokkos::View<Real**, pcms::DeviceMemorySpace> Columns(
  Kokkos::View<const Real**, pcms::DeviceMemorySpace> points, int first,
  int count)
{
  Kokkos::View<Real**, pcms::DeviceMemorySpace> out("columns", points.extent(0),
                                                    count);
  Kokkos::parallel_for(
    "Columns",
    Kokkos::RangePolicy<pcms::DefaultExecutionSpace>(0, points.extent(0)),
    KOKKOS_LAMBDA(const int p) {
      for (int d = 0; d < count; ++d) {
        out(p, d) = points(p, first + d);
      }
    });
  return out;
}

// The Kronecker rows of the leaves at the points, nested from the left
Rows KroneckerReference(
  const std::vector<std::shared_ptr<const pcms::FunctionSpace>>& leaves,
  Kokkos::View<const Real**, pcms::DeviceMemorySpace> points, bool gradient)
{
  const auto num_points = static_cast<LO>(points.extent(0));
  int first = 0;
  Rows product;
  for (const auto& leaf : leaves) {
    const int d = leaf->GetGeometry()->GetCoordinateSystem()->Dimension();
    const auto rows = FactorRowsAt(*leaf, Columns(points, first, d), gradient);
    first += d;
    if (leaf == leaves.front()) {
      product = rows;
      continue;
    }
    const FactorMatrices m{product.value, rows.value, product.gradient,
                           rows.gradient};
    Rows next;
    next.value = KroneckerRows(m, num_points, product.dimension, rows.dimension,
                               false, rows.holders);
    if (gradient) {
      next.gradient = KroneckerRows(m, num_points, product.dimension,
                                    rows.dimension, true, rows.holders);
    }
    next.dimension = product.dimension + rows.dimension;
    next.holders = product.holders * rows.holders;
    product = next;
  }
  return product;
}

// The rows applied to the DOFs, one row per point
std::vector<std::vector<Real>> ApplyRows(const pcms::detail::PointRows& rows,
                                         const std::vector<Real>& dofs,
                                         LO num_points)
{
  const auto x = pcms::test::DevicePoints([&] {
    std::vector<std::vector<Real>> c;
    for (const Real v : dofs) {
      c.push_back({v});
    }
    return c;
  }());
  Kokkos::View<Real**, pcms::DeviceMemorySpace> out("out", num_points,
                                                    rows.num_out);
  pcms::detail::Apply(rows, x, out);
  const auto host =
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace{}, out);
  std::vector<std::vector<Real>> result(num_points);
  for (LO p = 0; p < num_points; ++p) {
    for (int c = 0; c < rows.num_out; ++c) {
      result[p].push_back(host(p, c));
    }
  }
  return result;
}

std::shared_ptr<const pcms::Geometry> Line(int cells, Real lo, Real length)
{
  pcms::UniformGrid<1> grid;
  grid.bot_left = {lo};
  grid.edge_length = {length};
  grid.divisions = {cells};
  return pcms::uniform_grid::CreateGeometry(grid,
                                            pcms::csys::CreateCartesian(1));
}

// The points a quadrature's points were located at, one row per point
std::vector<std::vector<Real>> HostPoints(const pcms::BoundPoints& bound)
{
  const auto device = bound.LocatedPoints().GetValues();
  Kokkos::View<Real**, pcms::DeviceMemorySpace> copy("x", device.extent(0),
                                                     device.extent(1));
  pcms::detail::CopyDeviceRank2ViewToDeviceView(copy, device);
  const auto x =
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace{}, copy);
  std::vector<std::vector<Real>> points(x.extent(0));
  for (std::size_t p = 0; p < x.extent(0); ++p) {
    for (std::size_t d = 0; d < x.extent(1); ++d) {
      points[p].push_back(x(p, d));
    }
  }
  return points;
}

std::shared_ptr<const pcms::BoundPoints> BindPoints(
  const pcms::FunctionSpace& space,
  const std::vector<std::vector<Real>>& points)
{
  const auto device = pcms::test::DevicePoints(points);
  return space.GetGeometry()->Bind(
    pcms::CoordinateView<pcms::DeviceMemorySpace>(
      space.GetGeometry()->GetCoordinateSystem(),
      pcms::MakeConstRank2View(device)));
}

} // namespace

TEST_CASE("Product space: the layout's global ids and sizes",
          "[product][space]")
{
  Omega_h::Library lib;
  auto mesh = pcms::test::BuildSquareMesh(lib, 2);
  const auto plane = pcms::CreateLagrangeSpace(
    pcms::omega_h::CreateGeometry(mesh, pcms::detail::DeferredCartesian()), 1,
    1, {.backend = pcms::LagrangeBackend::OmegaH});
  const auto fourier = pcms::CreateFourierSpace(Periodic(4), {0, 1});
  const auto space = pcms::CreateProductSpace(plane, fourier);
  const auto& layout = *space->GetLayout();
  const auto na = plane->GetLayout()->GetNumDofHolder();
  REQUIRE(layout.GetNumDofHolder() == na * 3);
  CHECK(layout.GetNumGlobalDofHolder() == static_cast<pcms::GO>(na) * 3);
  CHECK(layout.GetNumComponents() == 1);
  CHECK(layout.GetDimension() == 3);
  const auto gids = layout.GetGidsHost();
  const auto gid_a = plane->GetLayout()->GetGidsHost();
  for (pcms::LO i = 0; i < na; ++i) {
    for (int k = 0; k < 3; ++k) {
      CHECK(gids(i * 3 + k) == gid_a(i) * 3 + k);
    }
  }
  CHECK_THROWS(pcms::CreateProductSpace(plane, nullptr));
  CHECK_THROWS(pcms::CreateProductSpace(
    plane, fourier,
    pcms::CreateProductGeometry(plane->GetGeometry(), Periodic(4))));
}

TEST_CASE("Product space: value and gradient of representable functions",
          "[product][space]")
{
  // Fourier x Fourier: u = cos(x) sin(2 z)... as DOFs a_i b_k
  {
    const Real period = 2.0;
    const auto xs = pcms::CreateFourierSpace(Periodic(4, period), {0, 1}, 1);
    const auto zs = pcms::CreateFourierSpace(Periodic(5), {0, 1, 2});
    const auto space = pcms::CreateProductSpace(xs, zs);
    const std::vector<Real> a = {0.5, 1.5, -0.5};
    const auto dofs = Outer(a, ZCoefficients);
    auto u = space->CreateFunction<Real>();
    SetDofs(u, dofs);
    const Real w = TwoPi / period;
    const auto fx = [&](Real x) {
      return a[0] + a[1] * std::cos(w * x) + a[2] * std::sin(w * x);
    };
    const auto dfx = [&](Real x) {
      return w * (-a[1] * std::sin(w * x) + a[2] * std::cos(w * x));
    };
    const auto points = Points(17, 2, -3.0, 7.0);
    const auto value = EvaluateAt(*space, u, points);
    const auto gradient =
      EvaluateAt(*space, u, points, pcms::DifferentialOperator::Gradient);
    for (std::size_t p = 0; p < points.size(); ++p) {
      const Real x = points[p][0];
      const Real z = points[p][1];
      CHECK(value[p][0] == Catch::Approx(fx(x) * Z(z)).margin(1e-11));
      REQUIRE(gradient[p].size() == 2);
      CHECK(gradient[p][0] == Catch::Approx(dfx(x) * Z(z)).margin(1e-10));
      CHECK(gradient[p][1] == Catch::Approx(fx(x) * DZ(z)).margin(1e-10));
    }
  }
  // piecewise linear x Fourier, and nested: (Lagrange x Fourier) x Fourier
  {
    const auto lagrange = pcms::CreateLagrangeSpace(Periodic(6), 1, 1);
    const auto zs = pcms::CreateFourierSpace(Periodic(5), {0, 1, 2});
    const auto space = pcms::CreateProductSpace(lagrange, zs);
    const std::vector<Real> a = {1, 4, -2, 0.5, 3, 7};
    auto u = space->CreateFunction<Real>();
    SetDofs(u, Outer(a, ZCoefficients));
    const Real h = TwoPi / 6;
    // at a node the Lagrange factor is its value; its slope is constant on a
    // cell
    const auto nodes = EvaluateAt(*space, u, {{2 * h, 1.1}, {5 * h, 0.2}});
    CHECK(nodes[0][0] == Catch::Approx(a[2] * Z(1.1)).margin(1e-12));
    CHECK(nodes[1][0] == Catch::Approx(a[5] * Z(0.2)).margin(1e-12));
    const auto slope = EvaluateAt(*space, u, {{2.25 * h, 0.7}},
                                  pcms::DifferentialOperator::Gradient);
    CHECK(slope[0][0] ==
          Catch::Approx((a[3] - a[2]) / h * Z(0.7)).margin(1e-11));
    CHECK(slope[0][1] ==
          Catch::Approx((0.75 * a[2] + 0.25 * a[3]) * DZ(0.7)).margin(1e-11));
    // nested: the third factor multiplies by 1 + 2 cos(y)
    const auto ys = pcms::CreateFourierSpace(Periodic(3), {0, 1});
    const auto nested = pcms::CreateProductSpace(space, ys);
    auto v = nested->CreateFunction<Real>();
    SetDofs(v, Outer(Outer(a, ZCoefficients), {1.0, 2.0, 0.0}));
    const auto n = EvaluateAt(*nested, v, {{2 * h, 1.1, 0.9}});
    CHECK(n[0][0] ==
          Catch::Approx(a[2] * Z(1.1) * (1 + 2 * std::cos(0.9))).margin(1e-11));
  }
  // Omega_h P1 x Fourier: u = (x + 2 y) f(z)
  {
    Omega_h::Library lib;
    auto mesh = pcms::test::BuildSquareMesh(lib, 3);
    const auto plane = pcms::CreateLagrangeSpace(
      pcms::omega_h::CreateGeometry(mesh, pcms::detail::DeferredCartesian()), 1,
      1, {.backend = pcms::LagrangeBackend::OmegaH});
    const auto zs = pcms::CreateFourierSpace(Periodic(5), {0, 1, 2});
    const auto space = pcms::CreateProductSpace(plane, zs);
    auto u = space->CreateFunction<Real>();
    SetDofs(u, Outer(NodalValues(*plane->GetLayout(),
                                 [](Real x, Real y) { return x + 2 * y; }),
                     ZCoefficients));
    const auto points = Points(13, 3, 0.05, 0.95);
    auto pts = points;
    for (auto& p : pts) {
      p[2] = (p[2] - 0.5) * 20;
    }
    const auto value = EvaluateAt(*space, u, pts);
    for (std::size_t p = 0; p < pts.size(); ++p) {
      CHECK(value[p][0] ==
            Catch::Approx((pts[p][0] + 2 * pts[p][1]) * Z(pts[p][2]))
              .margin(1e-11));
    }
    // the plane space has no gradient, so neither has the product
    CHECK_FALSE(space->Supports(pcms::DifferentialOperator::Gradient));
  }
}

TEST_CASE("Product space: <E x, y> = <x, E^T y> and the rows are E",
          "[product][adjoint]")
{
  const auto periodic = Periodic(5, 3.0);
  const auto fourier = pcms::CreateFourierSpace(Periodic(4), {0, 1, 3});
  pcms::test::CheckAdjointAndMatrix(
    pcms::CreateProductSpace(pcms::CreateLagrangeSpace(periodic, 1, 1),
                             fourier),
    Points(23, 2, -4.0, 8.0));
  pcms::test::CheckAdjointAndMatrix(
    pcms::CreateProductSpace(pcms::CreatePeriodicHermiteSpace(periodic),
                             pcms::CreatePeriodicHermiteSpace(Periodic(4))),
    Points(23, 2, -4.0, 8.0));
  Omega_h::Library lib;
  auto mesh = pcms::test::BuildSquareMesh(lib, 3);
  const auto plane = pcms::CreateLagrangeSpace(
    pcms::omega_h::CreateGeometry(mesh, pcms::detail::DeferredCartesian()), 1,
    1, {.backend = pcms::LagrangeBackend::OmegaH});
  auto pts = Points(29, 3, 0.02, 0.98);
  for (auto& p : pts) {
    p[2] = (p[2] - 0.5) * 20;
  }
  pcms::test::CheckAdjointAndMatrix(pcms::CreateProductSpace(plane, fourier),
                                    pts);
}

TEST_CASE("Product space: a spline factor evaluates; its product has no "
          "transpose",
          "[product][spline]")
{
  // f(x) = 1 + 2 x - x^3, which the spline reproduces, times f(z)
  const auto spline = pcms::CreateSplineSpace(Line(5, 0.0, 1.0));
  const auto zs = pcms::CreateFourierSpace(Periodic(5), {0, 1, 2});
  const auto space = pcms::CreateProductSpace(spline, zs);
  const auto f = [](Real x) { return 1 + 2 * x - x * x * x; };
  const auto df = [](Real x) { return 2 - 3 * x * x; };
  auto u = space->CreateFunction<Real>();
  SetDofs(u, Outer(NodalValues(*spline->GetLayout(),
                               [&](Real x, Real) { return f(x); }),
                   ZCoefficients));
  auto points = Points(19, 2, 0.0, 1.0);
  for (auto& p : points) {
    p[1] = (p[1] - 0.5) * 20;
  }
  const auto value = EvaluateAt(*space, u, points);
  const auto gradient =
    EvaluateAt(*space, u, points, pcms::DifferentialOperator::Gradient);
  for (std::size_t p = 0; p < points.size(); ++p) {
    const Real x = points[p][0];
    const Real z = points[p][1];
    CHECK(value[p][0] == Catch::Approx(f(x) * Z(z)).margin(1e-10));
    CHECK(gradient[p][0] == Catch::Approx(df(x) * Z(z)).margin(1e-9));
    CHECK(gradient[p][1] == Catch::Approx(f(x) * DZ(z)).margin(1e-10));
  }
  // at a tensor quadrature as well
  const auto quadrature = pcms::CreateGaussQuadrature(space->GetGeometry(), 4);
  const auto at = HostPoints(*quadrature.points);
  const auto tensor = EvaluateBound(*space, u, quadrature.points);
  for (std::size_t p = 0; p < at.size(); ++p) {
    CHECK(tensor[p][0] ==
          Catch::Approx(f(at[p][0]) * Z(at[p][1])).margin(1e-10));
  }
  const auto evaluator =
    space->CreatePointEvaluator<Real>(BindPoints(*space, points));
  CHECK_FALSE(evaluator->SupportsTranspose());
  CHECK(pcms::detail::AsRowsSource(*evaluator) == nullptr);
  CHECK_THROWS_AS(TransposeBound(*space, BindPoints(*space, points),
                                 std::vector<Real>(points.size(), 1.0)),
                  pcms::pcms_error);
  CHECK_THROWS_AS(TransposeBound(*space, quadrature.points,
                                 std::vector<Real>(at.size(), 1.0)),
                  pcms::pcms_error);
}

TEST_CASE("Product space: two evaluate-only factors evaluate at a tensor "
          "quadrature, not at scattered points",
          "[product][spline]")
{
  const auto xs = pcms::CreateSplineSpace(Line(4, 0.0, 1.0));
  const auto ys = pcms::CreateSplineSpace(Line(3, -1.0, 2.0));
  const auto space = pcms::CreateProductSpace(xs, ys);
  auto u = space->CreateFunction<Real>();
  SetDofs(u, Outer(NodalValues(*xs->GetLayout(),
                               [](Real x, Real) { return 1 + 2 * x; }),
                   NodalValues(*ys->GetLayout(),
                               [](Real y, Real) { return 3 - y * y; })));
  const auto quadrature = pcms::CreateGaussQuadrature(space->GetGeometry(), 4);
  const auto at = HostPoints(*quadrature.points);
  const auto value = EvaluateBound(*space, u, quadrature.points);
  const auto gradient = EvaluateBound(*space, u, quadrature.points,
                                      pcms::DifferentialOperator::Gradient);
  for (std::size_t p = 0; p < at.size(); ++p) {
    const Real x = at[p][0];
    const Real y = at[p][1];
    CHECK(value[p][0] ==
          Catch::Approx((1 + 2 * x) * (3 - y * y)).margin(1e-10));
    CHECK(gradient[p][0] == Catch::Approx(2 * (3 - y * y)).margin(1e-9));
    CHECK(gradient[p][1] == Catch::Approx((1 + 2 * x) * -2 * y).margin(1e-9));
  }
  CHECK_THROWS_WITH(
    EvaluateAt(*space, u, Points(3, 2, 0.0, 1.0)),
    Catch::Matchers::ContainsSubstring("the rows of all factors but one"));
}

TEST_CASE("Product space: three nested factors match their Kronecker rows",
          "[product][space]")
{
  const auto lagrange = pcms::CreateLagrangeSpace(Periodic(6), 1, 1);
  const auto fourier = pcms::CreateFourierSpace(Periodic(5, 3.0), {0, 1, 2});
  const auto hermite = pcms::CreatePeriodicHermiteSpace(Periodic(4, 2.0));
  const auto left = pcms::CreateProductSpace(
    pcms::CreateProductSpace(lagrange, fourier), hermite);
  const auto right = pcms::CreateProductSpace(
    lagrange, pcms::CreateProductSpace(fourier, hermite));
  const auto points = Points(37, 3, -4.0, 8.0);
  const auto device = pcms::test::DevicePoints(points);
  const auto reference =
    KroneckerReference({lagrange, fourier, hermite}, device, true);
  const auto num_points = static_cast<LO>(points.size());
  const auto x =
    pcms::test::Pseudo(static_cast<std::size_t>(reference.holders), 3);
  const auto y = pcms::test::Pseudo(points.size(), 4);
  for (const auto& space : {left, right}) {
    REQUIRE(space->GetLayout()->GetNumDofHolder() == reference.holders);
    auto u = space->CreateFunction<Real>();
    SetDofs(u, x);
    CheckClose(EvaluateAt(*space, u, points),
               ApplyRows(reference.value, x, num_points), 1e-12);
    CheckClose(
      EvaluateAt(*space, u, points, pcms::DifferentialOperator::Gradient),
      ApplyRows(reference.gradient, x, num_points), 1e-11);
    const auto transposed =
      TransposeBound(*space, BindPoints(*space, points), y);
    Kokkos::View<Real**, pcms::DeviceMemorySpace> total("total",
                                                        reference.holders, 1);
    pcms::detail::ApplyTranspose(reference.value, pcms::test::DevicePoints([&] {
                                   std::vector<std::vector<Real>> rows;
                                   for (const Real v : y) {
                                     rows.push_back({v});
                                   }
                                   return rows;
                                 }()),
                                 total);
    const auto expected =
      Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace{}, total);
    for (LO i = 0; i < reference.holders; ++i) {
      CHECK(transposed[i] == Catch::Approx(expected(i, 0)).margin(1e-12));
    }
    pcms::test::CheckAdjointAndMatrix(space, points);
  }
}

TEST_CASE("Product space: at a tensor quadrature, mode by mode as at the "
          "same points scattered",
          "[product][quadrature]")
{
  const auto lagrange = pcms::CreateLagrangeSpace(Periodic(6), 1, 1);
  const auto fourier = pcms::CreateFourierSpace(Periodic(5, 3.0), {0, 1, 2});
  const auto hermite = pcms::CreatePeriodicHermiteSpace(Periodic(4, 2.0));
  const auto spline = pcms::CreateSplineSpace(Line(4, 0.0, 1.0));
  for (const auto& space :
       {pcms::CreateProductSpace(pcms::CreateProductSpace(lagrange, fourier),
                                 hermite),
        pcms::CreateProductSpace(pcms::CreateProductSpace(fourier, spline),
                                 hermite)}) {
    const auto num_dofs =
      static_cast<std::size_t>(space->GetLayout()->GetNumDofHolder());
    auto u = space->CreateFunction<Real>();
    SetDofs(u, pcms::test::Pseudo(num_dofs, 5));
    const auto tensor =
      pcms::CreateGaussQuadrature(space->GetGeometry(), 3).points;
    const auto scattered = BindPoints(*space, HostPoints(*tensor));
    CheckClose(EvaluateBound(*space, u, tensor),
               EvaluateBound(*space, u, scattered), 1e-12);
    CheckClose(
      EvaluateBound(*space, u, tensor, pcms::DifferentialOperator::Gradient),
      EvaluateBound(*space, u, scattered, pcms::DifferentialOperator::Gradient),
      1e-11);
    const auto evaluator = space->CreatePointEvaluator<Real>(tensor);
    if (!evaluator->SupportsTranspose()) {
      continue;
    }
    const auto y =
      pcms::test::Pseudo(static_cast<std::size_t>(tensor->NumPoints()), 6);
    const auto a = TransposeBound(*space, tensor, y);
    const auto b = TransposeBound(*space, scattered, y);
    for (std::size_t i = 0; i < num_dofs; ++i) {
      CHECK(a[i] == Catch::Approx(b[i]).margin(1e-12));
    }
    // the lazy rows at the tensor points are the evaluation
    const auto* source = pcms::detail::AsRowsSource(*evaluator);
    REQUIRE(source != nullptr);
    const auto& rows = source->Rows();
    const auto offsets = Kokkos::create_mirror_view_and_copy(
      pcms::HostMemorySpace{}, rows.offsets);
    const auto holders = Kokkos::create_mirror_view_and_copy(
      pcms::HostMemorySpace{}, rows.holders);
    const auto values =
      Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace{}, rows.values);
    const auto dofs = pcms::test::GetDofs(u);
    const auto direct = EvaluateBound(*space, u, tensor);
    for (std::size_t p = 0; p + 1 < offsets.extent(0); ++p) {
      Real sum = 0;
      for (LO j = offsets(p); j < offsets(p + 1); ++j) {
        sum += values(j) * dofs[holders(j)];
      }
      CHECK(sum == Catch::Approx(direct[p][0]).margin(1e-12));
    }
  }
}

TEST_CASE("Product space: four factors in 5D on their tensor quadrature",
          "[product][quadrature]")
{
  pcms::UniformGrid<2> grid;
  grid.bot_left = {0.0, 0.0};
  grid.edge_length = {1.0, 2.0};
  grid.divisions = {3, 2};
  const auto plane = pcms::CreateLagrangeSpace(
    pcms::uniform_grid::CreateGeometry(grid, pcms::csys::CreateCartesian(2)), 1,
    1);
  const std::vector<std::shared_ptr<const pcms::FunctionSpace>> factors = {
    plane, pcms::CreateFourierSpace(Periodic(5), {0, 1, 2}),
    pcms::CreateLagrangeSpace(Periodic(4, 2.0), 1, 1),
    pcms::CreatePeriodicHermiteSpace(Periodic(3))};
  auto space = factors[0];
  for (std::size_t k = 1; k < factors.size(); ++k) {
    space = pcms::CreateProductSpace(space, factors[k]);
  }
  REQUIRE(space->GetGeometry()->GetCoordinateSystem()->Dimension() == 5);
  std::vector<std::vector<Real>> dofs;
  Real integral = 1;
  for (std::size_t k = 0; k < factors.size(); ++k) {
    const auto n =
      static_cast<std::size_t>(factors[k]->GetLayout()->GetNumDofHolder());
    dofs.push_back(pcms::test::Pseudo(n, static_cast<int>(k) + 7));
    auto f = factors[k]->CreateFunction<Real>();
    SetDofs(f, dofs.back());
    integral *= pcms::L2Integral(factors[k]).Apply(f);
  }
  auto u = space->CreateFunction<Real>();
  SetDofs(u, OuterAll(dofs));
  CHECK(pcms::L2Integral(space).Apply(u) ==
        Catch::Approx(integral).epsilon(1e-12).margin(1e-12));
  const auto& tensor = space->GetQuadrature().points;
  const auto scattered = BindPoints(*space, HostPoints(*tensor));
  CheckClose(EvaluateBound(*space, u, tensor),
             EvaluateBound(*space, u, scattered), 1e-12);
  CheckClose(
    EvaluateBound(*space, u, tensor, pcms::DifferentialOperator::Gradient),
    EvaluateBound(*space, u, scattered, pcms::DifferentialOperator::Gradient),
    1e-11);
}

TEST_CASE("Product space: Kronecker rows against factored evaluation",
          "[product][.bench]")
{
  using Clock = std::chrono::steady_clock;
  const auto seconds = [](Clock::time_point start) {
    Kokkos::fence();
    return std::chrono::duration<double>(Clock::now() - start).count();
  };
  for (int dim = 2; dim <= 6; ++dim) {
    std::vector<std::shared_ptr<const pcms::FunctionSpace>> leaves;
    auto space = pcms::CreateLagrangeSpace(Periodic(8), 1, 1);
    leaves.push_back(space);
    for (int k = 1; k < dim; ++k) {
      leaves.push_back(pcms::CreateLagrangeSpace(Periodic(8), 1, 1));
      space = pcms::CreateProductSpace(space, leaves.back());
    }
    auto u = space->CreateFunction<Real>();
    SetDofs(
      u, pcms::test::Pseudo(
           static_cast<std::size_t>(space->GetLayout()->GetNumDofHolder()), 1));
    for (const LO n : {10000, 100000, 1000000}) {
      Kokkos::View<Real**, pcms::DeviceMemorySpace> points("points", n, dim);
      Kokkos::parallel_for(
        "BenchPoints", Kokkos::RangePolicy<pcms::DefaultExecutionSpace>(0, n),
        KOKKOS_LAMBDA(const int p) {
          for (int d = 0; d < dim; ++d) {
            const Real s = 0.1234 + 0.6180339 * (p + 1) * (d + 1) + 0.31 * d;
            points(p, d) = TwoPi * (s - Kokkos::floor(s));
          }
        });
      Kokkos::View<Real**, pcms::DeviceMemorySpace> out("out", n, 1);
      const pcms::ValueView<Real, pcms::DeviceMemorySpace> values(
        u.GetData().GetValueBasis(), pcms::MakeRank2View(out));
      auto start = Clock::now();
      const auto reference = KroneckerReference(leaves, points, false);
      const double kron_setup = seconds(start);
      start = Clock::now();
      pcms::detail::Apply(reference.value, u.GetDOFHolderData().GetValues(),
                          out);
      const double kron_apply = seconds(start);
      start = Clock::now();
      const auto bound = space->GetGeometry()->Bind(
        pcms::CoordinateView<pcms::DeviceMemorySpace>(
          space->GetGeometry()->GetCoordinateSystem(),
          pcms::MakeConstRank2View(points)));
      const auto evaluator = space->CreatePointEvaluator<Real>(bound);
      const double factored_setup = seconds(start);
      start = Clock::now();
      evaluator->Evaluate(u, values);
      const double factored_apply = seconds(start);
      std::printf("bench %dD %7d points: kronecker setup %.4f s apply %.4f s "
                  "(%zu entries); factored setup %.4f s apply %.4f s\n",
                  dim, n, kron_setup, kron_apply,
                  reference.value.values.extent(0), factored_setup,
                  factored_apply);
    }
  }
}

TEST_CASE("Product space: the natural quadrature is the tensor product and "
          "the mass matrix is the Kronecker product of the factors'",
          "[product][quadrature]")
{
  const auto xs = pcms::CreateFourierSpace(Periodic(4, 2.0), {0, 1});
  const auto zs = pcms::CreateFourierSpace(Periodic(5), {0, 1, 2});
  const auto space = pcms::CreateProductSpace(xs, zs);
  const pcms::transfer::detail::BilinearForm mass(space);
  const std::vector<Real> mass_x = {2.0, 1.0, 1.0};
  const std::vector<Real> mass_z = {TwoPi, TwoPi / 2, TwoPi / 2, TwoPi / 2,
                                    TwoPi / 2};
  const std::size_t n = mass_x.size() * mass_z.size();
  for (std::size_t j = 0; j < n; ++j) {
    std::vector<Real> unit(n, 0.0);
    unit[j] = 1.0;
    auto e = space->CreateFunction<Real>();
    SetDofs(e, unit);
    auto column = space->CreateFunction<Real>();
    mass.Apply(e, column);
    const auto values = pcms::test::GetDofs(column);
    for (std::size_t i = 0; i < n; ++i) {
      const Real expected =
        i != j ? 0.0 : mass_x[i / mass_z.size()] * mass_z[i % mass_z.size()];
      CHECK(values[i] == Catch::Approx(expected).margin(1e-12));
    }
  }
}

#ifdef PCMS_ENABLE_OMEGA_H_MESHFIELDS
TEST_CASE("Product space: projection of an analytic source (matrix-free)",
          "[product][projection]")
{
  Omega_h::Library lib;
  auto mesh = pcms::test::BuildSquareMesh(lib, 3);
  const auto plane = pcms::CreateLagrangeSpace(
    pcms::omega_h::CreateGeometry(mesh, pcms::detail::DeferredCartesian()), 1,
    1, {.backend = pcms::LagrangeBackend::OmegaH});
  const auto periodic = Periodic(5);
  const auto geometry =
    pcms::CreateProductGeometry(plane->GetGeometry(), periodic);
  // a rule of the product of the plane's and 7 equispaced points in z, exact
  // for the source times any target below
  const auto rich = pcms::CreateProductSpace(
    plane, pcms::CreateFourierSpace(periodic, {0, 1, 2, 3}), geometry);
  for (const int modes : {1, 3}) {
    CAPTURE(modes);
    std::vector<int> list;
    for (int m = 0; m < modes; ++m) {
      list.push_back(m);
    }
    const auto space = pcms::CreateProductSpace(
      plane, pcms::CreateFourierSpace(periodic, list), geometry);
    pcms::detail::ProjectionOptions options;
    options.quadrature = rich->GetQuadrature();
    auto target = space->CreateFunction<Real>();
    const auto source = std::make_shared<SourceFactory>(geometry);
    pcms::detail::CreateProjection(source, space, options)
      ->Apply(space->CreateFunction<Real>(), target);
    auto pts = Points(15, 3, 0.05, 0.95);
    for (auto& p : pts) {
      p[2] = (p[2] - 0.5) * 20;
    }
    const auto value = EvaluateAt(*space, target, pts);
    for (std::size_t p = 0; p < pts.size(); ++p) {
      // modes = 3 reproduces the source; modes = 1 its average over z
      const Real f = modes == 3 ? Z(pts[p][2]) : 1.0;
      CHECK(value[p][0] ==
            Catch::Approx((pts[p][0] + 2 * pts[p][1]) * f).margin(1e-9));
    }
  }
}
#endif

TEST_CASE("Product space: twist-shift periodic axis wraps the plane's y into "
          "its interval and shifts z by the number of periods times a function "
          "of x",
          "[product][twist]")
{
  Omega_h::Library lib;
  auto mesh = pcms::test::BuildSquareMesh(lib, 3);
  const auto plane = pcms::CreateLagrangeSpace(
    pcms::omega_h::CreateGeometry(mesh, pcms::detail::DeferredCartesian()), 1,
    1, {.backend = pcms::LagrangeBackend::OmegaH});
  const auto zs = pcms::CreateFourierSpace(Periodic(5), {0, 1, 2});
  // the shift of z across y = 1: 0.3 + 0.8 cos(2 pi x), a function of x
  const auto xs = pcms::CreateFourierSpace(Periodic(4, 1.0), {0, 1});
  auto shift = xs->CreateFunction<Real>();
  SetDofs(shift, {0.3, 0.8, 0.0});
  const auto delta = [](Real x) { return 0.3 + 0.8 * std::cos(TwoPi * x); };
  const auto geometry = pcms::CreateProductGeometry(
    plane->GetGeometry(), zs->GetGeometry(),
    pcms::csys::CreateParameterSystem(
      "twisted", 3,
      {{1, 0.0, 1.0,
        pcms::CreateTwistShiftTransform(
          2, std::make_shared<const pcms::Function<Real>>(std::move(shift)),
          {0})},
       {2, 0.0, TwoPi}}));
  const auto space = pcms::CreateProductSpace(plane, zs, geometry);
  auto u = space->CreateFunction<Real>();
  SetDofs(u, Outer(NodalValues(*plane->GetLayout(),
                               [](Real x, Real y) { return x + 2 * y; }),
                   ZCoefficients));
  // (x, y + n, z) is (x, y, z + n delta(x)): u there is (x + 2 y) f(z + n
  // delta)
  for (const int n : {-2, -1, 0, 1, 3}) {
    CAPTURE(n);
    std::vector<std::vector<Real>> points;
    for (const Real x : {0.1, 0.45, 0.8}) {
      points.push_back({x, 0.37 + n, 1.9});
    }
    const auto value = EvaluateAt(*space, u, points);
    for (std::size_t p = 0; p < points.size(); ++p) {
      const Real x = points[p][0];
      CHECK(
        value[p][0] ==
        Catch::Approx((x + 2 * 0.37) * Z(1.9 + n * delta(x))).margin(1e-10));
    }
  }
  // without the periodic axis a point outside the plane is an error
  const auto plain = pcms::CreateProductSpace(plane, zs);
  CHECK_THROWS(
    EvaluateAt(*plain, plain->CreateFunction<Real>(), {{0.5, 1.37, 0.0}}));
  // the adjoint identity holds for points across the wrap
  auto pts = Points(31, 3, 0.02, 0.98);
  for (auto& p : pts) {
    p[1] = (p[1] - 0.5) * 6;
    p[2] = (p[2] - 0.5) * 20;
  }
  pcms::test::CheckAdjointAndMatrix(space, pts);
  // invalid periodic axes
  CHECK_THROWS(pcms::CreateTwistShiftTransform(2, nullptr, {0}));
  // the twist-shift of a second factor would shift the wrong product axes
  CHECK_THROWS(pcms::CreateProductGeometry(zs->GetGeometry(), geometry));
}

TEST_CASE("Product space: unwrapped Coordinates on a periodic axis evaluate "
          "through the factors; wrapped ones are refused",
          "[product][coordinates]")
{
  // u = (1 + 3 x)(2 + y) of period 1, unwrapped: nodes more than 1 / 2 apart
  const auto xs = pcms::CreateLagrangeSpace(Line(4, 0.0, 1.0), 1, 1);
  const auto ys = pcms::CreateLagrangeSpace(Line(3, 0.0, 1.0), 1, 1);
  const auto space = pcms::CreateProductSpace(xs, ys);
  auto basis = pcms::ValueBasis::Coordinates(
    pcms::csys::CreateParameterSystem("u", 1, {{0, 0.0, 1.0}}));
  const auto dofs =
    Outer(NodalValues(*xs->GetLayout(), [](Real x, Real) { return 1 + 3 * x; }),
          NodalValues(*ys->GetLayout(), [](Real y, Real) { return 2 + y; }));
  auto u = space->CreateFunction<Real>("u", basis);
  SetDofs(u, dofs);
  const auto points = Points(11, 2, 0.05, 0.95);
  const auto value = EvaluateAt(*space, u, points);
  for (std::size_t p = 0; p < points.size(); ++p) {
    CAPTURE(p);
    CHECK(
      value[p][0] ==
      Catch::Approx((1 + 3 * points[p][0]) * (2 + points[p][1])).margin(1e-12));
  }
  basis.wrapped = true;
  auto wrapped = space->CreateFunction<Real>("u", basis);
  SetDofs(wrapped, dofs);
  CHECK_THROWS_WITH(EvaluateAt(*space, wrapped, points),
                    Catch::Matchers::ContainsSubstring("wrapped Coordinates"));
}
