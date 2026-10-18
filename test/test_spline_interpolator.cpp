#include "mdspan/mdspan.hpp"
#include "pcms.h"
#include <Kokkos_Core.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>
#include "pcms/field/evaluator/spline_interpolator.hpp"

using namespace pcms;

using TestMemorySpace = Kokkos::DefaultExecutionSpace;

bool are_equal(double a, double b, double tolerance = 1e-7)
{
  return std::abs(a - b) < tolerance;
}

void createFlatGrid(Rank1View<double, TestMemorySpace> xvec,
                    Rank1View<double, TestMemorySpace> yvec,
                    Kokkos::View<double*, TestMemorySpace> X_flat,
                    Kokkos::View<double*, TestMemorySpace> Y_flat)
{
  size_t nx = xvec.size();
  size_t ny = yvec.size();

  Kokkos::parallel_for(
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ny, nx}),
    KOKKOS_LAMBDA(const int i, const int j) {
      size_t idx = i * nx + j;
      X_flat[idx] = xvec[j];
      Y_flat[idx] = yvec[i];
    });
}

void tset(int nth, Kokkos::View<double*, TestMemorySpace> th,
          Kokkos::View<double*, TestMemorySpace> sth,
          Kokkos::View<double*, TestMemorySpace> cth, double thmin,
          double thmax)
{
  Kokkos::parallel_for(
    Kokkos::RangePolicy<TestMemorySpace>(0, nth), KOKKOS_LAMBDA(const int ith) {
      th[ith] = thmin + static_cast<double>(ith) * (thmax - thmin) /
                          static_cast<double>(nth - 1);
      sth[ith] = 2.0 + std::sin(th[ith]);
      cth[ith] = std::cos(th[ith]);
    });
}

void xset(int nx, Kokkos::View<double*, TestMemorySpace> x,
          Kokkos::View<double*, TestMemorySpace> ex, double xmin, double xmax)
{
  if (nx < 2)
    return; // avoid division by zero

  Kokkos::parallel_for(
    Kokkos::RangePolicy<TestMemorySpace>(0, nx), KOKKOS_LAMBDA(const int ix) {
      x[ix] = xmin + static_cast<double>(ix) * (xmax - xmin) /
                       static_cast<double>(nx - 1);
      ex[ix] = std::exp(2.0 * x[ix] - 1.0);
    });
}

void ffset(int num, Kokkos::View<double*, TestMemorySpace> xf,
           Kokkos::View<double*, TestMemorySpace> tf,
           Kokkos::View<double*, TestMemorySpace> f)
{
  Kokkos::parallel_for(
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {num, num}),
    KOKKOS_LAMBDA(const int i, const int j) {
      f(i * num + j) = xf[i] * tf[j];
    });
}

void bset(Rank1View<double, TestMemorySpace> fx, int nx,
          Rank1View<double, TestMemorySpace> fth, int nth,
          Rank1View<double, TestMemorySpace> bcx1,
          Rank1View<double, TestMemorySpace> bcx2,
          Rank1View<double, TestMemorySpace> bcth1,
          Rank1View<double, TestMemorySpace> bcth2)
{
  Kokkos::parallel_for(
    Kokkos::RangePolicy<TestMemorySpace>(0, nth), KOKKOS_LAMBDA(const int ith) {
      bcx1[ith] = 2.0 * fx(0) * fth(ith);      // df/dx at x(1)
      bcx2[ith] = 2.0 * fx(nx - 1) * fth(ith); // df/dx at x(nx)
    });

  Kokkos::parallel_for(
    Kokkos::RangePolicy<TestMemorySpace>(0, nx), KOKKOS_LAMBDA(const int ix) {
      bcth1(ix) = fx(ix); // df/dth at th = 0 (th[0])
      bcth2(ix) = fx(ix); // df/dth at th = 2π (th[nth - 1])
    });
}

void generate_gt_2d(Rank1View<double, TestMemorySpace> gt_splinv,
                    Rank1View<double, TestMemorySpace> fxtest,
                    Rank1View<double, TestMemorySpace> fthtest, int ntest)
{
  Kokkos::parallel_for(
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ntest, ntest}),
    KOKKOS_LAMBDA(const int j, const int i) {
      gt_splinv(j * ntest + i) = fxtest(i) * fthtest(j);
    });
}

void compare(Rank2View<double, TestMemorySpace> splinv,
             Rank1View<double, TestMemorySpace> gt_splinv,
             Rank1View<double, HostMemorySpace> res, std::string label)
{
  // Compare the output of splinv with gt_splinv
  double result_fdif = 0.0;
  double result_fdifr = 0.0;
  Kokkos::parallel_reduce(
    "compare_splinv", splinv.extent(0),
    KOKKOS_LAMBDA(const int i, double& max_fdif, double& max_fdifr) {
      if (splinv(i, 0) != 0.0) {
        double dif = Kokkos::abs(splinv(i, 0) - gt_splinv(i));
        max_fdif = Kokkos::max(max_fdif, dif);
        max_fdifr = Kokkos::max(max_fdifr, dif / gt_splinv(i));
      }
    },
    Kokkos::Max<double>(result_fdif), Kokkos::Max<double>(result_fdifr));
  res(0) = result_fdif;
  res(1) = result_fdifr;
  std::cout << "2d " << label << " max absolute difference: " << res(0)
            << ", relative difference: " << res(1) << std::endl;
}

void compare_2d(Rank2View<double, TestMemorySpace> splinv,
                Rank1View<double, TestMemorySpace> gt_splinv,
                Rank1View<double, HostMemorySpace> res, std::string label)
{
  // Compare the output of splinv with gt_splinv
  double result_fdif = 0.0;
  double result_fdifr = 0.0;
  Kokkos::parallel_reduce(
    "compare_splinv", splinv.extent(0),
    KOKKOS_LAMBDA(const int i, double& max_fdif, double& max_fdifr) {
      if (splinv(i, 0) != 0.0) {
        double dif = Kokkos::abs(splinv(i, 0) - gt_splinv(i));
        max_fdif = Kokkos::max(max_fdif, dif);
        max_fdifr =
          Kokkos::max(max_fdifr, dif / (0.5 * (splinv(i, 0) + gt_splinv(i))));
      }
    },
    Kokkos::Max<double>(result_fdif), Kokkos::Max<double>(result_fdifr));
  res(0) = result_fdif;
  res(1) = result_fdifr;
  std::cout << "2d " << label << " max absolute difference: " << res(0)
            << ", relative difference: " << res(1) << std::endl;
}

void reset_2dspan(Rank2View<double, TestMemorySpace> span)
{
  Kokkos::parallel_for(
    "zero_init",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0},
                                           {span.extent(0), span.extent(1)}),
    KOKKOS_LAMBDA(int i, int j) { span(i, j) = 0.0; });
}

void dotest1(int ns, Rank1View<double, TestMemorySpace> x,
             Rank1View<double, TestMemorySpace> f, int nt,
             Rank1View<double, TestMemorySpace> xt,
             Rank1View<double, TestMemorySpace> ft,
             Kokkos::View<double*, HostMemorySpace> res_1d)
{
  using std::data;
  using std::size;

  Kokkos::View<double*, TestMemorySpace> splinv_view(
    "splinv_view", static_cast<std::size_t>(3) * nt);
  auto splinv = Rank2View<double, TestMemorySpace>(splinv_view.data(), nt, 3);
  reset_2dspan(splinv);

  // interpolator.cspline(x, ns, fspl, 1, 1, 1, 1, wk);
  ExplicitCubicSplineInterpolator<double, TestMemorySpace>
    explicit_interpolator(x, f, BoundaryCondition::FIRST_DERIVATIVE_MATCH, 1,
                          BoundaryCondition::FIRST_DERIVATIVE_MATCH, 1);
  explicit_interpolator.evaluate(xt, splinv);

  auto res1 = Rank1View<double, HostMemorySpace>(res_1d.data(), 2);
  compare(splinv, ft, res1, "cspline");
  reset_2dspan(splinv);

  ExplicitCubicSplineInterpolator<double, TestMemorySpace>
    explicit_interpolator_periodic(x, f, BoundaryCondition::PERIODIC, 0,
                                   BoundaryCondition::PERIODIC, 0);
  explicit_interpolator_periodic.evaluate(xt, splinv);

  auto res2 = Rank1View<double, HostMemorySpace>(res_1d.data() + 2, 2);
  compare(splinv, ft, res2, "cspline_periodic");
  reset_2dspan(splinv);

  CompactCubicSplineInterpolator<double, TestMemorySpace> compact_interpolator(
    x, f, BoundaryCondition::FIRST_DERIVATIVE_MATCH, 1,
    BoundaryCondition::FIRST_DERIVATIVE_MATCH, 1);
  compact_interpolator.evaluate(xt, splinv);

  auto res3 = Rank1View<double, HostMemorySpace>(res_1d.data() + 4, 2);
  compare(splinv, ft, res3, "compact_cubic");
}

void pspltest1(Kokkos::View<double*, HostMemorySpace> res_1d)
{
  const double pi2 = 6.28318530718;
  const double zero = 0.0;
  int inum = 10;

  // Local arrays
  Kokkos::View<double*, TestMemorySpace> zdum_view("zdum_view", 1000);
  Kokkos::View<double*, TestMemorySpace> xtest_view("xtest_view", 1000);
  Kokkos::View<double*, TestMemorySpace> ftest_view("ftest_view", 1000);
  Kokkos::View<double*, TestMemorySpace> x_view("x_view", 10);
  Kokkos::View<double*, TestMemorySpace> zcos_view("zcos_view", 10);
  Kokkos::View<double*, TestMemorySpace> z2sin_view("z2sin_view", 10);

  // Prepare test data
  tset(1000, xtest_view, ftest_view, zdum_view, zero - 0.1, pi2 + 0.1);
  tset(inum, x_view, z2sin_view, zcos_view, zero, pi2);

  auto x = Rank1View<double, TestMemorySpace>(x_view.data(), x_view.size());

  auto z2sin =
    Rank1View<double, TestMemorySpace>(z2sin_view.data(), z2sin_view.size());

  auto xtest = Rank1View<double, TestMemorySpace>(xtest_view.data(), 1000);
  auto ftest = Rank1View<double, TestMemorySpace>(ftest_view.data(), 1000);

  // Call test function
  dotest1(inum, x, z2sin, 1000, xtest, ftest, res_1d);
}

void pspltest1_long_periodic(Kokkos::View<double*, HostMemorySpace> res_1d)
{
  const double pi2 = 6.28318530718;
  const double zero = 0.0;
  int inum = 16;

  Kokkos::View<double*, TestMemorySpace> zdum_view("zdum_view_long", 1000);
  Kokkos::View<double*, TestMemorySpace> xtest_view("xtest_view_long", 1000);
  Kokkos::View<double*, TestMemorySpace> ftest_view("ftest_view_long", 1000);
  Kokkos::View<double*, TestMemorySpace> x_view("x_view_long", inum);
  Kokkos::View<double*, TestMemorySpace> zcos_view("zcos_view_long", inum);
  Kokkos::View<double*, TestMemorySpace> z2sin_view("z2sin_view_long", inum);

  tset(1000, xtest_view, ftest_view, zdum_view, zero - 0.1, pi2 + 0.1);
  tset(inum, x_view, z2sin_view, zcos_view, zero, pi2);

  auto x = Rank1View<double, TestMemorySpace>(x_view.data(), x_view.size());
  auto z2sin =
    Rank1View<double, TestMemorySpace>(z2sin_view.data(), z2sin_view.size());
  auto xtest = Rank1View<double, TestMemorySpace>(xtest_view.data(), 1000);
  auto ftest = Rank1View<double, TestMemorySpace>(ftest_view.data(), 1000);

  dotest1(inum, x, z2sin, 1000, xtest, ftest, res_1d);
}

void dotest2(Rank1View<double, TestMemorySpace> x,
             Rank1View<double, TestMemorySpace> fx, int nx,
             Rank1View<double, TestMemorySpace> th,
             Rank1View<double, TestMemorySpace> fth,
             Rank1View<double, TestMemorySpace> dfth, int nth,
             Rank4View<double, TestMemorySpace> f,
             Rank1View<double, TestMemorySpace> bcx1,
             Rank1View<double, TestMemorySpace> bcx2,
             Rank1View<double, TestMemorySpace> bcth1,
             Rank1View<double, TestMemorySpace> bcth2,
             Rank1View<double, TestMemorySpace> xtest,
             Rank1View<double, TestMemorySpace> fxtest,
             Rank1View<double, TestMemorySpace> thtest,
             Rank1View<double, TestMemorySpace> fthtest, int ntest,
             Kokkos::View<double*, HostMemorySpace> res_2d)
{

  Kokkos::View<double*, TestMemorySpace> values_view(
    "values_view", static_cast<std::size_t>(nth) * nx);
  Kokkos::parallel_for(
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nth, nx}),
    KOKKOS_LAMBDA(const int ith, const int ix) {
      values_view(ix * nth + ith) = f(0, 0, ix, ith);
    });
  auto values =
    Rank1View<double, TestMemorySpace>(values_view.data(), nth * nx);

  bset(fx, nx, fth, nth, bcx1, bcx2, bcth1, bcth2);
  Kokkos::View<double*, TestMemorySpace> splinv_view("splinv_view", 400000);
  auto splinv =
    Rank2View<double, TestMemorySpace>(splinv_view.data(), 40000, 10);
  reset_2dspan(splinv);

  Kokkos::View<double*, TestMemorySpace> xtest_grid_view("xtest_grid_view",
                                                         40000);
  Kokkos::View<double*, TestMemorySpace> thtest_grid_view("thtest_grid_view",
                                                          40000);
  createFlatGrid(xtest, thtest, xtest_grid_view, thtest_grid_view);
  auto xtest_grid =
    Rank1View<double, TestMemorySpace>(xtest_grid_view.data(), 40000);
  auto thtest_grid =
    Rank1View<double, TestMemorySpace>(thtest_grid_view.data(), 40000);
  Kokkos::View<double*, TestMemorySpace> splinv_gt_view("splinv_view", 40000);
  auto splinv_gt =
    Rank1View<double, TestMemorySpace>(splinv_gt_view.data(), 40000);
  generate_gt_2d(splinv_gt, fxtest, fthtest, ntest);

  ExplicitBiCubicSplineInterpolator<double, TestMemorySpace>
    explicit_interpolator(x, th, values,
                          BoundaryCondition::FIRST_DERIVATIVE_MATCH, bcx1,
                          BoundaryCondition::FIRST_DERIVATIVE_MATCH, bcx2,
                          BoundaryCondition::FIRST_DERIVATIVE_MATCH, bcth1,
                          BoundaryCondition::FIRST_DERIVATIVE_MATCH, bcth2);
  explicit_interpolator.evaluate(xtest_grid, thtest_grid, splinv);

  auto res1 = Rank1View<double, HostMemorySpace>(res_2d.data(), 2);
  compare_2d(splinv, splinv_gt, res1, "bcspline");
  reset_2dspan(splinv);

  CompactBiCubicSplineInterpolator<double, TestMemorySpace>
    compact_interpolator(x, th, values,
                         BoundaryCondition::FIRST_DERIVATIVE_MATCH, bcx1,
                         BoundaryCondition::FIRST_DERIVATIVE_MATCH, bcx2,
                         BoundaryCondition::FIRST_DERIVATIVE_MATCH, bcth1,
                         BoundaryCondition::FIRST_DERIVATIVE_MATCH, bcth2);
  compact_interpolator.evaluate(xtest_grid, thtest_grid, splinv);

  auto res2 = Rank1View<double, HostMemorySpace>(res_2d.data() + 2, 2);
  compare_2d(splinv, splinv_gt, res2, "mkbicub");

  // CompactBiCubicSplineInterpolator<double, TestMemorySpace>
  //     test_interpolator(x, th, values);
}

void pspltest2(Kokkos::View<double*, HostMemorySpace> res_2d)
{
  const double pi2 = 6.28318530718;
  const double zero = 0.0;
  const double one = 1.0;

  Kokkos::View<double*, TestMemorySpace> x1_view("x1_view", 10);
  Kokkos::View<double*, TestMemorySpace> ex1_view("ex1_view", 10);
  Kokkos::View<double*, TestMemorySpace> t1_view("t1_view", 10);
  Kokkos::View<double*, TestMemorySpace> st1_view("st1_view", 10);
  Kokkos::View<double*, TestMemorySpace> ct1_view("ct1_view", 10);
  Kokkos::View<double*, TestMemorySpace> xtest_view("xtest_view", 200);
  Kokkos::View<double*, TestMemorySpace> extest_view("extest_view", 200);
  Kokkos::View<double*, TestMemorySpace> ttest_view("ttest_view", 200);
  Kokkos::View<double*, TestMemorySpace> stest_view("stest_view", 200);
  Kokkos::View<double*, TestMemorySpace> ctest_view("ctest_view", 200);
  Kokkos::View<double*, TestMemorySpace> f1_view("f1_view", 1600);
  Kokkos::View<double*, TestMemorySpace> bcx1_view("bcx1_view", 40);
  Kokkos::View<double*, TestMemorySpace> bcx2_view("bcx2_view", 40);
  Kokkos::View<double*, TestMemorySpace> bcth1_view("bcth1_view", 40);
  Kokkos::View<double*, TestMemorySpace> bcth2_view("bcth2_view", 40);

  xset(10, x1_view, ex1_view, zero, one);
  xset(200, xtest_view, extest_view, zero, one);

  tset(10, t1_view, st1_view, ct1_view, zero, pi2);
  tset(200, ttest_view, stest_view, ctest_view, zero, pi2);

  ffset(10, ex1_view, st1_view, f1_view);

  auto x1 = Rank1View<double, TestMemorySpace>(x1_view.data(), x1_view.size());
  auto ex1 =
    Rank1View<double, TestMemorySpace>(ex1_view.data(), ex1_view.size());
  auto t1 = Rank1View<double, TestMemorySpace>(t1_view.data(), t1_view.size());
  auto st1 =
    Rank1View<double, TestMemorySpace>(st1_view.data(), st1_view.size());
  auto ct1 =
    Rank1View<double, TestMemorySpace>(ct1_view.data(), ct1_view.size());
  auto f1 = Rank4View<double, TestMemorySpace>(f1_view.data(), 4, 4, 10, 10);
  auto bcx1 =
    Rank1View<double, TestMemorySpace>(bcx1_view.data(), bcx1_view.size());
  auto bcx2 =
    Rank1View<double, TestMemorySpace>(bcx2_view.data(), bcx2_view.size());
  auto bcth1 =
    Rank1View<double, TestMemorySpace>(bcth1_view.data(), bcth1_view.size());
  auto bcth2 =
    Rank1View<double, TestMemorySpace>(bcth2_view.data(), bcth2_view.size());

  auto xtest =
    Rank1View<double, TestMemorySpace>(xtest_view.data(), xtest_view.size());
  auto extest =
    Rank1View<double, TestMemorySpace>(extest_view.data(), extest_view.size());
  auto ttest =
    Rank1View<double, TestMemorySpace>(ttest_view.data(), ttest_view.size());
  auto stest =
    Rank1View<double, TestMemorySpace>(stest_view.data(), stest_view.size());

  dotest2(x1, ex1, 10, t1, st1, ct1, 10, f1, bcx1, bcx2, bcth1, bcth2, xtest,
          extest, ttest, stest, 200, res_2d);
}

TEST_CASE("test_cubic_spline_interpolator")
{
  Kokkos::print_configuration(std::cout);
  std::cout << "DefaultExecutionSpace: "
            << Kokkos::DefaultExecutionSpace::name() << std::endl;
  Kokkos::Timer timer;
  {
    Kokkos::View<double*, HostMemorySpace> res_1d("res_1d", 6);
    double gt_1d[6] = {6.7572E-04, 6.6529E-04, 6.8669E-04,
                       6.7622E-04, 6.7572E-04, 6.6529E-04};
    pspltest1(res_1d);
    for (int i = 0; i < 6; ++i) {
      REQUIRE(are_equal(res_1d(i), gt_1d[i]));
    }

    Kokkos::View<double*, HostMemorySpace> res_1d_long("res_1d_long", 6);
    pspltest1_long_periodic(res_1d_long);
    REQUIRE(res_1d_long(2) < 1.0e-3);
    REQUIRE(res_1d_long(3) < 1.0e-3);
    REQUIRE(res_1d_long(4) < 1.0e-3);
    REQUIRE(res_1d_long(5) < 1.0e-3);

    Kokkos::View<double*, HostMemorySpace> res_2d("res_2d", 4);
    double gt_2d[4] = {1.8312E-03, 6.7151E-04, 1.8312E-03, 6.7151E-04};
    pspltest2(res_2d);
    for (int i = 0; i < 4; ++i) {
      REQUIRE(are_equal(res_2d(i), gt_2d[i]));
    }
  }

  double time = timer.seconds();
  std::cout << "Total time for spline tests: " << time << " seconds"
            << std::endl;
}

namespace
{

Kokkos::View<double*, TestMemorySpace> to_device(const std::vector<double>& h)
{
  Kokkos::View<double*, TestMemorySpace> d("to_device", h.size());
  auto m = Kokkos::create_mirror_view(d);
  for (size_t i = 0; i < h.size(); ++i)
    m(i) = h[i];
  Kokkos::deep_copy(d, m);
  return d;
}

Rank1View<double, TestMemorySpace> span_of(
  const Kokkos::View<double*, TestMemorySpace>& v)
{
  return Rank1View<double, TestMemorySpace>(v.data(), v.extent(0));
}

Kokkos::View<LO*, TestMemorySpace> selector_of(const std::vector<LO>& h)
{
  Kokkos::View<LO*, TestMemorySpace> d("selector", h.size());
  auto m = Kokkos::create_mirror_view(d);
  for (size_t i = 0; i < h.size(); ++i)
    m(i) = h[i];
  Kokkos::deep_copy(d, m);
  return d;
}

/// Evaluate a 1D interpolator with a selector and return the n x ncol output
template <typename Interpolator>
std::vector<double> eval_1d(Interpolator& interp, const std::vector<LO>& sel,
                            const std::vector<double>& xq, int ncol)
{
  auto xd = to_device(xq);
  Kokkos::View<double*, TestMemorySpace> out("out", xq.size() * ncol);
  interp.evaluate(
    selector_of(sel), span_of(xd),
    Rank2View<double, TestMemorySpace>(out.data(), xq.size(), ncol));
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  return std::vector<double>(h.data(), h.data() + h.extent(0));
}

/// Evaluate a 2D interpolator with a selector and return the n x ncol output
template <typename Interpolator>
std::vector<double> eval_2d(Interpolator& interp, const std::vector<LO>& sel,
                            const std::vector<double>& xq,
                            const std::vector<double>& yq, int ncol)
{
  auto xd = to_device(xq);
  auto yd = to_device(yq);
  Kokkos::View<double*, TestMemorySpace> out("out", xq.size() * ncol);
  interp.evaluate(
    selector_of(sel), span_of(xd), span_of(yd),
    Rank2View<double, TestMemorySpace>(out.data(), xq.size(), ncol));
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  return std::vector<double>(h.data(), h.data() + h.extent(0));
}

/// Uniform grid of n points on [a, b]
std::vector<double> linspace(double a, double b, int n)
{
  std::vector<double> x(n);
  for (int i = 0; i < n; ++i)
    x[i] = a + (b - a) * i / (n - 1);
  return x;
}

/// Sample f on the tensor grid with x as the slow index
template <typename F>
std::vector<double> sample_2d(const std::vector<double>& x,
                              const std::vector<double>& y, F f)
{
  std::vector<double> v;
  for (double xi : x)
    for (double yj : y)
      v.push_back(f(xi, yj));
  return v;
}

} // namespace

TEMPLATE_TEST_CASE("two-point splines keep both endpoint derivatives",
                   "[spline]",
                   (ExplicitCubicSplineInterpolator<double, TestMemorySpace>),
                   (CompactCubicSplineInterpolator<double, TestMemorySpace>))
{
  // f = x^3 on [1, 3]: any pair of first/second derivative end conditions
  // determines the cubic exactly
  auto x = to_device({1.0, 3.0});
  auto f = to_device({1.0, 27.0});
  const auto d1 = BoundaryCondition::FIRST_DERIVATIVE_MATCH;
  const auto d2 = BoundaryCondition::SECOND_DERIVATIVE_MATCH;
  auto [lo, lo_data, hi, hi_data] = GENERATE_COPY(
    std::make_tuple(d1, 3.0, d1, 27.0), std::make_tuple(d1, 3.0, d2, 18.0),
    std::make_tuple(d2, 6.0, d1, 27.0), std::make_tuple(d2, 6.0, d2, 18.0));
  TestType interp(span_of(x), span_of(f), lo, lo_data, hi, hi_data);
  auto out = eval_1d(interp, {1, 1, 1}, {1.5, 2.0, 2.5}, 3);
  for (int i = 0; i < 3; ++i) {
    const double xi = 1.5 + 0.5 * i;
    CHECK(out[3 * i] == Catch::Approx(xi * xi * xi));
    CHECK(out[3 * i + 1] == Catch::Approx(3 * xi * xi));
    CHECK(out[3 * i + 2] == Catch::Approx(6 * xi));
  }
}

TEMPLATE_TEST_CASE("non-zero boundary data on the lower side only is applied",
                   "[spline]",
                   (ExplicitBiCubicSplineInterpolator<double, TestMemorySpace>),
                   (CompactBiCubicSplineInterpolator<double, TestMemorySpace>))
{
  // f = g(x) (y - y^2/2) has df/dy = g(x) at y = 0 and 0 at y = 1
  auto g = [](double x) { return 1 + x + x * x + x * x * x; };
  auto f = [&](double x, double y) { return g(x) * (y - 0.5 * y * y); };
  auto xh = linspace(0, 1, 5);
  auto yh = linspace(0, 1, 5);
  std::vector<double> lo_h;
  for (double xi : xh)
    lo_h.push_back(g(xi));
  auto x = to_device(xh);
  auto y = to_device(yh);
  auto v = to_device(sample_2d(xh, yh, f));
  auto lo = to_device(lo_h);
  auto hi = to_device(std::vector<double>(xh.size(), 0.0));
  auto none = to_device({});
  const auto nak = BoundaryCondition::NOT_A_KNOT;
  const auto d1 = BoundaryCondition::FIRST_DERIVATIVE_MATCH;
  TestType interp(span_of(x), span_of(y), span_of(v), nak, span_of(none), nak,
                  span_of(none), d1, span_of(lo), d1, span_of(hi));
  std::vector<double> xq = {0.1, 0.45, 0.8};
  std::vector<double> yq = {0.05, 0.6, 0.95};
  auto out = eval_2d(interp, {1, 0, 0, 0, 0, 0}, xq, yq, 1);
  for (size_t i = 0; i < xq.size(); ++i)
    CHECK(out[i] == Catch::Approx(f(xq[i], yq[i])));
}

TEST_CASE("spline grids must be finite, ascending and non-degenerate",
          "[spline]")
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  auto grid = GENERATE_COPY(std::vector<double>{1.0},
                            std::vector<double>{1.0, 1.0, 1.0, 1.0},
                            std::vector<double>{3.0, 2.0, 1.0, 0.0},
                            std::vector<double>{0.0, nan, 2.0, 3.0},
                            std::vector<double>{0.0, 1.0, 2.0, inf});
  auto x = to_device(grid);
  auto f = to_device(std::vector<double>(grid.size(), 1.0));
  using Interp = CompactCubicSplineInterpolator<double, TestMemorySpace>;
  const auto nak = BoundaryCondition::NOT_A_KNOT;
  REQUIRE_THROWS_AS(Interp(span_of(x), span_of(f), nak, 0, nak, 0), pcms_error);
}
