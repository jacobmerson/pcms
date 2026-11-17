#include "fusionio/utilities.hpp"
#include <Kokkos_Core.hpp>
#include <pcms/field/coordinate_view.hpp>
#include <pcms/field/function_space.h>
#include <pcms/field/query_points.hpp>
#include <pcms/field/value_view.hpp>
#include <pcms/utility/arrays.h>
#include <pcms/utility/memory_spaces.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>

namespace fusionio
{

namespace
{

constexpr Real TwoPi = 2 * std::numbers::pi;
constexpr Real NaN = std::numeric_limits<Real>::quiet_NaN();
// 1.x's iteration limit of fio_find_val, fio_find_max and find_val_on_line
constexpr int MaxIterations = 100;

/// Values (and, if asked, coordinate derivatives d/dR, d/dphi, d/dZ) of a
/// function at batches of host points; points outside the domain get NaN.
class Probe
{
public:
  Probe(const pcms::Function<Real>& function, bool gradient)
    : function_(function),
      query_(function.GetSpace().GetGeometry(),
             {pcms::OutOfBoundsMode::FILL, NaN}),
      gradient_(gradient),
      num_components_(function.GetLayout().GetNumComponents())
  {
  }

  /// Evaluates at `x`; results stay valid until the next call.
  void At(std::span<const Point> x)
  {
    const size_t n = x.size();
    values_.assign(n * num_components_, NaN);
    gradients_.assign(n, {NaN, NaN, NaN});
    if (n == 0) {
      return;
    }
    Kokkos::View<Real**, pcms::HostMemorySpace> host("points", n, 3);
    for (size_t k = 0; k < n; ++k) {
      for (int c = 0; c < 3; ++c) {
        host(k, c) = x[k][c];
      }
    }
    // the device's own layout, which CoordinateView takes
    Kokkos::View<Real**, pcms::DeviceMemorySpace> points("points", n, 3);
    pcms::DeepCopyMismatchLayouts(points, host);
    const auto& geometry = function_.GetSpace().GetGeometry();
    query_.Locate(pcms::CoordinateView<pcms::DeviceMemorySpace>(
      geometry->GetCoordinateSystem(), pcms::MakeRank2View(points)));
    const auto stored = function_.GetData().GetValueBasis();
    Kokkos::View<Real**, pcms::DeviceMemorySpace> values("values", n,
                                                         num_components_);
    query_.Evaluate(function_, pcms::ValueView<Real, pcms::DeviceMemorySpace>(
                                 stored, pcms::MakeRank2View(values)));
    const auto host_values =
      Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), values);
    for (size_t k = 0; k < n; ++k) {
      for (size_t c = 0; c < num_components_; ++c) {
        values_[k * num_components_ + c] = host_values(k, c);
      }
    }
    if (!gradient_) {
      return;
    }
    Kokkos::View<Real**, pcms::DeviceMemorySpace> gradients("gradients", n, 3);
    query_.Evaluate(
      function_,
      pcms::ValueView<Real, pcms::DeviceMemorySpace>(
        pcms::GradientBasis(stored, geometry->GetCoordinateSystem()),
        pcms::MakeRank2View(gradients)),
      pcms::DifferentialOperator::Gradient);
    const auto host_gradients =
      Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace(), gradients);
    for (size_t k = 0; k < n; ++k) {
      // physical components to coordinate derivatives, as 1.x's eval_deriv
      gradients_[k] = {host_gradients(k, 0), x[k][0] * host_gradients(k, 1),
                       host_gradients(k, 2)};
    }
  }

  [[nodiscard]] bool Found(size_t k) const
  {
    return !std::isnan(values_[k * num_components_]);
  }
  [[nodiscard]] Real Value(size_t k, size_t c = 0) const
  {
    return values_[k * num_components_ + c];
  }
  [[nodiscard]] const Point& Gradient(size_t k) const { return gradients_[k]; }

private:
  const pcms::Function<Real>& function_;
  pcms::QueryPoints query_;
  bool gradient_;
  size_t num_components_;
  std::vector<Real> values_;
  std::vector<Point> gradients_;
};

void RequireScalar(const pcms::Function<Real>& function, const char* name)
{
  if (function.GetLayout().GetNumComponents() != 1) {
    throw pcms::pcms_error(std::string("fusionio::") + name +
                           ": the function is not a scalar");
  }
}

Real Dot(const Point& a, const Point& b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Point Unit(const Point& v)
{
  const Real d = std::sqrt(Dot(v, v));
  return {v[0] / d, v[1] / d, v[2] / d};
}

Real Norm(const Point& v)
{
  return std::sqrt(Dot(v, v));
}

/// x[k] for k in `which`, into `out`.
std::span<const Point> Gather(std::span<const Point> x,
                              const std::vector<size_t>& which,
                              std::vector<Point>& out)
{
  out.clear();
  for (const size_t k : which) {
    out.push_back(x[k]);
  }
  return out;
}

void WrapPhi(Point& x)
{
  if (x[1] < 0.) {
    x[1] += TwoPi;
  }
  if (x[1] >= TwoPi) {
    x[1] -= TwoPi;
  }
}

} // namespace

namespace
{

std::vector<SearchResult> FindMax(Probe& probe, std::span<Real> val,
                                  std::span<Point> x, Real tol, Real max_step,
                                  int dim, const Point& norm)
{
  // 1.x's step of the finite difference of |grad f|
  constexpr Real Delta = 1e-5;
  const Point n = Unit(norm);
  const auto restrict = [&](Point dv) {
    for (int c = 0; c < 3; ++c) {
      dv[c] *= dim == 1 ? n[c] : (dim == 2 ? 1. - n[c] : 1.);
    }
    return dv;
  };
  const size_t num = x.size();
  std::vector<SearchResult> result(num, SearchResult::Diverged);
  const std::vector<Point> x0(x.begin(), x.end());
  std::vector<Point> last_x(num), dx(num), dv(num);
  std::vector<size_t> active(num);
  std::iota(active.begin(), active.end(), 0);
  std::vector<size_t> next, stepping;
  std::vector<Point> at;
  for (int i = 0; i < MaxIterations && !active.empty(); ++i) {
    probe.At(Gather(x, active, at));
    next.clear();
    stepping.clear();
    std::vector<Point> at1;
    for (size_t j = 0; j < active.size(); ++j) {
      const size_t k = active[j];
      if (!probe.Found(j)) {
        if (i == 0) {
          result[k] = SearchResult::OutOfBounds;
          continue;
        }
        // moved out of bounds: halve the step and try again
        for (int c = 0; c < 3; ++c) {
          dx[k][c] /= 2.;
          x[k][c] = last_x[k][c] + dx[k][c];
        }
        next.push_back(k);
        continue;
      }
      dv[k] = restrict(probe.Gradient(j));
      const Real last_val = val[k];
      val[k] = probe.Value(j);
      if (i > 0 && std::abs(last_val - val[k]) < tol) {
        result[k] = SearchResult::Success;
        continue;
      }
      const Real df = Norm(dv[k]);
      stepping.push_back(k);
      at1.push_back({x[k][0] + dv[k][0] * Delta / df,
                     x[k][1] + dv[k][1] * Delta / df,
                     x[k][2] + dv[k][2] * Delta / df});
    }
    // the derivative of |grad f| along grad f, for a Newton step
    probe.At(at1);
    for (size_t j = 0; j < stepping.size(); ++j) {
      const size_t k = stepping[j];
      const Real df = Norm(dv[k]);
      const Real ddf = (Norm(restrict(probe.Gradient(j))) - df) / Delta;
      const Real dl = -df / ddf;
      for (int c = 0; c < 3; ++c) {
        dx[k][c] = dv[k][c] * dl / df;
        if (dl > max_step) {
          dx[k][c] *= max_step / dl;
        }
      }
      last_x[k] = x[k];
      for (int c = 0; c < 3; ++c) {
        x[k][c] += dx[k][c];
      }
      WrapPhi(x[k]);
      next.push_back(k);
    }
    active.swap(next);
  }
  for (const size_t k : active) {
    x[k] = x0[k];
  }
  return result;
}

} // namespace

std::vector<SearchResult> FindMax(const pcms::Function<Real>& f,
                                  std::span<Real> val, std::span<Point> x,
                                  Real tol, Real max_step, int dim,
                                  const Point& norm)
{
  RequireScalar(f, "FindMax");
  Probe probe(f, true);
  return FindMax(probe, val, x, tol, max_step, dim, norm);
}

namespace
{

std::vector<SearchResult> FindVal(Probe& probe, Real val, std::span<Point> x,
                                  Real tol, Real max_step, int dim,
                                  const Point& norm)
{
  const Point n = Unit(norm);
  const size_t num = x.size();
  std::vector<SearchResult> result(num, SearchResult::Diverged);
  std::vector<Point> last_x(num), dx(num);
  std::vector<size_t> active(num);
  std::iota(active.begin(), active.end(), 0);
  std::vector<size_t> next;
  std::vector<Point> at;
  for (int i = 0; i < MaxIterations && !active.empty(); ++i) {
    probe.At(Gather(x, active, at));
    next.clear();
    for (size_t j = 0; j < active.size(); ++j) {
      const size_t k = active[j];
      if (!probe.Found(j)) {
        if (i == 0) {
          result[k] = SearchResult::OutOfBounds;
          continue;
        }
        for (int c = 0; c < 3; ++c) {
          dx[k][c] /= 2.;
          x[k][c] = last_x[k][c] + dx[k][c];
        }
        next.push_back(k);
        continue;
      }
      const Real error = probe.Value(j) - val;
      if (std::abs(error) < tol) {
        result[k] = SearchResult::Success;
        continue;
      }
      const Point& dv = probe.Gradient(j);
      // along norm, or down the gradient within the plane normal to norm
      const Point s = dim == 1 ? n
                               : Unit({dv[0] * (1. - n[0]), dv[1] * (1. - n[1]),
                                       dv[2] * (1. - n[2])});
      const Real df = Dot(dv, s);
      if (std::abs(df) < tol) {
        continue; // zero gradient: diverged
      }
      const Real dl = std::clamp(-error / df, -max_step, max_step);
      last_x[k] = x[k];
      for (int c = 0; c < 3; ++c) {
        dx[k][c] = dl * s[c];
        x[k][c] += dx[k][c];
      }
      WrapPhi(x[k]);
      next.push_back(k);
    }
    active.swap(next);
  }
  return result;
}

} // namespace

std::vector<SearchResult> FindVal(const pcms::Function<Real>& f, Real val,
                                  std::span<Point> x, Real tol, Real max_step,
                                  int dim, const Point& norm)
{
  RequireScalar(f, "FindVal");
  Probe probe(f, true);
  return FindVal(probe, val, x, tol, max_step, dim, norm);
}

std::vector<SearchResult> FindValOnLine(const pcms::Function<Real>& f, Real f0,
                                        std::span<const Point> x0,
                                        std::span<const Point> x1,
                                        std::span<Point> x, Real tol)
{
  RequireScalar(f, "FindValOnLine");
  const size_t num = x.size();
  std::vector<SearchResult> result(num, SearchResult::Diverged);
  std::vector<Point> origin(num), vec(num);
  std::vector<Real> len(num), l(num), last_l(num, 0.);
  std::vector<bool> checklen(num);
  for (size_t k = 0; k < num; ++k) {
    // equal ends: the line through the origin and x1, unbounded
    checklen[k] = x0[k] != x1[k];
    origin[k] = checklen[k] ? x0[k] : Point{0., 0., 0.};
    const Point d{x1[k][0] - origin[k][0], x1[k][1] - origin[k][1],
                  x1[k][2] - origin[k][2]};
    len[k] = Norm(d);
    vec[k] = Unit(d);
    l[k] = Dot(
      {x[k][0] - origin[k][0], x[k][1] - origin[k][1], x[k][2] - origin[k][2]},
      vec[k]);
  }
  Probe probe(f, true);
  std::vector<size_t> active(num);
  std::iota(active.begin(), active.end(), 0);
  std::vector<size_t> next;
  std::vector<Point> at;
  for (int i = 0; i < MaxIterations && !active.empty(); ++i) {
    for (const size_t k : active) {
      l[k] = std::max(l[k], 0.);
      if (checklen[k]) {
        l[k] = std::min(l[k], len[k]);
      }
      for (int c = 0; c < 3; ++c) {
        x[k][c] = l[k] * vec[k][c] + origin[k][c];
      }
    }
    probe.At(Gather(x, active, at));
    next.clear();
    for (size_t j = 0; j < active.size(); ++j) {
      const size_t k = active[j];
      if (!probe.Found(j)) {
        if (i == 0) {
          result[k] = SearchResult::OutOfBounds;
          continue;
        }
        l[k] = (l[k] + last_l[k]) / 2.;
        next.push_back(k);
        continue;
      }
      const Real error = probe.Value(j) - f0;
      if (std::abs(error) < tol) {
        result[k] = SearchResult::Success;
        continue;
      }
      // as 1.x: the derivative along the unit vector, over the line's length
      const Real dfdl = Dot(probe.Gradient(j), vec[k]) / len[k];
      last_l[k] = l[k];
      l[k] -= error / dfdl;
      next.push_back(k);
    }
    active.swap(next);
  }
  return result;
}

SearchResult GridifyLoop(std::span<const Point> path0, const Point& axis,
                         std::span<Point> path, std::span<Real> theta,
                         LoopParameter param)
{
  const size_t m0 = path0.size();
  const size_t m = path.size();
  // the parameter at each point of the loop and back at its start
  std::vector<Real> l0(m0 + 1);
  int dir = 0;
  for (size_t i = 0; i <= m0; ++i) {
    const size_t ii = i < m0 ? i : 0;
    if (param == LoopParameter::PoloidalAngle) {
      l0[i] = std::atan2(path0[ii][2] - axis[2], path0[ii][0] - axis[0]);
      if (l0[i] < 0.) {
        l0[i] += TwoPi;
      }
    } else if (param == LoopParameter::ToroidalAngle) {
      l0[i] = path0[ii][1];
    } else if (i == 0) {
      l0[i] = 0.;
    } else {
      const Point& a = path0[i - 1];
      const Point& b = path0[ii];
      const Point d{b[0] * std::cos(b[1]) - a[0] * std::cos(a[1]),
                    b[0] * std::sin(b[1]) - a[0] * std::sin(a[1]), b[2] - a[2]};
      l0[i] = l0[i - 1] + Norm(d);
    }
    if (i > 0) {
      const Real d = l0[i] - l0[i - 1];
      if (i == 1) {
        dir = d < 0 ? -1 : 1;
      } else if (d * dir < 0) {
        return SearchResult::Diverged; // not single-valued in the parameter
      }
    }
  }
  path[0] = path0[0];
  theta[0] = 0.;
  size_t j = 0;
  for (size_t i = 1; i < m; ++i) {
    Real l = 0;
    if (param == LoopParameter::ArcLength) {
      theta[i] = static_cast<Real>(i) / static_cast<Real>(m);
      l = theta[i] * l0[m0];
    } else {
      theta[i] = static_cast<Real>(i) / static_cast<Real>(m) * TwoPi;
      l = theta[i];
    }
    while (l0[j + 1] < l) {
      ++j;
      if (j == m0) {
        j = m0 - 1;
        break;
      }
    }
    const Real w = (l - l0[j]) / (l0[j + 1] - l0[j]);
    const size_t jp = j + 1 < m0 ? j + 1 : 0;
    for (int c = 0; c < 3; ++c) {
      path[i][c] = path0[j][c] * (1. - w) + path0[jp][c] * w;
    }
  }
  return SearchResult::Success;
}

SearchResult GridifySurface(std::span<const Point> path0, const Point& axis,
                            int nphi, int ntheta, std::span<Point> path,
                            std::span<Real> phi, std::span<Real> theta)
{
  // path0 holds loops of constant phi one after another
  const size_t m0 = path0.size();
  size_t istart = 0;
  int iphi = 0;
  for (size_t i = 1; i < m0; ++i) {
    size_t m = 0;
    if (i == m0 - 1) {
      m = i - istart + 1;
    } else if (path0[i][1] != path0[i - 1][1]) {
      m = i - istart;
    }
    if (m == 0) {
      continue;
    }
    if (iphi >= nphi) {
      return SearchResult::Diverged;
    }
    phi[iphi] = path0[i - 1][1];
    const SearchResult result =
      GridifyLoop(path0.subspan(istart, m), axis,
                  path.subspan(static_cast<size_t>(ntheta) * iphi, ntheta),
                  theta, LoopParameter::ArcLength);
    if (result != SearchResult::Success) {
      return result;
    }
    istart = i;
    ++iphi;
  }
  return iphi == nphi ? SearchResult::Success : SearchResult::Diverged;
}

namespace
{

// Moves the points `which` of x by FindVal, writing their results.
void FindValAt(Probe& probe, Real val, std::vector<Point>& x,
               const std::vector<size_t>& which, Real tol, Real max_step,
               int dim, const Point& norm, std::vector<SearchResult>& result)
{
  std::vector<Point> at;
  Gather(x, which, at);
  const auto found = FindVal(probe, val, at, tol, max_step, dim, norm);
  for (size_t j = 0; j < which.size(); ++j) {
    x[which[j]] = at[j];
    result[which[j]] = found[j];
  }
}

} // namespace

std::vector<Path> Isosurface2D(const pcms::Function<Real>& f, Real val,
                               std::span<const Point> guess, const Point& norm,
                               const std::optional<Point>& dd, Real dl,
                               Real tol, Real max_step)
{
  constexpr size_t MaxPoints = 10000;
  RequireScalar(f, "Isosurface2D");
  Probe probe(f, true);
  const size_t num = guess.size();
  std::vector<Path> paths(num, Path{SearchResult::Diverged, {}});
  std::vector<Point> x(guess.begin(), guess.end());
  std::vector<SearchResult> found(num);
  std::vector<size_t> active(num), next;
  std::iota(active.begin(), active.end(), 0);
  std::vector<Point> at;
  while (!active.empty()) {
    FindValAt(probe, val, x, active, tol, max_step, 2, norm, found);
    // as 1.x, only leaving the domain ends the path
    std::erase_if(active, [&](size_t k) {
      if (found[k] == SearchResult::OutOfBounds) {
        paths[k].result = SearchResult::OutOfBounds;
        return true;
      }
      paths[k].points.push_back(x[k]);
      return false;
    });
    if (!dd) {
      probe.At(Gather(x, active, at));
    }
    next.clear();
    for (size_t j = 0; j < active.size(); ++j) {
      const size_t k = active[j];
      // the step: along dd, or along norm x grad(f)
      Point t{};
      if (dd) {
        t = *dd;
      } else {
        const Point& dv = probe.Gradient(j);
        t = {norm[1] * dv[2] - norm[2] * dv[1],
             norm[2] * dv[0] - norm[0] * dv[2],
             norm[0] * dv[1] - norm[1] * dv[0]};
      }
      const Real dt = Norm(t);
      if (dt == 0. || std::isnan(dt)) {
        continue; // no gradient: diverged
      }
      const auto& p = paths[k].points;
      const Point d{x[k][0] * std::cos(x[k][1]) - p[0][0] * std::cos(p[0][1]),
                    x[k][0] * std::sin(x[k][1]) - p[0][0] * std::sin(p[0][1]),
                    x[k][2] - p[0][2]};
      if (Dot(d, d) < 4. * dl * dl && p.size() > 3) {
        paths[k].result = SearchResult::Success;
        continue;
      }
      if (p.size() >= MaxPoints) {
        continue; // not closed: diverged
      }
      for (int c = 0; c < 3; ++c) {
        x[k][c] += dl * t[c] / dt;
      }
      next.push_back(k);
    }
    active.swap(next);
  }
  return paths;
}

std::vector<Path> Isosurface(const pcms::Function<Real>& f, Real val,
                             std::span<const Point> guess, const Point& enclose,
                             Real dl, Real tol, Real max_step, int nphi)
{
  constexpr size_t MaxPoints = 1000000;
  RequireScalar(f, "Isosurface");
  Probe probe(f, true);
  const Point plane{0., 1., 0.};
  const size_t num = guess.size();
  std::vector<Path> paths(num, Path{SearchResult::Diverged, {}});
  std::vector<Point> x(guess.begin(), guess.end());
  // direction of each guess from `enclose` in (R, Z)
  std::vector<Real> co(num), sn(num);
  std::vector<size_t> start(num, 0);
  std::vector<int> iphi(num, 0);
  for (size_t k = 0; k < num; ++k) {
    const Real dx = x[k][0] - enclose[0];
    const Real dz = x[k][2] - enclose[2];
    const Real d = std::hypot(dx, dz);
    co[k] = dx / d;
    sn[k] = dz / d;
  }
  std::vector<SearchResult> found(num);
  std::vector<size_t> active(num), first, rest, next;
  std::iota(active.begin(), active.end(), 0);
  std::vector<Point> at;
  while (!active.empty()) {
    // the first point of a loop keeps its direction (1.x searches along
    // `enclose` as a direction); the others stay in their phi plane
    first.clear();
    rest.clear();
    for (const size_t k : active) {
      (paths[k].points.size() == start[k] ? first : rest).push_back(k);
    }
    FindValAt(probe, val, x, first, tol, max_step, 1, enclose, found);
    FindValAt(probe, val, x, rest, tol, max_step, 2, plane, found);
    std::erase_if(active, [&](size_t k) {
      if (found[k] != SearchResult::Success) {
        paths[k].result = found[k];
        return true;
      }
      paths[k].points.push_back(x[k]);
      return false;
    });
    probe.At(Gather(x, active, at));
    next.clear();
    for (size_t j = 0; j < active.size(); ++j) {
      const size_t k = active[j];
      if (!probe.Found(j)) {
        paths[k].result = SearchResult::OutOfBounds;
        continue;
      }
      const Point& dv = probe.Gradient(j);
      // t = phi-hat x grad(f)
      const Real dt = std::hypot(dv[2], dv[0]);
      if (dt == 0.) {
        continue; // no gradient: diverged
      }
      const Real tr = dv[2] * dl / dt;
      const Real tz = -dv[0] * dl / dt;
      auto& p = paths[k].points;
      const size_t n = p.size();
      if (n > start[k] + 3 &&
          std::hypot(x[k][0] - p[start[k]][0], x[k][2] - p[start[k]][2]) < dl) {
        if (iphi[k] == nphi - 1) {
          paths[k].result = SearchResult::Success;
          continue;
        }
        // around once poloidally: step toroidally, keeping the poloidal
        // angle and moving to stay on the surface
        const Real d =
          std::hypot(p[start[k]][0] - enclose[0], p[start[k]][2] - enclose[2]);
        const Real dphi = TwoPi / nphi;
        const Real dd = -dv[1] * dphi / (dv[0] * co[k] + dv[2] * sn[k]);
        x[k][0] = enclose[0] + (d + dd) * co[k];
        x[k][1] = guess[k][1] + TwoPi * (iphi[k] + 1) / nphi;
        x[k][2] = enclose[2] + (d + dd) * sn[k];
        WrapPhi(x[k]);
        // as 1.x: the next loop starts one point after the next point
        start[k] = n + 1;
        ++iphi[k];
      }
      if (n >= MaxPoints) {
        continue; // not closed: diverged
      }
      x[k][0] += tr;
      x[k][2] += tz;
      next.push_back(k);
    }
    active.swap(next);
  }
  return paths;
}

Path GeomIsosurface(const pcms::Function<Real>& f, Real val, const Point& guess,
                    const Point& axis, Real tol, Real max_step, int nphi,
                    int ntheta)
{
  RequireScalar(f, "GeomIsosurface");
  Probe probe(f, true);
  const auto planes = static_cast<size_t>(nphi);
  Path path{SearchResult::Success,
            std::vector<Point>(planes * static_cast<size_t>(ntheta))};
  // the planes together; within a plane, each angle starts at the distance
  // from the axis found at the previous one
  std::vector<Real> l(planes,
                      std::hypot(guess[0] - axis[0], guess[2] - axis[2]));
  std::vector<Point> x(planes);
  std::vector<SearchResult> found(planes);
  std::vector<size_t> all(planes);
  std::iota(all.begin(), all.end(), 0);
  for (int j = 0; j < ntheta; ++j) {
    const Real theta = TwoPi * j / ntheta;
    for (size_t i = 0; i < planes; ++i) {
      x[i] = {l[i] * std::cos(theta) + axis[0], TwoPi * i / nphi,
              l[i] * std::sin(theta) + axis[2]};
    }
    // as 1.x, along `axis` as a direction
    FindValAt(probe, val, x, all, tol, max_step, 1, axis, found);
    for (size_t i = 0; i < planes; ++i) {
      if (found[i] != SearchResult::Success) {
        path.result = found[i];
        return path;
      }
      path.points[i * ntheta + j] = x[i];
      l[i] = std::hypot(x[i][0] - axis[0], x[i][2] - axis[2]);
    }
  }
  return path;
}

} // namespace fusionio
