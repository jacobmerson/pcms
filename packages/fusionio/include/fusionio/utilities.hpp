#ifndef FUSIONIO_UTILITIES_HPP
#define FUSIONIO_UTILITIES_HPP

/// Fusion-IO 1.x's utilities (fusion_io/isosurface.h and
/// fio_field::find_val_on_line) on batches of points.
///
/// Each function takes a function of a source (Source::CreateFunction) and
/// keeps 1.x's algorithm and arguments, but advances all of its points or
/// paths together: an iteration locates the points that still move once and
/// evaluates the function there in one batch, where 1.x evaluates point by
/// point. Points are (R, phi, Z); phi wraps into [0, 2 pi) as in 1.x.
/// Searches use derivatives, so their function's space must support
/// pcms::DifferentialOperator::Gradient. README.md maps the 1.x calls.

#include <fusionio/fusionio.hpp>
#include <optional>
#include <span>
#include <vector>

namespace fusionio
{

/// Outcome of a search (1.x's FIO_SUCCESS, FIO_OUT_OF_BOUNDS, FIO_DIVERGED).
enum class SearchResult
{
  Success,
  OutOfBounds,
  Diverged
};

/// fio_find_max: moves each point of `x` to a local extremum of the scalar
/// `f` (where its gradient vanishes) by Newton steps of at most `max_step`:
/// on the line through the point along `norm` (dim 1), in the plane through
/// it normal to `norm` (dim 2) or freely (dim 3). Converged when `f` changes
/// by less than `tol` in a step; `val` gets the value there. A point that
/// does not converge in 100 steps goes back to its start.
std::vector<SearchResult> FindMax(const pcms::Function<Real>& f,
                                  std::span<Real> val, std::span<Point> x,
                                  Real tol, Real max_step, int dim,
                                  const Point& norm);

/// fio_find_val: moves each point of `x` to where the scalar `f` equals
/// `val` (within `tol`): along `norm` (dim 1) or down the gradient within
/// the plane normal to `norm` (dim 2), by Newton steps of at most `max_step`.
std::vector<SearchResult> FindVal(const pcms::Function<Real>& f, Real val,
                                  std::span<Point> x, Real tol, Real max_step,
                                  int dim, const Point& norm);

/// fio_field::find_val_on_line: moves each point of `x` (the guess) to where
/// the scalar `f` equals `f0` (within `tol`) on the segment from x0[k] to
/// x1[k] (or, if they are equal, on the ray from the origin through x1[k]).
std::vector<SearchResult> FindValOnLine(const pcms::Function<Real>& f, Real f0,
                                        std::span<const Point> x0,
                                        std::span<const Point> x1,
                                        std::span<Point> x, Real tol = 1e-4);

/// What fio_gridify_loop spaces the points of a loop evenly in.
enum class LoopParameter
{
  ArcLength,
  PoloidalAngle, // about the axis
  ToroidalAngle
};

/// fio_gridify_loop: interpolates the closed loop `path0` linearly onto
/// path.size() points evenly spaced in `param`, writing the parameter of
/// each to `theta` (the fraction of the arc length, or the angle). Diverged
/// if the loop is not single-valued in the parameter.
SearchResult GridifyLoop(std::span<const Point> path0, const Point& axis,
                         std::span<Point> path, std::span<Real> theta,
                         LoopParameter param = LoopParameter::ArcLength);

/// fio_gridify_surface: GridifyLoop (by arc length) of each loop of constant
/// phi in `path0` onto `ntheta` points of `path`; `phi` gets the loops'
/// angles. Diverged if `path0` does not hold `nphi` loops.
SearchResult GridifySurface(std::span<const Point> path0, const Point& axis,
                            int nphi, int ntheta, std::span<Point> path,
                            std::span<Real> phi, std::span<Real> theta);

/// A traced path and how its tracing ended.
struct Path
{
  SearchResult result;
  std::vector<Point> points;
};

/// fio_isosurface_2d: for each guess, the closed contour of the scalar `f`
/// at `val` in the plane through it normal to `norm`, points about `dl`
/// apart (each found by FindVal, dim 2), stepping along `dd` if given, else
/// along norm x grad(f). Paths end when they close or leave the domain.
std::vector<Path> Isosurface2D(const pcms::Function<Real>& f, Real val,
                               std::span<const Point> guess, const Point& norm,
                               const std::optional<Point>& dd, Real dl,
                               Real tol, Real max_step);

/// fio_isosurface: for each guess, the surface of the scalar `f` at `val`
/// that encloses `enclose`, as poloidal loops (points about `dl` apart) in
/// `nphi` planes of phi one after another. As in 1.x, the first point of
/// each loop is searched along `enclose` taken as a direction.
std::vector<Path> Isosurface(const pcms::Function<Real>& f, Real val,
                             std::span<const Point> guess, const Point& enclose,
                             Real dl, Real tol, Real max_step, int nphi);

/// fio_geom_isosurface: the surface of the scalar `f` at `val` on a grid of
/// `nphi` x `ntheta` geometric angles about `axis` (point i * ntheta + j at
/// phi_i, theta_j), each point searched (FindVal, dim 1, along `axis` taken
/// as a direction as in 1.x) from the distance found at the previous angle.
Path GeomIsosurface(const pcms::Function<Real>& f, Real val, const Point& guess,
                    const Point& axis, Real tol, Real max_step, int nphi,
                    int ntheta);

/// A surface on a grid of nphi x ntheta points (path[i * ntheta + j] at
/// phi[i], theta[j]) and the axis found in each plane.
struct GriddedSurface
{
  SearchResult result;
  std::vector<Real> phi;
  std::vector<Real> theta;
  std::vector<Point> axis;
  std::vector<Point> path;
};

/// fio_gridded_isosurface: in each of `nphi` planes, the axis (FindMax from
/// `axis`, dim 2), a point of the contour at `val` out along R from it
/// (FindVal from R = guess[0]), the contour (Isosurface2D, points `dl_pol`
/// apart) and the contour gridded by arc length (GridifyLoop).
GriddedSurface GriddedIsosurface(const pcms::Function<Real>& f, Real val,
                                 const Point& guess, const Point& axis,
                                 Real dl_pol, Real tol, Real max_step, int nphi,
                                 int ntheta);

/// fio_find_rho_hybrid: for each ray from `axis` at angle (cos, sin) =
/// (ct[k], st[k]) in the plane `phi`, moves rho[k] to the distance where the
/// scalar `f` equals `target`: Newton steps, then a bracket and bisection.
std::vector<SearchResult> FindRhoHybrid(const pcms::Function<Real>& f,
                                        Real target, std::span<Real> rho,
                                        std::span<const Real> ct,
                                        std::span<const Real> st, Real phi,
                                        const Point& axis, Real tol);

/// fio_gridded_isosurface_hybrid: in each of `nphi` planes, the axis
/// (FindMax, dim 2, from the previous plane's) and the distance to the
/// surface at `val` along `ntheta` rays (FindRhoHybrid) from it.
GriddedSurface GriddedIsosurfaceHybrid(const pcms::Function<Real>& f, Real val,
                                       const Point& guess, const Point& axis,
                                       Real tol, Real max_step, int nphi,
                                       int ntheta);

/// fio_q_at_plane: the safety factor of the closed loop `x` from the
/// magnetic field `b`, the line integral of B_phi / (R B_pol) over 2 pi;
/// `bpol` and `b_out` get the poloidal field and the field at the points if
/// not empty. Nullopt if a point is outside the domain.
std::optional<Real> QAtPlane(const pcms::Function<Real>& b,
                             std::span<const Point> x,
                             std::span<Real> bpol = {},
                             std::span<Point> b_out = {});

/// fio_surface_average: the averages of the scalar `f` over `ns` surfaces of
/// nphi x ntheta points each, weighted by `jac` (IsosurfaceJacobian).
/// Nullopt if a point is outside the domain.
std::optional<std::vector<Real>> SurfaceAverage(const pcms::Function<Real>& f,
                                                int ns, int nphi, int ntheta,
                                                std::span<const Point> x,
                                                std::span<const Real> jac);

/// fio_surface_average (the loop variant): the average of the scalar `f`
/// over the closed loop `x`, weighted by dl / B_pol.
std::optional<Real> SurfaceAverage(const pcms::Function<Real>& f,
                                   std::span<const Point> x,
                                   std::span<const Real> bpol);

/// fio_isosurface_jacobian: the volume element at each point of `ns`
/// gridded surfaces of nphi x ntheta points. False if it changes sign.
bool IsosurfaceJacobian(int ns, int nphi, int ntheta, std::span<const Point> x,
                        std::span<Real> jac);

} // namespace fusionio

#endif // FUSIONIO_UTILITIES_HPP
