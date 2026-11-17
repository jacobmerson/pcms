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

} // namespace fusionio

#endif // FUSIONIO_UTILITIES_HPP
