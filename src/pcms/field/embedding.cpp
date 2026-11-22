#include "pcms/field/embedding.hpp"
#include "pcms/backends/uniform_grid/uniform_grid_geometry.hpp"
#include "pcms/field/detail/transformed_point_evaluator.hpp"
#include "pcms/geometry/detail/geometry.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/dense_linear_algebra.hpp"
#include "pcms/utility/detail/memory_spaces.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace pcms
{

namespace
{

template <typename T>
using DeviceView = Kokkos::View<T, DeviceMemorySpace>;

// jacobian(p, j, i) = d x_j / d xi_i from the gradient of the position, whose
// component j has the partials d x_j / d xi_i at columns j * dp + i.
void ReshapeJacobian(DeviceView<const Real**> gradient,
                     DeviceView<Real***> jacobian)
{
  const int dt = jacobian.extent(1);
  const int dp = jacobian.extent(2);
  detail::ForEach(
    "position_jacobian", static_cast<LO>(jacobian.extent(0)),
    KOKKOS_LAMBDA(const LO p) {
      for (int j = 0; j < dt; ++j) {
        for (int i = 0; i < dp; ++i) {
          jacobian(p, j, i) = gradient(p, j * dp + i);
        }
      }
    });
}

// The metric g = J^T G J induced in the parameter system and its volume factor:
// the physical volume element per unit of parameter volume, sqrt(det g) times
// the factor of the coordinate system's volume element relative to its metric.
void InduceMetric(View<3, const Real, DeviceMemorySpace> jacobian,
                  const BoundCoordinateSystem& target,
                  DeviceView<Real***> metric, DeviceView<Real*> volume)
{
  const int dp = metric.extent(1);
  const int dt = jacobian.extent(1);
  const auto g_target = target.Metric();
  const auto volume_target = target.VolumeFactor();
  detail::ForEach(
    "embedding_induced_metric", static_cast<LO>(metric.extent(0)),
    KOKKOS_LAMBDA(const LO p) {
      Real g[3][3] = {};
      Real g_t[3][3] = {};
      for (int j = 0; j < dt; ++j) {
        for (int k = 0; k < dt; ++k) {
          g_t[j][k] = g_target(p, j, k);
        }
      }
      for (int a = 0; a < dp; ++a) {
        for (int b = 0; b < dp; ++b) {
          for (int j = 0; j < dt; ++j) {
            for (int k = 0; k < dt; ++k) {
              g[a][b] += jacobian(p, j, a) * g_t[j][k] * jacobian(p, k, b);
            }
          }
          metric(p, a, b) = g[a][b];
        }
      }
      volume(p) = Kokkos::sqrt(detail::Determinant(g, dp)) * volume_target(p) /
                  Kokkos::sqrt(detail::Determinant(g_t, dt));
    });
}

// The covariant metric g_ab of the parameter system from the values of a
// given metric, [point][a * dp + b] (g^ab, inverted, if `contravariant`), and
// its volume factor sqrt(det g).
void GivenMetric(DeviceView<const Real**> values, bool contravariant,
                 DeviceView<Real***> metric, DeviceView<Real*> volume)
{
  const int dp = metric.extent(1);
  detail::ForEach(
    "embedding_given_metric", static_cast<LO>(metric.extent(0)),
    KOKKOS_LAMBDA(const LO p) {
      Real g[3][3] = {};
      for (int a = 0; a < dp; ++a) {
        for (int b = 0; b < dp; ++b) {
          g[a][b] = values(p, a * dp + b);
        }
      }
      if (contravariant) {
        Real inverse[3][3] = {};
        detail::Invert(g, dp, inverse);
        for (int a = 0; a < dp; ++a) {
          for (int b = 0; b < dp; ++b) {
            g[a][b] = inverse[a][b];
          }
        }
      }
      for (int a = 0; a < dp; ++a) {
        for (int b = 0; b < dp; ++b) {
          metric(p, a, b) = g[a][b];
        }
      }
      volume(p) = Kokkos::sqrt(detail::Determinant(g, dp));
    });
}

struct Evaluation
{
  DeviceView<Real**> mapped;    // [point][coordinate system dimension]
  DeviceView<Real***> jacobian; // [point][coordinate system][parameter]
};

// A position Function: its value and Jacobian at parameter points.
class Position
{
public:
  /// @param who the caller, for the messages
  Position(std::shared_ptr<const Function<Real>> position, const char* who)
    : position_(std::move(position))
  {
    const std::string prefix = std::string(who) + ": ";
    if (position_ == nullptr) {
      throw pcms_error(prefix + "the position must not be null");
    }
    if (position_->GetData().GetValueBasis().kind != ValueKind::Coordinates) {
      throw pcms_error(prefix + "the position's values must be coordinates "
                                "(ValueBasis::Coordinates)");
    }
    if (!position_->GetSpace().Supports(DifferentialOperator::Gradient)) {
      throw pcms_error(prefix +
                       "the position's space does not evaluate gradients");
    }
    if (ParameterSystem()->Dimension() > 3 || System()->Dimension() > 3) {
      throw pcms_error(prefix + "dimensions above 3 are not supported");
    }
  }

  [[nodiscard]] const std::shared_ptr<const Function<Real>>& Get() const
  {
    return position_;
  }
  /// the coordinate system the position's values are points of
  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>& System() const
  {
    return position_->GetData().GetValueBasis().system;
  }
  /// the coordinate system of the parameter geometry
  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>& ParameterSystem()
    const
  {
    return position_->GetSpace().GetCoordinateSystem();
  }

  /// The position and its Jacobian at parameter points; points outside the
  /// parameter geometry use its nearest cell.
  [[nodiscard]] Evaluation Evaluate(
    Rank2View<const Real, DeviceMemorySpace> xi) const
  {
    const FunctionSpace& space = position_->GetSpace();
    const auto geometry = space.GetGeometry();
    const auto n = xi.extent(0);
    const int dp = ParameterSystem()->Dimension();
    const int dt = System()->Dimension();
    const auto bound = geometry->Bind(
      CoordinateView<DeviceMemorySpace>(geometry->GetCoordinateSystem(), xi),
      OutOfBoundsPolicy{OutOfBoundsMode::NEAREST_BOUNDARY});
    const ValueBasis basis = position_->GetData().GetValueBasis();
    Evaluation e{DeviceView<Real**>("position_mapped", n, dt),
                 DeviceView<Real***>("position_jacobian", n, dt, dp)};
    space.CreatePointEvaluator<Real>(bound)->Evaluate(
      *position_,
      ValueView<Real, DeviceMemorySpace>(basis, MakeRank2View(e.mapped)));
    DeviceView<Real**> gradient("position_gradient", n,
                                static_cast<std::size_t>(dt) * dp);
    const auto evaluator =
      space.CreatePointEvaluator<Real>(bound, DifferentialOperator::Gradient);
    evaluator->Evaluate(
      *position_, ValueView<Real, DeviceMemorySpace>(
                    evaluator->OutputBasis(basis), MakeRank2View(gradient)));
    ReshapeJacobian(gradient, e.jacobian);
    return e;
  }

private:
  std::shared_ptr<const Function<Real>> position_;
};

// The coordinate map of a position: parameter system -> the position's
// coordinate system.
class PositionMap final : public CoordinateMap
{
public:
  explicit PositionMap(Position position)
    : CoordinateMap(position.ParameterSystem(), position.System()),
      position_(std::move(position))
  {
  }

protected:
  [[nodiscard]] std::unique_ptr<BoundCoordinateMap> BindImpl(
    const CoordinateView<DeviceMemorySpace>& query_points) const override
  {
    auto e = position_.Evaluate(query_points.GetValues());
    return std::make_unique<detail::BoundJacobianMap>(
      GetTargetCoordinateSystem(), std::move(e.mapped), std::move(e.jacobian));
  }

private:
  Position position_;
};

// ---------------------------------------------------------------------------
// The embedded geometry
// ---------------------------------------------------------------------------

// Newton iterations of one candidate cell, the residual that counts as
// converged (relative to the extent of the image), the fraction of a cell
// extent the image bounding boxes are padded by (the image of a cell bulges
// beyond its sampled points), and the sample lattice per cell axis.
constexpr int MaxNewtonIterations = 30;
constexpr Real ResidualTol = 1e-10;
constexpr Real BoxPadding = 0.1;
constexpr int SamplesPerAxis = 4;

// Where the points of the embedded geometry live in the target system and how
// they are binned: a uniform grid of `count` bins over [lo, lo + count *
// width], periodic axes wrapped into [wrap_lo, wrap_lo + period).
struct BinGrid
{
  int dim = 0;
  Real lo[3] = {};
  Real width[3] = {};
  int count[3] = {};
  Real wrap_lo[3] = {};
  Real period[3] = {}; // 0: not periodic
};

// Wraps the points into their periods (and shifts them by `shift` periods:
// where the image of the geometry is wider than a period, a point can be the
// image of a cell in either of two windows) and finds the candidate cells of
// each: the range [start, start + count) of the bins' cell list.
void FindCandidates(Rank2View<const Real, DeviceMemorySpace> points, int shift,
                    DeviceView<const bool*> found, BinGrid bins,
                    DeviceView<const LO*> bin_offsets,
                    DeviceView<Real**> wrapped, DeviceView<LO*> start,
                    DeviceView<LO*> count)
{
  detail::ForEach(
    "embedded_candidates", static_cast<LO>(points.extent(0)),
    KOKKOS_LAMBDA(const LO p) {
      if (found(p)) {
        // keeps the reduced coordinates of the window it was found in
        start(p) = 0;
        count(p) = 0;
        return;
      }
      LO bin = 0;
      bool inside = true;
      for (int d = bins.dim - 1; d >= 0; --d) {
        Real q = points(p, d);
        if (bins.period[d] > 0) {
          const Real t = q - bins.wrap_lo[d];
          q = bins.wrap_lo[d] + t -
              bins.period[d] * (Kokkos::floor(t / bins.period[d]) - shift);
        }
        wrapped(p, d) = q;
        const Real t = (q - bins.lo[d]) / bins.width[d];
        const LO index = static_cast<LO>(Kokkos::floor(t));
        inside = inside && t >= 0 && index < bins.count[d];
        bin = bin * bins.count[d] + (inside ? index : 0);
      }
      start(p) = inside ? bin_offsets(bin) : 0;
      count(p) = inside ? bin_offsets(bin + 1) - bin_offsets(bin) : 0;
    });
}

// Starts candidate round `round` at the points that are not found and have
// that many candidates: they become active at the center of the candidate
// cell.
template <unsigned Dim>
void StartRound(int round, UniformGrid<Dim> grid,
                DeviceView<const LO*> bin_cells, DeviceView<const LO*> start,
                DeviceView<const LO*> count, DeviceView<const bool*> found,
                DeviceView<bool*> active, DeviceView<bool*> converged,
                DeviceView<LO*> cell, DeviceView<Real**> xi)
{
  detail::ForEach(
    "embedded_start_round", static_cast<LO>(start.extent(0)),
    KOKKOS_LAMBDA(const LO p) {
      active(p) = !found(p) && count(p) > round;
      converged(p) = false;
      if (active(p)) {
        cell(p) = bin_cells(start(p) + round);
        const auto box = grid.GetCellBBOX(cell(p));
        for (unsigned d = 0; d < Dim; ++d) {
          xi(p, d) = box.center[d];
        }
      }
    });
}

// One Gauss-Newton step of the active points that have not converged: the
// residual r = target - x(xi) is reduced by the step solving
// (J J^T) delta = J r, which is Newton's for a square J. The step is
// regularized where J is singular (a pole of the map): J J^T + eps I.
// @return the number of points that took a step
template <unsigned Dim>
LO NewtonStep(UniformGrid<Dim> grid, Real residual_tolerance, bool update,
              Rank2View<const Real, DeviceMemorySpace> target,
              DeviceView<const Real**> mapped,
              View<3, const Real, DeviceMemorySpace> jacobian,
              DeviceView<const bool*> active, DeviceView<bool*> converged,
              DeviceView<const LO*> cell, DeviceView<Real**> xi)
{
  const int dt = target.extent(1);
  LO stepped = 0;
  Kokkos::parallel_reduce(
    "embedded_newton_step",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(
      0, static_cast<LO>(target.extent(0))),
    KOKKOS_LAMBDA(const LO p, LO& count) {
      if (!active(p) || converged(p)) {
        return;
      }
      Real r[3] = {};
      Real norm2 = 0;
      for (int j = 0; j < dt; ++j) {
        r[j] = target(p, j) - mapped(p, j);
        norm2 += r[j] * r[j];
      }
      // a point within the tolerance still takes this step, which puts it
      // within roundoff of the solution (Newton converges quadratically):
      // otherwise a point on the face of a cell can be rejected by its cell
      const bool close = Kokkos::sqrt(norm2) <= residual_tolerance;
      if (!update) {
        converged(p) = close;
        return;
      }
      Real m[3][3] = {};
      Real b[3] = {};
      Real trace = 0;
      for (unsigned a = 0; a < Dim; ++a) {
        for (unsigned c = 0; c < Dim; ++c) {
          for (int j = 0; j < dt; ++j) {
            m[a][c] += jacobian(p, j, a) * jacobian(p, j, c);
          }
        }
        for (int j = 0; j < dt; ++j) {
          b[a] += jacobian(p, j, a) * r[j];
        }
        trace += m[a][a];
      }
      const Real scale = trace / Dim;
      Real reference = 1;
      for (unsigned a = 0; a < Dim; ++a) {
        reference *= scale;
      }
      Real regular[3][3];
      for (unsigned a = 0; a < Dim; ++a) {
        for (unsigned c = 0; c < Dim; ++c) {
          regular[a][c] = m[a][c];
        }
      }
      if (detail::Determinant(m, Dim) <= 1e-12 * reference) {
        for (unsigned a = 0; a < Dim; ++a) {
          regular[a][a] += 1e-6 * scale + 1e-300;
        }
      }
      detail::Solve(regular, Dim, b);
      const auto box = grid.GetCellBBOX(cell(p));
      for (unsigned d = 0; d < Dim; ++d) {
        // stay within a cell of the candidate: the position function is
        // evaluated by extrapolation outside the grid
        xi(p, d) =
          Kokkos::clamp(xi(p, d) + b[d], box.center[d] - 2 * box.half_width[d],
                        box.center[d] + 2 * box.half_width[d]);
      }
      converged(p) = close;
      ++count;
    },
    stepped);
  return stepped;
}

// A converged point whose parameter coordinates are in its candidate cell is
// found there.
template <unsigned Dim>
void AcceptRound(UniformGrid<Dim> grid, Real slack,
                 DeviceView<const bool*> active,
                 DeviceView<const bool*> converged, DeviceView<const LO*> cell,
                 DeviceView<const Real**> xi, DeviceView<bool*> found)
{
  detail::ForEach(
    "embedded_accept_round", static_cast<LO>(active.extent(0)),
    KOKKOS_LAMBDA(const LO p) {
      if (!active(p) || !converged(p)) {
        return;
      }
      const auto box = grid.GetCellBBOX(cell(p));
      bool inside = true;
      for (unsigned d = 0; d < Dim; ++d) {
        inside = inside && Kokkos::fabs(xi(p, d) - box.center[d]) <=
                             box.half_width[d] * (1 + slack);
      }
      found(p) = inside;
    });
}

// ---------------------------------------------------------------------------

// What the BoundPoints of an embedded geometry hold: the parameter
// coordinates of the points, bound on the parameter geometry, and the
// geometric factors there (computed on first use).
class EmbeddedBound : public BoundPoints
{
public:
  /// @return the points bound on the parameter geometry, at the parameter
  ///         coordinates: what the evaluators of its spaces take
  [[nodiscard]] const std::shared_ptr<const BoundPoints>& Inner() const noexcept
  {
    return inner_;
  }

  /// @return the Pushforward of the Jacobian: parameter-system components to
  ///         components in the geometry's coordinate system, at the points
  [[nodiscard]] const ValueTransform& Pushforward() const
  {
    if (!pushforward_) {
      const auto jacobian = Jacobian();
      if (system_ == nullptr && position_.System()->HasMetric()) {
        // a given metric: the coordinate system's is not bound yet
        system_ = position_.System()->Bind(MakeConstRank2View(mapped_));
      }
      pushforward_.emplace(detail::Pushforward(
        jacobian, position_.ParameterSystem(), position_.System(),
        system_ != nullptr ? system_->Metric() : detail::MetricView{}));
    }
    return *pushforward_;
  }

protected:
  EmbeddedBound(std::shared_ptr<const Geometry> geometry,
                const BindRequest& request, Position position,
                std::shared_ptr<const Function<Real>> metric)
    : BoundPoints(std::move(geometry), request),
      position_(std::move(position)),
      metric_(std::move(metric))
  {
  }

  // Takes the parameter coordinates of the points: `xi`, bound on `parameter`.
  void SetParameters(const Geometry& parameter, DeviceView<Real**> xi,
                     OutOfBoundsPolicy policy)
  {
    xi_ = std::move(xi);
    inner_ = parameter.Bind(
      CoordinateView<DeviceMemorySpace>(parameter.GetCoordinateSystem(),
                                        MakeConstRank2View(xi_)),
      policy);
    system_.reset();
    pushforward_.reset();
  }

  // J from the Gradient of the position at the parameter coordinates of the
  // points, and the given metric there or the metric J^T G J the position
  // induces
  [[nodiscard]] detail::GeometricFactors ComputeFactors() const override
  {
    auto e = position_.Evaluate(MakeConstRank2View(xi_));
    const auto n = xi_.extent(0);
    const int dp = position_.ParameterSystem()->Dimension();
    const int dt = position_.System()->Dimension();
    DeviceView<Real***> metric("parameter_metric", n, dp, dp);
    DeviceView<Real*> volume("parameter_volume_factor", n);
    if (metric_ != nullptr) {
      const ValueBasis basis = metric_->GetData().GetValueBasis();
      DeviceView<Real**> values("given_metric", n,
                                static_cast<std::size_t>(dp) * dp);
      metric_->GetSpace().CreatePointEvaluator<Real>(inner_)->Evaluate(
        *metric_,
        ValueView<Real, DeviceMemorySpace>(basis, MakeRank2View(values)));
      GivenMetric(values, basis.variance[0] == Variance::Contravariant, metric,
                  volume);
    } else {
      system_ = position_.System()->Bind(MakeConstRank2View(e.mapped));
      InduceMetric(
        View<3, const Real, DeviceMemorySpace>(e.jacobian.data(), n, dt, dp),
        *system_, metric, volume);
    }
    mapped_ = e.mapped;
    return detail::GeometricFactors(
      static_cast<LO>(n), e.jacobian,
      std::make_shared<const BoundCoordinateSystem>(static_cast<LO>(n), metric,
                                                    volume));
  }

private:
  Position position_;
  std::shared_ptr<const Function<Real>> metric_;
  std::shared_ptr<const BoundPoints> inner_;
  DeviceView<Real**> xi_;
  mutable DeviceView<Real**> mapped_;
  // the coordinate system's metric at the mapped points
  mutable std::shared_ptr<const BoundCoordinateSystem> system_;
  mutable std::optional<ValueTransform> pushforward_;
};

} // namespace

// The search of an embedded geometry, by the dimension of its grid.
class detail::EmbeddedSearch
{
public:
  virtual ~EmbeddedSearch() = default;

  [[nodiscard]] virtual std::shared_ptr<BoundPoints> Bind(
    std::shared_ptr<const EmbeddedGeometry> geometry,
    const BindRequest& request) const = 0;

  [[nodiscard]] virtual Quadrature CreateQuadrature(
    const EmbeddedGeometry& geometry, const QuadratureRule& rule) const = 0;
};

namespace
{

// The search over the cells of a uniform grid of the parameters: their images
// are binned, and the points are located by Newton in the cells of their bin.
template <unsigned Dim>
class GridSearch final : public detail::EmbeddedSearch
{
public:
  GridSearch(const uniform_grid::UniformGridGeometry<Dim>& parameter,
             Position position, const EmbeddingOptions& options)
    : position_(std::move(position)),
      grid_(parameter.GetGrid()),
      options_(options.search)
  {
    BuildSearch();
  }

  [[nodiscard]] std::shared_ptr<BoundPoints> Bind(
    std::shared_ptr<const EmbeddedGeometry> geometry,
    const BindRequest& request) const override;

  [[nodiscard]] const Position& PositionOf() const noexcept
  {
    return position_;
  }

  struct Located
  {
    DeviceView<Real**> xi;      // [point][parameter dimension]
    DeviceView<bool*> found;    // [point]
    DeviceView<Real**> reduced; // [point][coordinate system dimension]: the
                                // points reduced by the identifications
  };

  /// @brief The parameter coordinates of points of the target system: the
  ///        candidate cells of the bin of each point (periodic axes wrapped)
  ///        are tried in turn by Gauss-Newton from the cell center.
  [[nodiscard]] Located Locate(
    Rank2View<const Real, DeviceMemorySpace> points) const
  {
    const auto n = points.extent(0);
    const int dt = points.extent(1);
    DeviceView<Real**> wrapped("embedded_wrapped", n, dt);
    DeviceView<LO*> start("embedded_start", n);
    DeviceView<LO*> count("embedded_count", n);
    DeviceView<LO*> cell("embedded_cell", n);
    DeviceView<bool*> active("embedded_active", n);
    DeviceView<bool*> converged("embedded_converged", n);
    Located located{DeviceView<Real**>("embedded_xi", n, Dim),
                    DeviceView<bool*>("embedded_found", n), wrapped};
    const auto target = MakeConstRank2View(wrapped);
    // the points not found in the first window of a periodic axis are tried
    // in the next one
    for (int shift = 0; shift <= (two_windows_ ? 1 : 0); ++shift) {
      FindCandidates(points, shift, located.found, bins_, bin_offsets_, wrapped,
                     start, count);
      for (int round = 0; round < max_candidates_; ++round) {
        StartRound<Dim>(round, grid_, bin_cells_, start, count, located.found,
                        active, converged, cell, located.xi);
        for (int iteration = 0; iteration <= MaxNewtonIterations; ++iteration) {
          const auto e = position_.Evaluate(MakeConstRank2View(located.xi));
          const LO stepped =
            NewtonStep<Dim>(grid_, residual_tolerance_,
                            iteration < MaxNewtonIterations, target, e.mapped,
                            View<3, const Real, DeviceMemorySpace>(
                              e.jacobian.data(), e.jacobian.extent(0),
                              e.jacobian.extent(1), e.jacobian.extent(2)),
                            active, converged, cell, located.xi);
          if (stepped == 0) {
            break;
          }
        }
        AcceptRound<Dim>(grid_, options_.tolerance, active, converged, cell,
                         located.xi, located.found);
      }
    }
    return located;
  }

  /// @brief The tensor Gauss-Legendre rule of the parameter cells, with the
  ///        weights of the physical measure.
  [[nodiscard]] Quadrature CreateQuadrature(
    const EmbeddedGeometry& geometry,
    const detail::QuadratureRule& rule) const override
  {
    const auto* gauss = std::get_if<detail::GaussRule>(&rule);
    if (gauss == nullptr) {
      throw pcms_error("CreateQuadrature: an embedded geometry has only the "
                       "Gauss rule");
    }
    auto [xi, weights] =
      static_cast<const uniform_grid::UniformGridGeometry<Dim>&>(
        *geometry.GetParameterGeometry())
        .CellRule(gauss->order);
    return detail::BindQuadrature(
      geometry,
      CoordinateView<DeviceMemorySpace>(geometry.GetParameterSystem(),
                                        MakeConstRank2View(xi)),
      weights);
  }

private:
  // Bins the images of the cells: their bounding boxes, from a lattice of
  // sample points of each cell, padded.
  void BuildSearch()
  {
    const auto& system = position_.System();
    const int dt = system->Dimension();
    const LO num_cells = grid_.GetNumCells();
    size_t per_cell = 1;
    for (unsigned d = 0; d < Dim; ++d) {
      per_cell *= SamplesPerAxis + 1;
    }
    Kokkos::View<Real**, HostMemorySpace> samples("embedded_samples",
                                                  num_cells * per_cell, Dim);
    for (LO c = 0; c < num_cells; ++c) {
      const auto box = grid_.GetCellBBOX(c);
      for (size_t q = 0; q < per_cell; ++q) {
        size_t digits = q;
        for (unsigned d = 0; d < Dim; ++d) {
          const auto digit = static_cast<Real>(digits % (SamplesPerAxis + 1));
          digits /= SamplesPerAxis + 1;
          samples(c * per_cell + q, d) =
            box.center[d] +
            (2 * digit / SamplesPerAxis - 1) * box.half_width[d];
        }
      }
    }
    DeviceView<Real**> device("embedded_samples_device", samples.extent(0),
                              Dim);
    Kokkos::deep_copy(device, samples);
    const auto images = Kokkos::create_mirror_view_and_copy(
      HostMemorySpace(), position_.Evaluate(MakeConstRank2View(device)).mapped);

    // the image of each cell and of the geometry
    std::vector<std::array<Real, 3>> lo(num_cells), hi(num_cells);
    std::array<Real, 3> image_lo, image_hi;
    image_lo.fill(std::numeric_limits<Real>::max());
    image_hi.fill(std::numeric_limits<Real>::lowest());
    for (LO c = 0; c < num_cells; ++c) {
      lo[c].fill(std::numeric_limits<Real>::max());
      hi[c].fill(std::numeric_limits<Real>::lowest());
      for (size_t q = 0; q < per_cell; ++q) {
        for (int j = 0; j < dt; ++j) {
          const Real x = images(c * per_cell + q, j);
          lo[c][j] = std::min(lo[c][j], x);
          hi[c][j] = std::max(hi[c][j], x);
        }
      }
      for (int j = 0; j < dt; ++j) {
        image_lo[j] = std::min(image_lo[j], lo[c][j]);
        image_hi[j] = std::max(image_hi[j], hi[c][j]);
      }
    }
    Real extent2 = 0;
    for (int j = 0; j < dt; ++j) {
      extent2 += (image_hi[j] - image_lo[j]) * (image_hi[j] - image_lo[j]);
    }
    const Real extent = std::sqrt(extent2);
    residual_tolerance_ = ResidualTol * extent;
    // pad: the image of a cell bulges beyond its samples
    std::array<Real, 3> bin_lo, bin_hi;
    bin_lo.fill(std::numeric_limits<Real>::max());
    bin_hi.fill(std::numeric_limits<Real>::lowest());
    for (LO c = 0; c < num_cells; ++c) {
      for (int j = 0; j < dt; ++j) {
        const Real pad = BoxPadding * (hi[c][j] - lo[c][j]) + 1e-9 * extent;
        lo[c][j] -= pad;
        hi[c][j] += pad;
        bin_lo[j] = std::min(bin_lo[j], lo[c][j]);
        bin_hi[j] = std::max(bin_hi[j], hi[c][j]);
      }
    }

    bins_.dim = dt;
    const int default_count =
      std::max(1, static_cast<int>(std::ceil(std::pow(num_cells, 1.0 / dt))));
    LO num_bins = 1;
    for (int j = 0; j < dt; ++j) {
      bins_.count[j] =
        options_.divisions > 0 ? options_.divisions : default_count;
      bins_.lo[j] = bin_lo[j];
      bins_.width[j] =
        std::max((bin_hi[j] - bin_lo[j]) / bins_.count[j], 1e-300);
      bins_.wrap_lo[j] = image_lo[j];
      for (const auto& id : system->Identifications()) {
        if (id.axis == j && id.map == nullptr) {
          bins_.period[j] = id.length;
        }
      }
      two_windows_ = two_windows_ ||
                     (bins_.period[j] > 0 && image_hi[j] - image_lo[j] >
                                               bins_.period[j] * (1 + 1e-12));
      num_bins *= bins_.count[j];
    }

    // the cells of each bin: those whose padded image overlaps it
    const auto for_bins_of = [&](LO c, const auto& visit) {
      int first[3] = {}, last[3] = {};
      for (int j = 0; j < dt; ++j) {
        first[j] = std::clamp(
          static_cast<int>((lo[c][j] - bins_.lo[j]) / bins_.width[j]), 0,
          bins_.count[j] - 1);
        last[j] = std::clamp(
          static_cast<int>((hi[c][j] - bins_.lo[j]) / bins_.width[j]), 0,
          bins_.count[j] - 1);
      }
      int index[3] = {first[0], first[1], first[2]};
      while (true) {
        LO bin = 0;
        for (int j = dt - 1; j >= 0; --j) {
          bin = bin * bins_.count[j] + index[j];
        }
        visit(bin);
        int j = 0;
        while (j < dt && ++index[j] > last[j]) {
          index[j] = first[j];
          ++j;
        }
        if (j == dt) {
          break;
        }
      }
    };
    Kokkos::View<LO*, HostMemorySpace> offsets("embedded_bin_offsets",
                                               num_bins + 1);
    for (LO c = 0; c < num_cells; ++c) {
      for_bins_of(c, [&](LO bin) { ++offsets(bin + 1); });
    }
    max_candidates_ = 0;
    for (LO b = 0; b < num_bins; ++b) {
      max_candidates_ = std::max(max_candidates_, offsets(b + 1));
      offsets(b + 1) += offsets(b);
    }
    Kokkos::View<LO*, HostMemorySpace> cells("embedded_bin_cells",
                                             offsets(num_bins));
    std::vector<LO> next(offsets.data(), offsets.data() + num_bins);
    for (LO c = 0; c < num_cells; ++c) {
      for_bins_of(c, [&](LO bin) { cells(next[bin]++) = c; });
    }
    bin_offsets_ = DeviceView<LO*>("embedded_bin_offsets_device", num_bins + 1);
    bin_cells_ = DeviceView<LO*>("embedded_bin_cells_device", cells.extent(0));
    Kokkos::deep_copy(bin_offsets_, offsets);
    Kokkos::deep_copy(bin_cells_, cells);
  }

  Position position_;
  UniformGrid<Dim> grid_;
  SearchOptions options_;
  BinGrid bins_;
  DeviceView<LO*> bin_offsets_;
  DeviceView<LO*> bin_cells_;
  LO max_candidates_ = 0;
  Real residual_tolerance_ = 0;
  // the image is wider than a period of a periodic axis
  bool two_windows_ = false;
};

template <unsigned Dim>
class EmbeddedBinding final : public EmbeddedBound
{
public:
  EmbeddedBinding(std::shared_ptr<const EmbeddedGeometry> geometry,
                  const GridSearch<Dim>& search, const BindRequest& request)
    : EmbeddedBound(geometry, request, search.PositionOf(),
                    geometry->GetMetric()),
      geometry_(std::move(geometry)),
      search_(search)
  {
    Locate();
  }

protected:
  void RebindImpl() override { Locate(); }

private:
  // Points in the parameter system are bound on the parameter geometry as
  // they are; points of the target system are searched for first. The
  // parameter geometry gets all the points (a missing one at its last
  // iterate, clamped): the evaluators fill the rows of the missing points.
  void Locate()
  {
    const auto points = LocatedPointValues();
    const auto n = NumPoints();
    const bool parametric =
      GetBoundMap() == nullptr &&
      pcms::detail::SameCoordinateSystem(GetCoordinateSystem(),
                                         geometry_->GetParameterSystem());
    DeviceView<bool*> found;
    if (parametric) {
      DeviceView<Real**> xi("embedded_xi", n, Dim);
      Kokkos::deep_copy(xi, points);
      SetParameters(*geometry_->GetParameterGeometry(), xi, Policy());
      found = detail::FoundMask(n, {Inner()->MissingPoints()});
    } else {
      const auto located = search_.Locate(MakeConstRank2View(points));
      ReplaceLocatedPoints(located.reduced);
      SetParameters(*geometry_->GetParameterGeometry(), located.xi,
                    OutOfBoundsPolicy{OutOfBoundsMode::NEAREST_BOUNDARY});
      found = located.found;
    }
    SetFound(found);
  }

  std::shared_ptr<const EmbeddedGeometry> geometry_;
  const GridSearch<Dim>& search_;
};

template <unsigned Dim>
std::shared_ptr<BoundPoints> GridSearch<Dim>::Bind(
  std::shared_ptr<const EmbeddedGeometry> geometry,
  const BindRequest& request) const
{
  return std::make_shared<EmbeddedBinding<Dim>>(std::move(geometry), *this,
                                                request);
}

// Evaluates the fields of a space on the parameter geometry at the points of
// an embedded geometry: at their parameter coordinates. Vectors and covectors
// in parameter-system components come out in the geometry's coordinate system,
// through the Pushforward of the position's Jacobian.
class EmbeddedEvaluatorFactory final : public PointEvaluatorFactory
{
public:
  EmbeddedEvaluatorFactory(std::shared_ptr<const EmbeddedGeometry> embedded,
                           std::shared_ptr<const FunctionSpace> space)
    : embedded_(std::move(embedded)), space_(std::move(space))
  {
  }

  [[nodiscard]] std::shared_ptr<const Geometry> GetGeometry()
    const noexcept override
  {
    return embedded_;
  }
  [[nodiscard]] bool Supports(DifferentialOperator op) const noexcept override
  {
    return op == DifferentialOperator::Value;
  }

protected:
  PointEvaluatorVariant CreatePointEvaluatorImpl(
    Type value_type, std::shared_ptr<const BoundPoints> bound,
    DifferentialOperator op) const override
  {
    const auto embedded = std::dynamic_pointer_cast<const EmbeddedBound>(bound);
    if (value_type != Type::Real || embedded == nullptr) {
      throw pcms_error("CreatePointEvaluator: an embedded space evaluates "
                       "Real fields at the points of its embedded geometry");
    }
    return std::unique_ptr<PointEvaluator<Real>>(
      std::make_unique<detail::TransformedPointEvaluator<Real>>(
        embedded,
        CreateUntransformedPointEvaluator<Real>(*space_, embedded->Inner(), op),
        std::vector<detail::LazyValueTransform>{
          [embedded]() -> const ValueTransform& {
            return embedded->Pushforward();
          }}));
  }

private:
  std::shared_ptr<const EmbeddedGeometry> embedded_;
  std::shared_ptr<const FunctionSpace> space_;
};

/// @brief A space on the parameter geometry evaluated on an embedded
///        geometry; its quadrature is the space's, in the physical measure.
class EmbeddedSpace final : public FunctionSpace
{
public:
  EmbeddedSpace(std::shared_ptr<const EmbeddedGeometry> embedded,
                std::shared_ptr<const FunctionSpace> space)
    : FunctionSpace(space->GetLayout(),
                    std::make_shared<EmbeddedEvaluatorFactory>(embedded, space),
                    embedded),
      embedded_(std::move(embedded)),
      space_(std::move(space))
  {
  }

  [[nodiscard]] Quadrature CreateQuadrature() const override
  {
    const auto parameter = space_->CreateQuadrature();
    DeviceView<Real*> weights(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "embedded_weights"),
      parameter.weights.extent(0));
    Kokkos::deep_copy(weights, parameter.weights);
    if (embedded_->GetParameterSystem()->HasMetric()) {
      detail::ScaleByVolumeFactor(weights, *parameter.points, true);
    }
    return detail::BindQuadrature(*embedded_, parameter.points->LocatedPoints(),
                                  weights);
  }

private:
  std::shared_ptr<const EmbeddedGeometry> embedded_;
  std::shared_ptr<const FunctionSpace> space_;
};

} // namespace

EmbeddedGeometry::EmbeddedGeometry(
  std::shared_ptr<const Geometry> parameter,
  std::shared_ptr<const Function<Real>> position,
  std::shared_ptr<const Function<Real>> metric,
  std::shared_ptr<const detail::EmbeddedSearch> search)
  : parameter_(std::move(parameter)),
    position_(std::move(position)),
    metric_(std::move(metric)),
    system_(position_->GetData().GetValueBasis().system),
    search_(std::move(search))
{
}

EmbeddedGeometry::~EmbeddedGeometry() noexcept = default;

int EmbeddedGeometry::TopologicalDimension() const noexcept
{
  return parameter_->GetCoordinateSystem()->Dimension();
}

std::shared_ptr<const CoordinateSystem> EmbeddedGeometry::GetParameterSystem()
  const noexcept
{
  return parameter_->GetCoordinateSystem();
}

std::shared_ptr<BoundPoints> EmbeddedGeometry::BindImpl(
  const BindRequest& request) const
{
  return search_->Bind(
    std::static_pointer_cast<const EmbeddedGeometry>(shared_from_this()),
    request);
}

Quadrature EmbeddedGeometry::CreateQuadrature(
  const detail::QuadratureRule& rule) const
{
  return search_->CreateQuadrature(*this, rule);
}

std::shared_ptr<const CoordinateMap> CreateCoordinateMap(
  std::shared_ptr<const Function<Real>> position)
{
  return std::make_shared<const PositionMap>(
    Position(std::move(position), "CreateCoordinateMap"));
}

std::shared_ptr<const EmbeddedGeometry> CreateEmbeddedGeometry(
  std::shared_ptr<const Geometry> parameter_geometry,
  std::shared_ptr<const Function<Real>> position, EmbeddingOptions options)
{
  Position checked(std::move(position), "CreateEmbeddedGeometry");
  if (parameter_geometry == nullptr ||
      checked.Get()->GetSpace().GetGeometry() != parameter_geometry) {
    throw pcms_error("CreateEmbeddedGeometry: the position must be a Function "
                     "on the parameter geometry");
  }
  if (const auto& metric = options.metric) {
    const ValueBasis& basis = metric->GetData().GetValueBasis();
    if (metric->GetSpace().GetGeometry() != parameter_geometry ||
        basis.Rank() != 2 ||
        !detail::SameCoordinateSystem(basis.system,
                                      checked.ParameterSystem()) ||
        basis.variance[0] != basis.variance[1]) {
      throw pcms_error("CreateEmbeddedGeometry: the metric must be a Function "
                       "on the parameter geometry of rank 2 in the parameter "
                       "system, with both indices covariant (g_ij) or both "
                       "contravariant (g^ij)");
    }
  }
  if (detail::SameCoordinateSystem(checked.ParameterSystem(),
                                   checked.System())) {
    throw pcms_error(
      "CreateEmbeddedGeometry: the parameter geometry's coordinate system "
      "must differ from the coordinate system of the position (give the "
      "parameter geometry a parameter system)");
  }
  auto search = uniform_grid::VisitUniformGrid(
    parameter_geometry,
    [&]<unsigned Dim>(
      const std::shared_ptr<const uniform_grid::UniformGridGeometry<Dim>>& grid)
      -> std::shared_ptr<const detail::EmbeddedSearch> {
      if constexpr (Dim == 1) {
        return nullptr;
      } else {
        return std::make_shared<GridSearch<Dim>>(*grid, checked, options);
      }
    });
  if (search == nullptr) {
    throw pcms_error("CreateEmbeddedGeometry: only 2D and 3D uniform-grid "
                     "parameter geometries are supported");
  }
  return std::shared_ptr<const EmbeddedGeometry>(
    new EmbeddedGeometry(std::move(parameter_geometry), checked.Get(),
                         std::move(options.metric), std::move(search)));
}

std::shared_ptr<const FunctionSpace> CreateEmbeddedSpace(
  std::shared_ptr<const EmbeddedGeometry> embedded,
  std::shared_ptr<const FunctionSpace> space)
{
  if (space == nullptr || embedded == nullptr ||
      embedded->GetParameterGeometry() != space->GetGeometry()) {
    throw pcms_error("CreateEmbeddedSpace: the geometry must be an embedded "
                     "geometry of the space's geometry");
  }
  return std::make_shared<EmbeddedSpace>(std::move(embedded), std::move(space));
}

} // namespace pcms
