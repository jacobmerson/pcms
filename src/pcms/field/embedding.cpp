#include "pcms/field/embedding.hpp"
#include "pcms/backends/uniform_grid/uniform_grid_geometry.hpp"
#include "pcms/field/small_matrix.hpp"
#include "pcms/geometry/detail/geometry.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/detail/memory_spaces.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace pcms
{

namespace
{

template <typename T>
using DeviceView = Kokkos::View<T, DeviceMemorySpace>;

// jacobian(p, i, j) = d x_j / d xi_i from the gradient of the position, whose
// component j has the partials d x_j / d xi_i at columns j * dp + i.
void ReshapeJacobian(DeviceView<const Real**> gradient,
                     DeviceView<Real***> jacobian)
{
  const int dp = jacobian.extent(1);
  const int dt = jacobian.extent(2);
  detail::ForEach(
    "position_jacobian", static_cast<LO>(jacobian.extent(0)),
    KOKKOS_LAMBDA(const LO p) {
      for (int i = 0; i < dp; ++i) {
        for (int j = 0; j < dt; ++j) {
          jacobian(p, i, j) = gradient(p, j * dp + i);
        }
      }
    });
}

// The metric g = J^T G J induced in the parameter chart and its volume factor:
// the physical volume element per unit of parameter volume, sqrt(det g) times
// the factor of the chart's volume element relative to its metric.
void InduceMetric(View<3, const Real, DeviceMemorySpace> jacobian,
                  const BoundCoordinateSystem& target,
                  DeviceView<Real***> metric, DeviceView<Real*> volume)
{
  const int dp = metric.extent(1);
  const int dt = jacobian.extent(2);
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
              g[a][b] += jacobian(p, a, j) * g_t[j][k] * jacobian(p, b, k);
            }
          }
          metric(p, a, b) = g[a][b];
        }
      }
      volume(p) = Kokkos::sqrt(detail::Determinant(g, dp)) * volume_target(p) /
                  Kokkos::sqrt(detail::Determinant(g_t, dt));
    });
}

// The covariant metric g_ab of the parameter chart from the values of a
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
        for (int c = 0; c < dp; ++c) {
          Real m[3][3] = {};
          Real column[3] = {};
          for (int a = 0; a < dp; ++a) {
            for (int b = 0; b < dp; ++b) {
              m[a][b] = g[a][b];
            }
          }
          column[c] = 1;
          detail::Solve(m, dp, column);
          for (int a = 0; a < dp; ++a) {
            inverse[a][c] = column[a];
          }
        }
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
  DeviceView<Real**> mapped;    // [point][chart dimension]
  DeviceView<Real***> jacobian; // [point][parameter][chart]
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
    if (!position_->GetData().GetValueBasis().coordinates) {
      throw pcms_error(prefix + "the position's values must be coordinates "
                                "(ValueBasis::Coordinates)");
    }
    if (!position_->GetSpace().Supports(DifferentialOperator::Gradient)) {
      throw pcms_error(prefix +
                       "the position's space does not evaluate gradients");
    }
    if (ParameterChart()->Dimension() > 3 || Chart()->Dimension() > 3) {
      throw pcms_error(prefix + "dimensions above 3 are not supported");
    }
  }

  [[nodiscard]] const std::shared_ptr<const Function<Real>>& Get() const
  {
    return position_;
  }
  /// the chart the position's values are points of
  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>& Chart() const
  {
    return position_->GetData().GetValueBasis().system;
  }
  /// the chart of the parameter geometry
  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>& ParameterChart()
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
    const int dp = ParameterChart()->Dimension();
    const int dt = Chart()->Dimension();
    const auto bound = geometry->Bind(
      CoordinateView<DeviceMemorySpace>(geometry->GetCoordinateSystem(), xi),
      OutOfBoundsPolicy{OutOfBoundsMode::NEAREST_BOUNDARY});
    const ValueBasis basis = position_->GetData().GetValueBasis();
    Evaluation e{DeviceView<Real**>("position_mapped", n, dt),
                 DeviceView<Real***>("position_jacobian", n, dp, dt)};
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

// The coordinate map of a position: parameter chart -> the position's chart.
class PositionMap final : public CoordinateMap
{
public:
  explicit PositionMap(Position position)
    : CoordinateMap(position.ParameterChart(), position.Chart()),
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
            m[a][c] += jacobian(p, a, j) * jacobian(p, c, j);
          }
        }
        for (int j = 0; j < dt; ++j) {
          b[a] += jacobian(p, a, j) * r[j];
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

// Marks the listed points found, the others missing.
void MarkFound(DeviceView<const LO*> indices, DeviceView<bool*> found)
{
  Kokkos::deep_copy(found, false);
  detail::ForEach(
    "embedded_mark_found", static_cast<LO>(indices.extent(0)),
    KOKKOS_LAMBDA(const LO k) { found(indices(k)) = true; });
}

// The tensor Gauss-Legendre rule over each cell of the grid, in the parameter
// space: point q of cell c is at the nodes of the digits of q, and its weight
// is the product of the nodes' weights times the cell's half-widths (the
// rule's weights sum to 2).
template <unsigned Dim>
void FillCellRule(UniformGrid<Dim> grid, DeviceView<const Real*> nodes,
                  DeviceView<const Real*> rule_weights, DeviceView<Real**> xi,
                  DeviceView<Real*> weights)
{
  const int n = nodes.extent(0);
  int per_cell = 1;
  for (unsigned d = 0; d < Dim; ++d) {
    per_cell *= n;
  }
  detail::ForEach(
    "embedded_cell_rule", static_cast<LO>(weights.extent(0)),
    KOKKOS_LAMBDA(const LO i) {
      const LO cell = i / per_cell;
      int digits = i % per_cell;
      const auto box = grid.GetCellBBOX(cell);
      Real weight = 1;
      for (unsigned d = 0; d < Dim; ++d) {
        const int k = digits % n;
        digits /= n;
        xi(i, d) = box.center[d] + box.half_width[d] * nodes(k);
        weight *= box.half_width[d] * rule_weights(k);
      }
      weights(i) = weight;
    });
}

// The nodes and weights of the n-point Gauss-Legendre rule on [-1, 1].
std::pair<std::vector<Real>, std::vector<Real>> GaussLegendre(int n)
{
  std::vector<Real> nodes(n);
  std::vector<Real> weights(n);
  for (int i = 0; i < n; ++i) {
    Real x = std::cos(std::numbers::pi * (i + 0.75) / (n + 0.5));
    Real derivative = 1;
    for (int iteration = 0; iteration < 100; ++iteration) {
      Real previous = 1;
      Real current = x;
      for (int k = 2; k <= n; ++k) {
        const Real next = ((2 * k - 1) * x * current - (k - 1) * previous) / k;
        previous = current;
        current = next;
      }
      derivative = n * (x * current - previous) / (x * x - 1);
      const Real step = current / derivative;
      x -= step;
      if (std::abs(step) < 1e-15) {
        break;
      }
    }
    nodes[i] = x;
    weights[i] = 2 / ((1 - x * x) * derivative * derivative);
  }
  return {nodes, weights};
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

  /// J from the Gradient of the position at the parameter coordinates of the
  /// points, and the given metric there or the metric J^T G J the position
  /// induces; computed on the first call (like the rest of a BoundPoints, not
  /// thread-safe).
  [[nodiscard]] detail::GeometricFactors Factors(FactorsKey) const override
  {
    return Evaluate();
  }

  /// @return the Pushforward of the Jacobian: parameter-chart components to
  ///         components in the geometry's chart, at the points
  [[nodiscard]] const ValueTransform& Pushforward() const
  {
    const auto& factors = Evaluate();
    if (!pushforward_) {
      pushforward_.emplace(detail::Pushforward(
        factors.Jacobian(),
        CoordinateView<DeviceMemorySpace>(position_.ParameterChart(),
                                          MakeConstRank2View(xi_)),
        CoordinateView<DeviceMemorySpace>(position_.Chart(),
                                          MakeConstRank2View(mapped_))));
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
    factors_.reset();
    pushforward_.reset();
  }

private:
  // the position and its Jacobian at the parameter coordinates, and the
  // factors from them
  const detail::GeometricFactors& Evaluate() const
  {
    if (factors_) {
      return *factors_;
    }
    auto e = position_.Evaluate(MakeConstRank2View(xi_));
    const auto n = xi_.extent(0);
    const int dp = position_.ParameterChart()->Dimension();
    const int dt = position_.Chart()->Dimension();
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
      const auto chart = position_.Chart()->Bind(MakeConstRank2View(e.mapped));
      InduceMetric(
        View<3, const Real, DeviceMemorySpace>(e.jacobian.data(), n, dp, dt),
        *chart, metric, volume);
      // the chart's bound metric is released on return
      Kokkos::fence("EmbeddedBound::Evaluate");
    }
    mapped_ = e.mapped;
    return factors_.emplace(static_cast<LO>(n), e.jacobian,
                            std::make_shared<const BoundCoordinateSystem>(
                              static_cast<LO>(n), metric, volume));
  }

  Position position_;
  std::shared_ptr<const Function<Real>> metric_;
  std::shared_ptr<const BoundPoints> inner_;
  DeviceView<Real**> xi_;
  mutable DeviceView<Real**> mapped_;
  mutable std::optional<detail::GeometricFactors> factors_;
  mutable std::optional<ValueTransform> pushforward_;
};

// The parameter geometry and the position, whatever the kind of the
// parameter geometry; the generic part of the embedded geometry.
class EmbeddedGeometryBase : public Geometry
{
public:
  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>&
  GetCoordinateSystem() const noexcept override
  {
    return chart_;
  }
  [[nodiscard]] int TopologicalDimension() const noexcept override
  {
    return parameter_->GetCoordinateSystem()->Dimension();
  }
  [[nodiscard]] std::shared_ptr<const CoordinateSystem> GetParameterSystem()
    const noexcept override
  {
    return parameter_->GetCoordinateSystem();
  }
  [[nodiscard]] std::shared_ptr<const Function<Real>> GetPosition()
    const noexcept override
  {
    return position_.Get();
  }
  [[nodiscard]] const std::shared_ptr<const Geometry>& ParameterGeometry()
    const noexcept
  {
    return parameter_;
  }
  [[nodiscard]] std::shared_ptr<const Function<Real>> GetMetric()
    const noexcept override
  {
    return metric_;
  }
  [[nodiscard]] const Position& PositionOf() const noexcept
  {
    return position_;
  }

protected:
  EmbeddedGeometryBase(std::shared_ptr<const Geometry> parameter,
                       Position position,
                       std::shared_ptr<const Function<Real>> metric)
    : parameter_(std::move(parameter)),
      position_(std::move(position)),
      metric_(std::move(metric)),
      chart_(position_.Chart())
  {
  }

  // The parameter coordinates of the points, tagged with the parameter chart.
  [[nodiscard]] CoordinateView<DeviceMemorySpace> Parametric(
    DeviceView<const Real**> xi) const
  {
    return CoordinateView<DeviceMemorySpace>(GetParameterSystem(),
                                             MakeConstRank2View(xi));
  }

  std::shared_ptr<const Geometry> parameter_;
  Position position_;
  std::shared_ptr<const Function<Real>> metric_;
  std::shared_ptr<const CoordinateSystem> chart_;
};

template <unsigned Dim>
class EmbeddedGeometry final : public EmbeddedGeometryBase
{
public:
  EmbeddedGeometry(
    std::shared_ptr<const uniform_grid::UniformGridGeometry<Dim>> parameter,
    Position position, const EmbeddingOptions& options)
    : EmbeddedGeometryBase(parameter, std::move(position), options.metric),
      grid_(parameter->GetGrid()),
      options_(options.search)
  {
    BuildSearch(options.symmetry);
  }

  struct Located
  {
    DeviceView<Real**> xi;      // [point][parameter dimension]
    DeviceView<bool*> found;    // [point]
    DeviceView<Real**> reduced; // [point][chart dimension]: the points
                                // reduced by the periods and symmetry
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

  /// The tensor Gauss-Legendre rule of the parameter cells, exact for
  /// polynomials of degree `options.order` in the parameters, with the
  /// weights of the physical measure: the rule's weight times the volume
  /// factor of the geometric factors at the points.
  [[nodiscard]] Quadrature CreateQuadrature(
    const QuadratureOptions& options) const override
  {
    if (options.samples_per_cell > 0) {
      throw pcms_error("EmbeddedGeometry::CreateQuadrature: only the "
                       "deterministic rule is supported");
    }
    if (options.order < 1) {
      throw pcms_error("EmbeddedGeometry::CreateQuadrature: the order must be "
                       "positive");
    }
    const int per_axis = (options.order + 2) / 2;
    const auto [nodes_host, weights_host] = GaussLegendre(per_axis);
    DeviceView<Real*> nodes("gauss_nodes", per_axis);
    DeviceView<Real*> rule("gauss_weights", per_axis);
    Kokkos::deep_copy(nodes,
                      Kokkos::View<const Real*, HostMemorySpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                        nodes_host.data(), per_axis));
    Kokkos::deep_copy(rule,
                      Kokkos::View<const Real*, HostMemorySpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                        weights_host.data(), per_axis));
    size_t num_points = grid_.GetNumCells();
    for (unsigned d = 0; d < Dim; ++d) {
      num_points *= per_axis;
    }
    DeviceView<Real**> xi("quadrature_xi", num_points, Dim);
    DeviceView<Real*> weights("quadrature_weights", num_points);
    FillCellRule<Dim>(grid_, nodes, rule, xi, weights);
    auto bound = Bind(Parametric(xi));
    pcms::detail::ScaleByVolumeFactor(weights, *bound);
    return Quadrature{std::move(bound), weights, {}};
  }

protected:
  [[nodiscard]] std::shared_ptr<BoundPoints> BindImpl(
    const BindRequest& request) const override;

private:
  // Bins the images of the cells: their bounding boxes, from a lattice of
  // sample points of each cell, padded.
  void BuildSearch(const std::optional<TranslationSymmetry>& symmetry)
  {
    const int dt = GetCoordinateSystem()->Dimension();
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
      bins_.period[j] = GetCoordinateSystem()->Period(j).value_or(0);
      if (symmetry && symmetry->axis == j) {
        bins_.period[j] = symmetry->length;
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
  EmbeddedBinding(std::shared_ptr<const EmbeddedGeometry<Dim>> geometry,
                  const BindRequest& request)
    : EmbeddedBound(geometry, request, geometry->PositionOf(),
                    geometry->GetMetric()),
      geometry_(std::move(geometry))
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
    DeviceView<bool*> found("embedded_found", n);
    if (parametric) {
      DeviceView<Real**> xi("embedded_xi", n, Dim);
      Kokkos::deep_copy(xi, points);
      SetParameters(*geometry_->ParameterGeometry(), xi, Policy());
      MarkFound(Inner()->FoundPoints(), found);
    } else {
      const auto located = geometry_->Locate(MakeConstRank2View(points));
      if (GetBoundMap() == nullptr) {
        ReplaceLocatedPoints(located.reduced);
      }
      SetParameters(*geometry_->ParameterGeometry(), located.xi,
                    OutOfBoundsPolicy{OutOfBoundsMode::NEAREST_BOUNDARY});
      found = located.found;
    }
    SetFound(found);
  }

  std::shared_ptr<const EmbeddedGeometry<Dim>> geometry_;
};

template <unsigned Dim>
std::shared_ptr<BoundPoints> EmbeddedGeometry<Dim>::BindImpl(
  const BindRequest& request) const
{
  return std::make_shared<EmbeddedBinding<Dim>>(
    std::static_pointer_cast<const EmbeddedGeometry<Dim>>(shared_from_this()),
    request);
}

// Evaluates a field of a space on the parameter geometry at the points of an
// embedded geometry: at their parameter coordinates. Vectors and covectors
// in parameter-chart components come out in the geometry's chart, through
// the Pushforward of the position's Jacobian.
class EmbeddedEvaluator final : public PointEvaluator<Real>
{
public:
  EmbeddedEvaluator(std::shared_ptr<const EmbeddedBound> bound,
                    std::unique_ptr<PointEvaluator<Real>> inner)
    : PointEvaluator<Real>(bound, DifferentialOperator::Value,
                           inner->AcceptedLayout()),
      bound_(std::move(bound)),
      inner_(std::move(inner))
  {
  }

  [[nodiscard]] ValueBasis OutputBasis(const ValueBasis& stored) const override
  {
    const ValueBasis inner_out = inner_->OutputBasis(stored);
    if (!InParameterChart(inner_out)) {
      return inner_out;
    }
    return bound_->Pushforward().OutputBasis(inner_out);
  }

  void EvaluateImpl(const Field<Real>& field,
                    ValueView<Real, DeviceMemorySpace> values) const override
  {
    const ValueBasis inner_out =
      inner_->OutputBasis(field.GetData().GetValueBasis());
    if (!InParameterChart(inner_out)) {
      inner_->Evaluate(field, values);
      return;
    }
    const auto out = values.GetValues();
    const int din = inner_out.system->Dimension();
    const int dout = values.GetBasis().system->Dimension();
    // in place, unless the chart has another dimension
    DeviceView<Real**> scratch;
    if (din != dout) {
      scratch = DeviceView<Real**>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "parameter_values"),
        out.extent(0), static_cast<std::size_t>(out.extent(1) / dout) * din);
    }
    const ValueView<Real, DeviceMemorySpace> source(
      inner_out, din != dout ? MakeRank2View(scratch) : out);
    inner_->Evaluate(field, source);
    bound_->Pushforward().Apply(source, values);
  }

private:
  [[nodiscard]] bool InParameterChart(const ValueBasis& basis) const
  {
    return basis.Rank() > 0 &&
           detail::SameCoordinateSystem(
             basis.system, bound_->GetGeometry()->GetParameterSystem());
  }

  std::shared_ptr<const EmbeddedBound> bound_;
  std::unique_ptr<PointEvaluator<Real>> inner_;
};

class EmbeddedEvaluatorFactory final : public PointEvaluatorFactory
{
public:
  EmbeddedEvaluatorFactory(std::shared_ptr<const Geometry> embedded,
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
      std::make_unique<EmbeddedEvaluator>(
        embedded, CreateUntransformedPointEvaluator<Real>(
                    *space_, embedded->Inner(), op)));
  }

private:
  std::shared_ptr<const Geometry> embedded_;
  std::shared_ptr<const FunctionSpace> space_;
};

} // namespace

std::shared_ptr<const CoordinateMap> CreateCoordinateMap(
  std::shared_ptr<const Function<Real>> position)
{
  return std::make_shared<const PositionMap>(
    Position(std::move(position), "CreateCoordinateMap"));
}

std::shared_ptr<const Geometry> CreateEmbeddedGeometry(
  std::shared_ptr<const Geometry> parameter_geometry,
  std::shared_ptr<const Function<Real>> position, EmbeddingOptions options)
{
  Position checked(std::move(position), "CreateEmbeddedGeometry");
  if (parameter_geometry == nullptr ||
      checked.Get()->GetSpace().GetGeometry() != parameter_geometry) {
    throw pcms_error("CreateEmbeddedGeometry: the position must be a Function "
                     "on the parameter geometry");
  }
  const auto& symmetry = options.symmetry;
  if (symmetry && (symmetry->length <= 0 || symmetry->axis < 0 ||
                   symmetry->axis >= checked.Chart()->Dimension())) {
    throw pcms_error("CreateEmbeddedGeometry: the symmetry must translate an "
                     "axis of the position's chart by a positive length");
  }
  if (const auto& metric = options.metric) {
    const ValueBasis& basis = metric->GetData().GetValueBasis();
    if (metric->GetSpace().GetGeometry() != parameter_geometry ||
        basis.Rank() != 2 ||
        !detail::SameCoordinateSystem(basis.system, checked.ParameterChart()) ||
        basis.variance[0] != basis.variance[1]) {
      throw pcms_error("CreateEmbeddedGeometry: the metric must be a Function "
                       "on the parameter geometry of rank 2 in the parameter "
                       "chart, with both indices covariant (g_ij) or both "
                       "contravariant (g^ij)");
    }
  }
  if (detail::SameCoordinateSystem(checked.ParameterChart(), checked.Chart())) {
    throw pcms_error("CreateEmbeddedGeometry: the parameter geometry's chart "
                     "must differ from the chart of the position (give the "
                     "parameter geometry a parameter chart)");
  }
  if (auto grid =
        std::dynamic_pointer_cast<const uniform_grid::UniformGridGeometry<2>>(
          parameter_geometry)) {
    return std::make_shared<EmbeddedGeometry<2>>(std::move(grid),
                                                 std::move(checked), options);
  }
  if (auto grid =
        std::dynamic_pointer_cast<const uniform_grid::UniformGridGeometry<3>>(
          parameter_geometry)) {
    return std::make_shared<EmbeddedGeometry<3>>(std::move(grid),
                                                 std::move(checked), options);
  }
  throw pcms_error("CreateEmbeddedGeometry: only uniform-grid parameter "
                   "geometries are supported");
}

std::shared_ptr<const FunctionSpace> CreateEmbeddedSpace(
  std::shared_ptr<const Geometry> embedded,
  std::shared_ptr<const FunctionSpace> space)
{
  const auto base =
    std::dynamic_pointer_cast<const EmbeddedGeometryBase>(embedded);
  if (space == nullptr || base == nullptr ||
      base->ParameterGeometry() != space->GetGeometry()) {
    throw pcms_error("CreateEmbeddedSpace: the geometry must be an embedded "
                     "geometry of the space's geometry");
  }
  auto layout = space->GetLayout();
  return std::make_shared<FunctionSpace>(
    std::move(layout),
    std::make_shared<EmbeddedEvaluatorFactory>(embedded, std::move(space)),
    embedded);
}

} // namespace pcms
