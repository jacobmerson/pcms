#include "pcms/field/embedding.hpp"
#include "pcms/backends/uniform_grid/uniform_grid_geometry.hpp"
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

// Dense helpers for the n x n systems of a parameter space (n <= 3), usable
// in kernels.
KOKKOS_INLINE_FUNCTION Real Determinant(const Real (&m)[3][3], int n)
{
  if (n == 1) {
    return m[0][0];
  }
  if (n == 2) {
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
  }
  return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
         m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
         m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

// Solves m x = b by Gaussian elimination with partial pivoting; b becomes x.
KOKKOS_INLINE_FUNCTION void Solve(Real (&m)[3][3], int n, Real (&b)[3])
{
  for (int c = 0; c < n; ++c) {
    int pivot = c;
    for (int r = c + 1; r < n; ++r) {
      if (Kokkos::fabs(m[r][c]) > Kokkos::fabs(m[pivot][c])) {
        pivot = r;
      }
    }
    for (int k = 0; k < n; ++k) {
      const Real t = m[c][k];
      m[c][k] = m[pivot][k];
      m[pivot][k] = t;
    }
    const Real t = b[c];
    b[c] = b[pivot];
    b[pivot] = t;
    for (int r = c + 1; r < n; ++r) {
      const Real f = m[r][c] / m[c][c];
      for (int k = c; k < n; ++k) {
        m[r][k] -= f * m[c][k];
      }
      b[r] -= f * b[c];
    }
  }
  for (int c = n - 1; c >= 0; --c) {
    for (int k = c + 1; k < n; ++k) {
      b[c] -= m[c][k] * b[k];
    }
    b[c] /= m[c][c];
  }
}

// jacobian(p, i, j) = d x_j / d xi_i from the gradient of the position, whose
// component j has the partials d x_j / d xi_i at columns j * dp + i.
void ReshapeJacobian(DeviceView<const Real**> gradient,
                     DeviceView<Real***> jacobian)
{
  const int dp = jacobian.extent(1);
  const int dt = jacobian.extent(2);
  detail::ForEach(
    "embedding_jacobian", static_cast<LO>(jacobian.extent(0)),
    KOKKOS_LAMBDA(const LO p) {
      for (int i = 0; i < dp; ++i) {
        for (int j = 0; j < dt; ++j) {
          jacobian(p, i, j) = gradient(p, j * dp + i);
        }
      }
    });
}

// The metric g = J G J^T of the parameter system and its volume factor:
// the physical volume element per unit of parameter volume, sqrt(det g) times
// the factor of the target system's volume element relative to its metric.
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
      volume(p) = Kokkos::sqrt(Determinant(g, dp)) * volume_target(p) /
                  Kokkos::sqrt(Determinant(g_t, dt));
    });
}

// Re-expresses vectors (components on d x / d xi: x_j = sum_i J_ij v_i) or
// covectors (J^-1: the partials dxi_i = sum_j J_ij dx_j are solved for dx_j)
// of the parameter system in place; J is square.
void TransformComponents(View<3, const Real, DeviceMemorySpace> jacobian,
                         Rank2View<Real, DeviceMemorySpace> values,
                         bool covariant)
{
  const int n = jacobian.extent(1);
  detail::ForEach(
    "embedding_transform_values", static_cast<LO>(values.extent(0)),
    KOKKOS_LAMBDA(const LO p) {
      Real m[3][3] = {};
      Real v[3] = {};
      Real out[3] = {};
      for (int i = 0; i < n; ++i) {
        v[i] = values(p, i);
        for (int j = 0; j < n; ++j) {
          m[i][j] = jacobian(p, i, j);
        }
      }
      if (covariant) {
        Solve(m, n, v);
        for (int j = 0; j < n; ++j) {
          out[j] = v[j];
        }
      } else {
        for (int j = 0; j < n; ++j) {
          for (int i = 0; i < n; ++i) {
            out[j] += m[i][j] * v[i];
          }
        }
      }
      for (int j = 0; j < n; ++j) {
        values(p, j) = out[j];
      }
    });
}

struct Evaluation
{
  DeviceView<Real**> mapped;    // [point][target dimension]
  DeviceView<Real***> jacobian; // [point][parameter][target]
};

// What an embedding is: the position function and the target system. Shared
// by the map and its parameter system.
class EmbeddingData
{
public:
  EmbeddingData(std::shared_ptr<const Function<Real>> position,
                std::shared_ptr<const CoordinateSystem> target)
    : position_(std::move(position)), target_(std::move(target))
  {
    if (position_ == nullptr || target_ == nullptr) {
      throw pcms_error("CreateEmbedding: the position and the target system "
                       "must not be null");
    }
    if (!position_->GetSpace().Supports(DifferentialOperator::Gradient)) {
      throw pcms_error("CreateEmbedding: the position's space does not "
                       "evaluate gradients");
    }
    // the gradient of a field with several components is not shaped
    // [component][derivative] unless the field is a vector
    if (position_->GetData().GetValueBasis().Rank() != 1) {
      throw pcms_error("CreateEmbedding: the position must be a vector "
                       "(values::Vector) of the parameter geometry's "
                       "coordinate system");
    }
    if (position_->GetLayout().GetNumComponents() != target_->Dimension()) {
      throw pcms_error("CreateEmbedding: the parameter space and the target "
                       "system must have the same dimension");
    }
    if (ParameterDimension() > 3 || target_->Dimension() > 3) {
      throw pcms_error("CreateEmbedding: dimensions above 3 are not "
                       "supported");
    }
  }

  [[nodiscard]] const Function<Real>& Position() const { return *position_; }
  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>& Target() const
  {
    return target_;
  }
  [[nodiscard]] int ParameterDimension() const
  {
    return position_->GetSpace().GetCoordinateSystem()->Dimension();
  }

  /// The position and its Jacobian at parameter points; points outside the
  /// parameter geometry use its nearest cell.
  [[nodiscard]] Evaluation Evaluate(
    Rank2View<const Real, DeviceMemorySpace> xi) const
  {
    const FunctionSpace& space = position_->GetSpace();
    const auto geometry = space.GetGeometry();
    const auto n = xi.extent(0);
    const int dp = ParameterDimension();
    const int dt = target_->Dimension();
    const auto bound = geometry->Bind(
      CoordinateView<DeviceMemorySpace>(geometry->GetCoordinateSystem(), xi),
      OutOfBoundsPolicy{OutOfBoundsMode::NEAREST_BOUNDARY});
    const ValueBasis basis = position_->GetData().GetValueBasis();
    Evaluation e{DeviceView<Real**>("embedding_mapped", n, dt),
                 DeviceView<Real***>("embedding_jacobian", n, dp, dt)};
    space.CreatePointEvaluator<Real>(bound)->Evaluate(
      *position_,
      ValueView<Real, DeviceMemorySpace>(basis, MakeRank2View(e.mapped)));
    DeviceView<Real**> gradient("embedding_gradient", n, dt * dp);
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
  std::shared_ptr<const CoordinateSystem> target_;
};

// The parameter system of an embedding: coordinates xi of the parameter
// space, with the metric induced by the embedding.
class ParameterSystem final : public CoordinateSystem
{
public:
  explicit ParameterSystem(std::shared_ptr<const EmbeddingData> data)
    : data_(std::move(data))
  {
  }

  [[nodiscard]] std::string_view Kind() const noexcept override
  {
    return "parameter";
  }
  [[nodiscard]] int Dimension() const noexcept override
  {
    return data_->ParameterDimension();
  }
  [[nodiscard]] bool IsOrthogonal() const noexcept override { return false; }
  [[nodiscard]] bool IsConstantMetric() const noexcept override
  {
    return false;
  }
  [[nodiscard]] std::unique_ptr<BoundCoordinateSystem> Bind(
    Rank2View<const Real, DeviceMemorySpace> points) const override
  {
    const auto e = data_->Evaluate(points);
    const auto target = data_->Target()->Bind(MakeConstRank2View(e.mapped));
    const auto n = points.extent(0);
    const int dp = Dimension();
    DeviceView<Real***> metric("induced_metric", n, dp, dp);
    DeviceView<Real*> volume("induced_volume_factor", n);
    InduceMetric(View<3, const Real, DeviceMemorySpace>(
                   e.jacobian.data(), n, dp, data_->Target()->Dimension()),
                 *target, metric, volume);
    return std::make_unique<BoundCoordinateSystem>(n, metric, volume);
  }
  [[nodiscard]] bool operator==(
    const CoordinateSystem& other) const noexcept override
  {
    const auto* p = dynamic_cast<const ParameterSystem*>(&other);
    return p != nullptr && p->data_ == data_;
  }

private:
  std::shared_ptr<const EmbeddingData> data_;
};

class BoundEmbedding final : public BoundCoordinateMap
{
public:
  BoundEmbedding(std::shared_ptr<const CoordinateSystem> source,
                 std::shared_ptr<const CoordinateSystem> target,
                 Evaluation evaluation)
    : BoundCoordinateMap(target, evaluation.mapped),
      source_(std::move(source)),
      target_(std::move(target)),
      jacobian_(evaluation.jacobian)
  {
  }

  [[nodiscard]] View<3, const Real, DeviceMemorySpace> Jacobian() const override
  {
    return View<3, const Real, DeviceMemorySpace>(
      jacobian_.data(), jacobian_.extent(0), jacobian_.extent(1),
      jacobian_.extent(2));
  }

  [[nodiscard]] ValueBasis OutputBasis(const ValueBasis& stored) const override
  {
    if (stored.Rank() != 1 ||
        !detail::SameCoordinateSystem(stored.system, source_) ||
        stored.scaling != ComponentScaling::Natural) {
      throw pcms_error("Embedding: only the natural components of vectors "
                       "and covectors of the parameter system can be "
                       "re-expressed");
    }
    // the natural components of a system with a constant metric are physical
    return ValueBasis{target_,
                      target_->IsConstantMetric() ? ComponentScaling::Physical
                                                  : ComponentScaling::Natural,
                      stored.variance, stored.parity};
  }

  ValueView<Real, DeviceMemorySpace> TransformValues(
    ValueView<Real, DeviceMemorySpace> values) const override
  {
    const ValueBasis output = ValidateTransformValues("Embedding", values);
    if (jacobian_.extent(1) != jacobian_.extent(2)) {
      throw pcms_error("Embedding: values can be re-expressed only where the "
                       "parameter space and the target have one dimension");
    }
    TransformComponents(Jacobian(), values.GetValues(),
                        values.GetBasis().variance == values::Covector);
    return ValueView<Real, DeviceMemorySpace>(output, values.GetValues());
  }

private:
  std::shared_ptr<const CoordinateSystem> source_;
  std::shared_ptr<const CoordinateSystem> target_;
  DeviceView<Real***> jacobian_;
};

class EmbeddingMap final : public CoordinateMap
{
public:
  EmbeddingMap(std::shared_ptr<const EmbeddingData> data)
    : CoordinateMap(std::make_shared<const ParameterSystem>(data),
                    data->Target()),
      data_(std::move(data))
  {
  }

  [[nodiscard]] const EmbeddingData& Data() const noexcept { return *data_; }

protected:
  [[nodiscard]] std::unique_ptr<BoundCoordinateMap> BindImpl(
    const CoordinateView<DeviceMemorySpace>& query_points) const override
  {
    return std::make_unique<BoundEmbedding>(
      GetSourceCoordinateSystem(), GetTargetCoordinateSystem(),
      data_->Evaluate(query_points.GetValues()));
  }

private:
  std::shared_ptr<const EmbeddingData> data_;
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
      if (Determinant(m, Dim) <= 1e-12 * reference) {
        for (unsigned a = 0; a < Dim; ++a) {
          regular[a][a] += 1e-6 * scale + 1e-300;
        }
      }
      Solve(regular, Dim, b);
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

// The determinant factor of a Jacobian [point][parameter][target]: det J, or
// sqrt(det J J^T) where the parameter space has fewer dimensions than the
// target.
void JacobianDeterminants(View<3, const Real, DeviceMemorySpace> jacobian,
                          DeviceView<Real*> determinant)
{
  const int dp = jacobian.extent(1);
  const int dt = jacobian.extent(2);
  detail::ForEach(
    "embedded_determinants", static_cast<LO>(jacobian.extent(0)),
    KOKKOS_LAMBDA(const LO p) {
      Real m[3][3] = {};
      for (int a = 0; a < dp; ++a) {
        for (int b = 0; b < dp; ++b) {
          for (int j = 0; j < dt; ++j) {
            m[a][b] += jacobian(p, a, j) * jacobian(p, b, j);
          }
        }
      }
      determinant(p) = Kokkos::sqrt(Determinant(m, dp));
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

// xi of the listed rows of `xi`.
void GatherRows(DeviceView<const Real**> xi, DeviceView<const LO*> rows,
                DeviceView<Real**> out)
{
  const int dim = xi.extent(1);
  detail::ForEach(
    "embedded_gather", static_cast<LO>(rows.extent(0)),
    KOKKOS_LAMBDA(const LO k) {
      for (int d = 0; d < dim; ++d) {
        out(k, d) = xi(rows(k), d);
      }
    });
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

void ScaleWeights(DeviceView<Real*> weights, DeviceView<const Real*> factor)
{
  detail::ForEach(
    "embedded_scale_weights", static_cast<LO>(weights.extent(0)),
    KOKKOS_LAMBDA(const LO i) { weights(i) *= factor(i); });
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
// coordinates of the points, bound on the parameter geometry.
class EmbeddedBound : public BoundPoints
{
public:
  /// @return the points bound on the parameter geometry, at the parameter
  ///         coordinates: what the evaluators of its spaces take
  [[nodiscard]] const std::shared_ptr<const BoundPoints>& Inner() const noexcept
  {
    return inner_;
  }

  /// The Jacobian of the embedding at the found points; computed on the
  /// first call (like the rest of a BoundPoints, not thread-safe).
  [[nodiscard]] DifferentialView Differential() const override
  {
    if (evaluation_ == nullptr) {
      const auto found = FoundPoints();
      DeviceView<Real**> xi("embedded_found_xi", found.extent(0),
                            xi_.extent(1));
      GatherRows(xi_, found, xi);
      evaluation_ = embedding_->Bind(CoordinateView<DeviceMemorySpace>(
        embedding_->GetSourceCoordinateSystem(), MakeConstRank2View(xi)));
      determinant_ = DeviceView<Real*>("embedded_determinant", found.extent(0));
      JacobianDeterminants(evaluation_->Jacobian(), determinant_);
    }
    return DifferentialView{evaluation_->Jacobian(),
                            MakeRank1View(determinant_), false};
  }

protected:
  EmbeddedBound(std::shared_ptr<const Geometry> geometry,
                const BindRequest& request,
                std::shared_ptr<const CoordinateMap> embedding)
    : BoundPoints(std::move(geometry), request),
      embedding_(std::move(embedding))
  {
  }

  // Takes the parameter coordinates of the points: `xi`, bound on `parameter`
  // with the points `found` among them found.
  void SetParameters(const Geometry& parameter, DeviceView<Real**> xi,
                     OutOfBoundsPolicy policy)
  {
    xi_ = std::move(xi);
    inner_ = parameter.Bind(
      CoordinateView<DeviceMemorySpace>(parameter.GetCoordinateSystem(),
                                        MakeConstRank2View(xi_)),
      policy);
    evaluation_ = nullptr;
  }

  DeviceView<const Real**> Parameters() const { return xi_; }

private:
  std::shared_ptr<const CoordinateMap> embedding_;
  std::shared_ptr<const BoundPoints> inner_;
  DeviceView<Real**> xi_;
  mutable std::unique_ptr<BoundCoordinateMap> evaluation_;
  mutable DeviceView<Real*> determinant_;
};

// The parameter geometry and the embedding, whatever the kind of the
// parameter geometry; the generic part of the embedded geometry.
class EmbeddedGeometryBase : public Geometry
{
public:
  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>&
  GetCoordinateSystem() const noexcept override
  {
    return embedding_->GetTargetCoordinateSystem();
  }
  [[nodiscard]] int TopologicalDimension() const noexcept override
  {
    return parameter_->GetCoordinateSystem()->Dimension();
  }
  [[nodiscard]] std::shared_ptr<const CoordinateSystem> GetParameterSystem()
    const noexcept override
  {
    return embedding_->GetSourceCoordinateSystem();
  }
  [[nodiscard]] std::shared_ptr<const CoordinateMap> GetEmbedding()
    const noexcept override
  {
    return embedding_;
  }
  [[nodiscard]] const std::shared_ptr<const Geometry>& ParameterGeometry()
    const noexcept
  {
    return parameter_;
  }

protected:
  EmbeddedGeometryBase(std::shared_ptr<const Geometry> parameter,
                       std::shared_ptr<const CoordinateMap> embedding)
    : parameter_(std::move(parameter)), embedding_(std::move(embedding))
  {
  }

  // The parameter coordinates of the points, tagged for the embedding.
  [[nodiscard]] CoordinateView<DeviceMemorySpace> Parametric(
    DeviceView<const Real**> xi) const
  {
    return CoordinateView<DeviceMemorySpace>(GetParameterSystem(),
                                             MakeConstRank2View(xi));
  }

  std::shared_ptr<const Geometry> parameter_;
  std::shared_ptr<const CoordinateMap> embedding_;
};

template <unsigned Dim>
class EmbeddedGeometry final : public EmbeddedGeometryBase
{
public:
  EmbeddedGeometry(
    std::shared_ptr<const uniform_grid::UniformGridGeometry<Dim>> parameter,
    std::shared_ptr<const CoordinateMap> embedding, SearchOptions options,
    std::optional<TranslationSymmetry> symmetry)
    : EmbeddedGeometryBase(parameter, std::move(embedding)),
      grid_(parameter->GetGrid()),
      options_(options)
  {
    BuildSearch(symmetry);
  }

  struct Located
  {
    DeviceView<Real**> xi;      // [point][parameter dimension]
    DeviceView<bool*> found;    // [point]
    DeviceView<Real**> reduced; // [point][target dimension]: the points
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
          const auto evaluated = embedding_->Bind(Parametric(located.xi));
          const LO stepped = NewtonStep<Dim>(
            grid_, residual_tolerance_, iteration < MaxNewtonIterations, target,
            evaluated->Mapped(), evaluated->Jacobian(), active, converged, cell,
            located.xi);
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
  /// weights of the physical measure: the rule's weight times |det J| and
  /// the volume factor of the target system.
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
    const auto evaluated = embedding_->Bind(Parametric(xi));
    DeviceView<Real*> determinant("quadrature_determinant", num_points);
    JacobianDeterminants(evaluated->Jacobian(), determinant);
    ScaleWeights(weights, determinant);
    pcms::detail::ScaleByVolumeFactor(weights, *GetCoordinateSystem(),
                                      MakeConstRank2View(evaluated->Mapped()));
    return Quadrature{Bind(Parametric(xi)), weights, {}};
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
      HostMemorySpace(), embedding_->Bind(Parametric(device))->Mapped());

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
    : EmbeddedBound(geometry, request, geometry->GetEmbedding()),
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
// embedded geometry: at their parameter coordinates.
class EmbeddedEvaluator final : public PointEvaluator<Real>
{
public:
  EmbeddedEvaluator(std::shared_ptr<const BoundPoints> bound,
                    std::unique_ptr<PointEvaluator<Real>> inner)
    : PointEvaluator<Real>(std::move(bound), DifferentialOperator::Value,
                           inner->AcceptedLayout()),
      inner_(std::move(inner))
  {
  }

  void EvaluateImpl(const Field<Real>& field,
                    ValueView<Real, DeviceMemorySpace> values) const override
  {
    inner_->Evaluate(field, values);
  }

private:
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
        std::move(bound), CreateUntransformedPointEvaluator<Real>(
                            *space_, embedded->Inner(), op)));
  }

private:
  std::shared_ptr<const Geometry> embedded_;
  std::shared_ptr<const FunctionSpace> space_;
};

} // namespace

std::shared_ptr<const CoordinateMap> CreateEmbedding(
  std::shared_ptr<const Function<Real>> position,
  std::shared_ptr<const CoordinateSystem> target)
{
  return std::make_shared<const EmbeddingMap>(
    std::make_shared<EmbeddingData>(std::move(position), std::move(target)));
}

std::shared_ptr<const Geometry> CreateEmbeddedGeometry(
  std::shared_ptr<const Geometry> parameter_geometry,
  std::shared_ptr<const CoordinateMap> embedding, SearchOptions options,
  std::optional<TranslationSymmetry> symmetry)
{
  if (symmetry &&
      (symmetry->length <= 0 || symmetry->axis < 0 || embedding == nullptr ||
       symmetry->axis >= embedding->GetTargetCoordinateSystem()->Dimension())) {
    throw pcms_error("CreateEmbeddedGeometry: the symmetry must translate an "
                     "axis of the target system by a positive length");
  }
  const auto* map = dynamic_cast<const EmbeddingMap*>(embedding.get());
  if (parameter_geometry == nullptr || map == nullptr ||
      map->Data().Position().GetSpace().GetGeometry() != parameter_geometry) {
    throw pcms_error("CreateEmbeddedGeometry: the embedding must be a "
                     "CreateEmbedding of a function on the parameter "
                     "geometry");
  }
  if (auto grid =
        std::dynamic_pointer_cast<const uniform_grid::UniformGridGeometry<2>>(
          parameter_geometry)) {
    return std::make_shared<EmbeddedGeometry<2>>(
      std::move(grid), std::move(embedding), options, symmetry);
  }
  if (auto grid =
        std::dynamic_pointer_cast<const uniform_grid::UniformGridGeometry<3>>(
          parameter_geometry)) {
    return std::make_shared<EmbeddedGeometry<3>>(
      std::move(grid), std::move(embedding), options, symmetry);
  }
  throw pcms_error("CreateEmbeddedGeometry: only uniform-grid parameter "
                   "geometries are supported");
}

std::shared_ptr<const FunctionSpace> CreateEmbeddedSpace(
  std::shared_ptr<const FunctionSpace> space,
  std::shared_ptr<const Geometry> embedded)
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
