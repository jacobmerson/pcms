#include "pcms/field/embedding.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/detail/memory_spaces.hpp"
#include <cmath>
#include <string>

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
    DeviceView<Real**> gradient("embedding_gradient", n,
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

} // namespace

std::shared_ptr<const CoordinateMap> CreateEmbedding(
  std::shared_ptr<const Function<Real>> position,
  std::shared_ptr<const CoordinateSystem> target)
{
  return std::make_shared<const EmbeddingMap>(
    std::make_shared<EmbeddingData>(std::move(position), std::move(target)));
}

} // namespace pcms
