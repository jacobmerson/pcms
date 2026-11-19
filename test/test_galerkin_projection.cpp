#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <pcms/field/function_spaces.hpp>
#include <pcms/transfer/projection.hpp>
#include <pcms/backends/omega_h/omega_h.hpp>
#include <pcms/transfer/omega_h_conservative_projection.hpp>
#include <pcms/transfer/omega_h_control_variate_projection.hpp>
#include "field_test_utils.hpp"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace
{

using Host2D = Kokkos::View<pcms::Real**, pcms::HostMemorySpace>;
using Device2D = Kokkos::View<pcms::Real**, pcms::DeviceMemorySpace>;

// The generic adjoint test: <E x, y> == <x, E^T y> for random DOFs x of the
// space and random point values y, for any evaluator that SupportsTranspose.
// The space must be unghosted (serial), so that the holders are the owned ones.
void CheckAdjoint(const pcms::FunctionSpace& space,
                  const pcms::PointEvaluator<pcms::Real>& evaluator,
                  unsigned seed)
{
  REQUIRE(evaluator.SupportsTranspose());
  const int num_components = space.GetLayout()->GetNumComponents();
  const int num_holders = space.GetLayout()->GetNumDofHolder();
  const int num_points = evaluator.NumPoints();
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);

  Host2D x("x", num_holders, num_components);
  Host2D y("y", num_points, num_components);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x.data()[i] = uniform(rng);
  }
  for (std::size_t i = 0; i < y.size(); ++i) {
    y.data()[i] = uniform(rng);
  }

  auto field = space.CreateFunction<pcms::Real>();
  field.SetDOFHolderDataHost(
    pcms::ValueView<const pcms::Real, pcms::HostMemorySpace>(
      field.GetData().GetValueBasis(),
      pcms::Rank2View<const pcms::Real, pcms::HostMemorySpace>(
        x.data(), num_holders, num_components)));

  Device2D e_x("Ex", num_points, num_components);
  evaluator.Evaluate(field, pcms::ValueView<pcms::Real, pcms::DeviceMemorySpace>(
                              field.GetData().GetValueBasis(),
                              pcms::MakeRank2View(e_x)));

  const auto y_device =
    Kokkos::create_mirror_view_and_copy(pcms::DeviceMemorySpace{}, y);
  Device2D transposed("ETy", num_holders, num_components);
  evaluator.EvaluateTranspose(pcms::MakeConstRank2View(y_device),
                              pcms::MakeRank2View(transposed));

  const auto e_x_host =
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace{}, e_x);
  const auto transposed_host =
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace{}, transposed);
  // (a mirror aliases the view in a host build: keep a copy of E^T y)
  const std::vector<double> once(transposed_host.data(),
                                 transposed_host.data() + x.size());
  double lhs = 0, rhs = 0, scale = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    lhs += e_x_host.data()[i] * y.data()[i];
    scale += std::abs(e_x_host.data()[i] * y.data()[i]);
  }
  for (std::size_t i = 0; i < x.size(); ++i) {
    rhs += x.data()[i] * transposed_host.data()[i];
  }
  CAPTURE(lhs, rhs);
  CHECK(std::abs(lhs - rhs) <= 1e-12 * std::max(1.0, scale));

  // accumulates: a second call doubles the result
  evaluator.EvaluateTranspose(pcms::MakeConstRank2View(y_device),
                              pcms::MakeRank2View(transposed));
  const auto twice =
    Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace{}, transposed);
  for (std::size_t i = 0; i < x.size(); ++i) {
    CHECK(twice.data()[i] ==
          Catch::Approx(2 * once[i]).margin(1e-14));
  }
}

std::vector<pcms::Real> RandomPoints(int dim, int n, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uniform(0.01, 0.99);
  std::vector<pcms::Real> points(static_cast<std::size_t>(n) * dim);
  for (auto& p : points) {
    p = uniform(rng);
  }
  return points;
}

} // namespace

TEST_CASE("EvaluateTranspose is the adjoint of Evaluate (Omega_h Lagrange)",
          "[quadrature][adjoint]")
{
  Omega_h::Library lib;
  const int dim = GENERATE(2, 3);
  const int order = GENERATE(0, 1);
  const int num_components = GENERATE(1, 2);
  CAPTURE(dim, order, num_components);
  auto mesh = dim == 2 ? pcms::test::BuildSquareMesh(lib, 3)
                       : pcms::test::BuildUnitCube(lib, 2);
  auto geometry =
    pcms::omega_h::CreateGeometry(mesh, pcms::detail::DeferredCartesian());
  const auto space = pcms::CreateLagrangeSpace(
    geometry, order, num_components,
    pcms::LagrangeOptions{.backend = pcms::LagrangeBackend::OmegaH});
  const auto bound = pcms::test::Bind(*geometry, RandomPoints(dim, 200, 3));
  const auto evaluator = space->CreatePointEvaluator<pcms::Real>(bound);
  CheckAdjoint(*space, *evaluator, 11);
}

TEST_CASE("EvaluateTranspose: evaluators without it say so", "[adjoint]")
{
  Omega_h::Library lib;
  auto mesh = pcms::test::BuildSquareMesh(lib, 1);
  auto geometry =
    pcms::omega_h::CreateGeometry(mesh, pcms::detail::DeferredCartesian());
  const auto space = pcms::test::MakeP1Space(geometry);
  // the values of an evaluator that is gradient-valued are not a Value map
  const auto bound = pcms::test::Bind(*geometry, RandomPoints(2, 4, 1));
  const auto evaluator = space->CreatePointEvaluator<pcms::Real>(bound);
  Device2D values("v", 4, 1);
  Device2D wrong_shape("d", 3, 1);
  CHECK_THROWS_AS(
    evaluator->EvaluateTranspose(pcms::MakeConstRank2View(values),
                                 pcms::MakeRank2View(wrong_shape)),
    std::invalid_argument);
}

namespace
{

std::vector<double> FieldValues(const pcms::Field<pcms::Real>& field)
{
  const auto values = pcms::detail::FlattenToRank1View(
    field.GetDOFHolderDataHost().GetValues());
  return std::vector<double>(values.data_handle(),
                             values.data_handle() + values.size());
}

std::shared_ptr<const pcms::omega_h::OmegaHGeometry> MakeGeometry(
  Omega_h::Mesh& mesh)
{
  return std::dynamic_pointer_cast<const pcms::omega_h::OmegaHGeometry>(
    pcms::omega_h::CreateGeometry(mesh, pcms::detail::DeferredCartesian()));
}

} // namespace

TEST_CASE("CreateProjection with a reference-rule quadrature reproduces fields "
          "of the target space exactly",
          "[quadrature][projection]")
{
  Omega_h::Library lib;
  const int dim = GENERATE(2, 3);
  const int target_order = GENERATE(0, 1);
  CAPTURE(dim, target_order);
  auto source_mesh = dim == 2 ? pcms::test::BuildSquareMesh(lib, 5)
                              : pcms::test::BuildUnitCube(lib, 3);
  auto target_mesh = dim == 2 ? pcms::test::BuildSquareMesh(lib, 3)
                              : pcms::test::BuildUnitCube(lib, 2);
  const auto source_geometry = MakeGeometry(source_mesh);
  const auto target_geometry = MakeGeometry(target_mesh);
  const auto source_space = pcms::test::MakeP1Space(source_geometry);
  const auto target_space =
    pcms::CreateLagrangeSpace(target_geometry, target_order, 1,
                              pcms::LagrangeOptions{
                                .backend = pcms::LagrangeBackend::OmegaH});
  auto source = source_space->CreateFunction<pcms::Real>();
  auto target = target_space->CreateFunction<pcms::Real>();
  pcms::test::SetFieldComponents(
    source, [](pcms::Real x, pcms::Real y, pcms::Real* out) {
      out[0] = 2.0;
    });
  pcms::ProjectionOptions options;
  options.quadrature_options.order = 2;
  auto projection = pcms::CreateProjection(source_space, target_space, options);
  projection->Apply(source, target);
  // a constant is in both spaces: the projection is the constant
  for (const double value : FieldValues(target)) {
    CHECK(value == Catch::Approx(2.0).margin(1e-10));
  }
}

TEST_CASE("CreateProjection projects several components at once",
          "[quadrature][projection]")
{
  Omega_h::Library lib;
  auto source_mesh = pcms::test::BuildSquareMesh(lib, 5);
  auto target_mesh = pcms::test::BuildSquareMesh(lib, 3);
  const auto source_geometry = MakeGeometry(source_mesh);
  const auto target_geometry = MakeGeometry(target_mesh);
  const auto options = pcms::LagrangeOptions{.backend =
                                               pcms::LagrangeBackend::OmegaH};
  const auto source_space =
    pcms::CreateLagrangeSpace(source_geometry, 1, 2, options);
  const auto target_space =
    pcms::CreateLagrangeSpace(target_geometry, 1, 2, options);
  auto source = source_space->CreateFunction<pcms::Real>();
  auto target = target_space->CreateFunction<pcms::Real>();
  pcms::test::SetFieldComponents(
    source, [](pcms::Real x, pcms::Real y, pcms::Real* out) {
      out[0] = x + y;
      out[1] = 2 * x - y + 0.5;
    });
  auto projection = pcms::CreateProjection(source_space, target_space, {});
  projection->Apply(source, target);
  const auto values = FieldValues(target);
  const auto coords = pcms::test::CopyCoordinatesToHost(
    target_space->GetLayout()->GetDOFHolderCoordinates().GetValues());
  REQUIRE(values.size() == 2 * coords.extent(0));
  for (std::size_t i = 0; i < coords.extent(0); ++i) {
    CHECK(values[2 * i] ==
          Catch::Approx(coords(i, 0) + coords(i, 1)).margin(1e-10));
    CHECK(values[2 * i + 1] ==
          Catch::Approx(2 * coords(i, 0) - coords(i, 1) + 0.5).margin(1e-10));
  }
}

TEST_CASE("the generic projection reproduces the intersection-integrator "
          "conservative projection",
          "[quadrature][equivalence]")
{
  Omega_h::Library lib;
  const int dim = GENERATE(2, 3);
  const int source_order = GENERATE(0, 1);
  const int target_order = GENERATE(0, 1);
  const auto mass_type =
    GENERATE(pcms::MassMatrixType::Consistent, pcms::MassMatrixType::Lumped);
  CAPTURE(dim, source_order, target_order, static_cast<int>(mass_type));
  auto source_mesh = dim == 2 ? pcms::test::BuildSquareMesh(lib, 5)
                              : pcms::test::BuildUnitCube(lib, 3);
  auto target_mesh = dim == 2 ? pcms::test::BuildUnitSquare(lib, 0)
                              : pcms::test::BuildUnitCube(lib, 2);
  const auto source_geometry = MakeGeometry(source_mesh);
  const auto target_geometry = MakeGeometry(target_mesh);
  const auto lagrange = pcms::LagrangeOptions{.backend =
                                                pcms::LagrangeBackend::OmegaH};
  const auto source_space =
    pcms::CreateLagrangeSpace(source_geometry, source_order, 1, lagrange);
  const auto target_space =
    pcms::CreateLagrangeSpace(target_geometry, target_order, 1, lagrange);
  auto source = source_space->CreateFunction<pcms::Real>();
  if (dim == 2) {
    pcms::test::SetField(source, OMEGA_H_LAMBDA(pcms::Real x, pcms::Real y) {
      return std::sin(3 * x) * (1 + y * y);
    });
  } else {
    pcms::test::SetField(
      source, OMEGA_H_LAMBDA(pcms::Real x, pcms::Real y, pcms::Real z) {
        return std::sin(3 * x) * (1 + y * z);
      });
  }

  auto expected = target_space->CreateFunction<pcms::Real>();
  pcms::transfer::OmegaHConservativeProjection old_projection(
    source_space, target_space, mass_type);
  old_projection.Apply(source, expected);

  auto actual = target_space->CreateFunction<pcms::Real>();
  auto projection = pcms::CreateConservativeProjection(
    source_space, target_space, {.mass_matrix_type = mass_type});
  projection->Apply(source, actual);

  const auto expected_values = FieldValues(expected);
  const auto actual_values = FieldValues(actual);
  REQUIRE(expected_values.size() == actual_values.size());
  double max_difference = 0, max_value = 0;
  for (std::size_t i = 0; i < expected_values.size(); ++i) {
    max_difference =
      std::max(max_difference, std::abs(expected_values[i] - actual_values[i]));
    max_value = std::max(max_value, std::abs(expected_values[i]));
  }
  CAPTURE(max_difference, max_value);
  CHECK(max_value > 0.1);
  CHECK(max_difference < 1e-12 * std::max(1.0, max_value));
}

TEST_CASE("the generic control-variate projection reproduces the old one with "
          "the same seed",
          "[quadrature][equivalence]")
{
  Omega_h::Library lib;
  const int dim = GENERATE(2, 3);
  const auto mass_type =
    GENERATE(pcms::MassMatrixType::Consistent, pcms::MassMatrixType::Lumped);
  CAPTURE(dim, static_cast<int>(mass_type));
  auto source_mesh = dim == 2 ? pcms::test::BuildSquareMesh(lib, 5)
                              : pcms::test::BuildUnitCube(lib, 3);
  auto target_mesh = dim == 2 ? pcms::test::BuildUnitSquare(lib, 0)
                              : pcms::test::BuildUnitCube(lib, 2);
  const auto source_space = pcms::test::MakeP1Space(MakeGeometry(source_mesh));
  const auto target_space = pcms::test::MakeP1Space(MakeGeometry(target_mesh));
  auto source = source_space->CreateFunction<pcms::Real>();
  if (dim == 2) {
    pcms::test::SetField(source, OMEGA_H_LAMBDA(pcms::Real x, pcms::Real y) {
      return std::sin(3 * x) * (1 + y * y);
    });
  } else {
    pcms::test::SetField(
      source, OMEGA_H_LAMBDA(pcms::Real x, pcms::Real y, pcms::Real z) {
        return std::sin(3 * x) * (1 + y * z);
      });
  }
  const int samples = 6;
  const std::uint64_t seed = 42;

  auto expected = target_space->CreateFunction<pcms::Real>();
  pcms::transfer::OmegaHControlVariateProjection old_projection(
    source_space, target_space, samples, seed, mass_type);
  old_projection.Apply(source, expected);

  auto actual = target_space->CreateFunction<pcms::Real>();
  auto projection = pcms::CreateControlVariateProjection(
    source_space, target_space,
    {.samples_per_element = samples,
     .seed = seed,
     .mass_matrix_type = mass_type});
  projection->Apply(source, actual);

  const auto expected_values = FieldValues(expected);
  const auto actual_values = FieldValues(actual);
  REQUIRE(expected_values.size() == actual_values.size());
  double max_difference = 0;
  for (std::size_t i = 0; i < expected_values.size(); ++i) {
    max_difference =
      std::max(max_difference, std::abs(expected_values[i] - actual_values[i]));
  }
  CAPTURE(max_difference);
  CHECK(max_difference < 1e-12);
}

// ---------------------------------------------------------------------------
// Matrix-free solve and conditioning
// ---------------------------------------------------------------------------

namespace
{

// The mass operator x -> E^T W E x of a P1 space on a quadrature, using only
// Evaluate and EvaluateTranspose.
class MatrixFreeMass
{
public:
  MatrixFreeMass(const pcms::FunctionSpace& space,
                 const pcms::Quadrature& quadrature)
    : space_(space),
      evaluator_(space.CreatePointEvaluator<pcms::Real>(quadrature.points)),
      weights_(Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace{},
                                                   quadrature.weights)),
      field_(space.CreateFunction<pcms::Real>()),
      size_(space.GetLayout()->GetNumDofHolder())
  {
  }

  int Size() const { return size_; }

  // y = E^T diag(scale) (E x + shift)
  std::vector<double> Apply(const std::vector<double>& x) const
  {
    field_.SetDOFHolderDataHost(
      pcms::ValueView<const pcms::Real, pcms::HostMemorySpace>(
        field_.GetData().GetValueBasis(),
        pcms::Rank2View<const pcms::Real, pcms::HostMemorySpace>(x.data(),
                                                                 size_, 1)));
    Device2D values("values", evaluator_->NumPoints(), 1);
    evaluator_->Evaluate(field_,
                         pcms::ValueView<pcms::Real, pcms::DeviceMemorySpace>(
                           field_.GetData().GetValueBasis(),
                           pcms::MakeRank2View(values)));
    return Transpose(values);
  }

  // E^T (W values), the values being one per point
  std::vector<double> Transpose(Device2D values) const
  {
    auto host = Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace{},
                                                    values);
    for (std::size_t p = 0; p < host.extent(0); ++p) {
      host(p, 0) *= weights_(p);
    }
    const auto weighted =
      Kokkos::create_mirror_view_and_copy(pcms::DeviceMemorySpace{}, host);
    Device2D dofs("dofs", size_, 1);
    evaluator_->EvaluateTranspose(pcms::MakeConstRank2View(weighted),
                                  pcms::MakeRank2View(dofs));
    const auto dofs_host =
      Kokkos::create_mirror_view_and_copy(pcms::HostMemorySpace{}, dofs);
    return std::vector<double>(dofs_host.data(), dofs_host.data() + size_);
  }

  const pcms::PointEvaluator<pcms::Real>& Evaluator() const
  {
    return *evaluator_;
  }

private:
  const pcms::FunctionSpace& space_;
  std::unique_ptr<pcms::PointEvaluator<pcms::Real>> evaluator_;
  Kokkos::View<const pcms::Real*, pcms::HostMemorySpace> weights_;
  mutable pcms::Field<pcms::Real> field_;
  int size_;
};

double Dot(const std::vector<double>& a, const std::vector<double>& b)
{
  double sum = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

// Conjugate gradients on a symmetric positive definite operator; returns the
// number of iterations to reach a relative residual of `tolerance`.
template <typename Operator>
int ConjugateGradients(const Operator& apply, const std::vector<double>& b,
                       std::vector<double>& x, double tolerance)
{
  x.assign(b.size(), 0.0);
  std::vector<double> r = b, p = b;
  double rr = Dot(r, r);
  const double target = tolerance * tolerance * Dot(b, b);
  int iterations = 0;
  while (rr > target && iterations < 1000) {
    const auto ap = apply(p);
    const double alpha = rr / Dot(p, ap);
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * ap[i];
    }
    const double rr_new = Dot(r, r);
    const double beta = rr_new / rr;
    for (std::size_t i = 0; i < p.size(); ++i) {
      p[i] = r[i] + beta * p[i];
    }
    rr = rr_new;
    ++iterations;
  }
  return iterations;
}

// Eigenvalues (min, max) of a dense symmetric matrix by cyclic Jacobi sweeps
std::pair<double, double> ExtremeEigenvalues(std::vector<std::vector<double>> a)
{
  const std::size_t n = a.size();
  for (int sweep = 0; sweep < 60; ++sweep) {
    double off = 0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        off += a[i][j] * a[i][j];
      }
    }
    if (off < 1e-28) {
      break;
    }
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = p + 1; q < n; ++q) {
        if (std::abs(a[p][q]) < 1e-300) {
          continue;
        }
        const double theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const double t =
          (theta >= 0 ? 1.0 : -1.0) /
          (std::abs(theta) + std::sqrt(theta * theta + 1));
        const double c = 1 / std::sqrt(t * t + 1), s = t * c;
        for (std::size_t k = 0; k < n; ++k) {
          const double akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (std::size_t k = 0; k < n; ++k) {
          const double apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
      }
    }
  }
  double low = a[0][0], high = a[0][0];
  for (std::size_t i = 0; i < n; ++i) {
    low = std::min(low, a[i][i]);
    high = std::max(high, a[i][i]);
  }
  return {low, high};
}

double Factorial(int n)
{
  return n <= 1 ? 1.0 : n * Factorial(n - 1);
}

} // namespace

TEST_CASE("a matrix-free CG on E^T W E x = E^T W f reproduces the PETSc "
          "projection, with the conditioning of the mass matrix",
          "[quadrature][matrix_free]")
{
  Omega_h::Library lib;
  auto source_mesh = pcms::test::BuildSquareMesh(lib, 6);
  auto target_mesh = pcms::test::BuildSquareMesh(lib, 4);
  const auto source_geometry = MakeGeometry(source_mesh);
  const auto target_geometry = MakeGeometry(target_mesh);
  const auto source_space = pcms::test::MakeP1Space(source_geometry);
  const auto target_space = pcms::test::MakeP1Space(target_geometry);
  auto source = source_space->CreateFunction<pcms::Real>();
  pcms::test::SetField(source, OMEGA_H_LAMBDA(pcms::Real x, pcms::Real y) {
    return std::sin(4 * x) * std::cos(3 * y);
  });

  pcms::ProjectionOptions options;
  options.quadrature_options.order = 2;
  auto target = target_space->CreateFunction<pcms::Real>();
  pcms::CreateProjection(source_space, target_space, options)
    ->Apply(source, target);

  // the same right-hand side, by hand
  const auto quadrature =
    target_geometry->CreateQuadrature(options.quadrature_options);
  const MatrixFreeMass mass(*target_space, quadrature);
  const pcms::CoordinateView<pcms::DeviceMemorySpace> points(
    quadrature.points->GetCoordinateSystem(),
    pcms::MakeConstRank2View(quadrature.points->LocatedPointValues()));
  const auto source_at_points = source_space->CreatePointEvaluator<pcms::Real>(
    source_geometry->Bind(points));
  Device2D f("f", source_at_points->NumPoints(), 1);
  source_at_points->Evaluate(
    source, pcms::ValueView<pcms::Real, pcms::DeviceMemorySpace>(
              source.GetData().GetValueBasis(), pcms::MakeRank2View(f)));
  const auto b = mass.Transpose(f);

  std::vector<double> x;
  const int iterations = ConjugateGradients(
    [&](const std::vector<double>& v) { return mass.Apply(v); }, b, x, 1e-13);
  const auto expected = FieldValues(target);
  REQUIRE(expected.size() == x.size());
  double max_difference = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    max_difference = std::max(max_difference, std::abs(x[i] - expected[i]));
  }
  CAPTURE(iterations, max_difference);
  CHECK(max_difference < 1e-10);
  std::printf("[matrix-free] P1 mass %d unknowns: CG iterations to 1e-13 = %d, "
              "max |x_cg - x_petsc| = %.2e\n",
              mass.Size(), iterations, max_difference);

  // dense M = E^T W E for a few meshes: conditioning of the Lagrange basis
  for (const int n : {2, 4, 8}) {
    auto mesh = pcms::test::BuildSquareMesh(lib, n);
    const auto geometry = MakeGeometry(mesh);
    const auto space = pcms::test::MakeP1Space(geometry);
    const MatrixFreeMass operator_(
      *space, geometry->CreateQuadrature(options.quadrature_options));
    const int size = operator_.Size();
    std::vector<std::vector<double>> dense(size);
    double total = 0;
    for (int j = 0; j < size; ++j) {
      std::vector<double> unit(size, 0.0);
      unit[j] = 1.0;
      const auto column = operator_.Apply(unit);
      for (int i = 0; i < size; ++i) {
        dense[i].resize(size);
        dense[i][j] = column[i];
        total += column[i];
      }
    }
    CHECK(total == Catch::Approx(1.0).margin(1e-12)); // 1^T M 1 = area
    const auto [low, high] = ExtremeEigenvalues(dense);
    CHECK(low > 0);
    std::printf("[conditioning] P1 mass, %dx%d square, %d unknowns: "
                "lambda_min = %.3e, lambda_max = %.3e, cond = %.2f\n",
                n, n, size, low, high, high / low);
    // CG iterations on a random right-hand side
    std::vector<double> rhs(size, 1.0), solution;
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    for (auto& v : rhs) {
      v = uniform(rng);
    }
    const int cg = ConjugateGradients(
      [&](const std::vector<double>& v) { return operator_.Apply(v); }, rhs,
      solution, 1e-10);
    std::printf("[conditioning]   CG iterations to 1e-10 = %d\n", cg);
  }

  // the monomial basis x^a y^b, a + b <= p, on the reference triangle:
  // M = int x^(a+c) y^(b+d) = (a+c)! (b+d)! / (a+b+c+d+2)!
  for (const int degree : {1, 2, 3, 4, 5}) {
    std::vector<std::pair<int, int>> basis;
    for (int a = 0; a <= degree; ++a) {
      for (int b = 0; a + b <= degree; ++b) {
        basis.emplace_back(a, b);
      }
    }
    std::vector<std::vector<double>> dense(basis.size(),
                                           std::vector<double>(basis.size()));
    for (std::size_t i = 0; i < basis.size(); ++i) {
      for (std::size_t j = 0; j < basis.size(); ++j) {
        const int a = basis[i].first + basis[j].first;
        const int c = basis[i].second + basis[j].second;
        dense[i][j] = Factorial(a) * Factorial(c) / Factorial(a + c + 2);
      }
    }
    const auto [low, high] = ExtremeEigenvalues(dense);
    std::printf("[conditioning] monomial mass, degree %d on the reference "
                "triangle (%zu functions): cond = %.3e\n",
                degree, basis.size(), high / low);
  }
}
