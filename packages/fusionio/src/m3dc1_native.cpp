#include "m3dc1_native.hpp"
#include "fusion_io_source.hpp"
#include "host_evaluation.hpp"
#include "pcms/discretization/discretization/empty.hpp"
#include "pcms/field/coordinate_systems/cylindrical.hpp"
#include "pcms/field/evaluator/omega_h_geometry.hpp"
#include "pcms/field/layout/empty.h"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/profile.h"
#include <Omega_h_build.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numbers>
#include <numeric>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace fusionio::detail
{

using namespace pcms;

namespace
{

/// m3dc1_field::nbasis: the terms xi^mi eta^ni of an element's polynomial
constexpr int NumBasis = 20;

/// Columns of M3DC1NativeMesh::Frames
constexpr int FrameX = 0, FrameZ = 1, FrameCo = 2, FrameSn = 3, FrameB = 4;
constexpr int NumFrameColumns = 5;

/// Coefficients of a field: [element][term], the m3dc1_field's data.
using Coefficients =
  Kokkos::View<const Real**, Kokkos::LayoutRight, DeviceMemorySpace>;

/// m3dc1_field::eval of the polynomial of element coefficients `c` at the
/// local coordinates (xi, eta) of an element rotated by (co, sn), with
/// GET_VAL, GET_DVAL (order >= 1) and GET_DDVAL (order >= 2): val is OP_1,
/// OP_DR, OP_DZ, OP_DRR, OP_DRZ, OP_DZZ. The arithmetic is 1.x's, term by
/// term, so the results are 1.x's up to the compiler's contraction of
/// multiply-adds.
KOKKOS_INLINE_FUNCTION void EvaluatePolynomial(const Real* c, Real xi, Real eta,
                                               Real co, Real sn, int order,
                                               Real* val)
{
  // m3dc1_field::mi, m3dc1_field::ni
  constexpr int mi[NumBasis] = {0, 1, 0, 2, 1, 0, 3, 2, 1, 0,
                                4, 3, 2, 1, 0, 5, 3, 2, 1, 0};
  constexpr int ni[NumBasis] = {0, 0, 1, 0, 1, 2, 0, 1, 2, 3,
                                0, 1, 2, 3, 4, 0, 2, 3, 4, 5};
  // xip[i] = xi^i for i in [-2, 5], with xi^-1 = xi^-2 = 1 as in 1.x (they
  // are multiplied by zero)
  Real xipow[8];
  Real etapow[8];
  Real* xip = &xipow[2];
  Real* etap = &etapow[2];
  for (int i = -2; i <= 0; ++i) {
    xip[i] = 1.;
    etap[i] = 1.;
  }
  for (int i = 1; i <= 5; ++i) {
    xip[i] = xip[i - 1] * xi;
    etap[i] = etap[i - 1] * eta;
  }
  for (int i = 0; i < 6; ++i) {
    val[i] = 0.;
  }
  const Real co2 = co * co;
  const Real sn2 = sn * sn;
  const Real cosn = co * sn;
  for (int p = 0; p < NumBasis; ++p) {
    val[0] = val[0] + c[p] * xip[mi[p]] * etap[ni[p]];
    if (order >= 1) {
      Real temp = c[p] * mi[p] * xip[mi[p] - 1] * etap[ni[p]];
      val[1] += co * temp;
      val[2] += sn * temp;
      temp = c[p] * ni[p] * xip[mi[p]] * etap[ni[p] - 1];
      val[1] -= sn * temp;
      val[2] += co * temp;
    }
    if (order >= 2) {
      Real temp = c[p] * (mi[p] - 1) * mi[p] * xip[mi[p] - 2] * etap[ni[p]];
      val[3] += co2 * temp;
      val[4] += cosn * temp;
      val[5] += sn2 * temp;
      temp = c[p] * mi[p] * ni[p] * xip[mi[p] - 1] * etap[ni[p] - 1];
      val[3] -= 2. * cosn * temp;
      val[4] += (co2 - sn2) * temp;
      val[5] += 2. * cosn * temp;
      temp = c[p] * (ni[p] - 1) * ni[p] * xip[mi[p]] * etap[ni[p] - 2];
      val[3] += sn2 * temp;
      val[4] -= cosn * temp;
      val[5] += co2 * temp;
    }
  }
}

/// A stored field on the device: an m3dc1_field's coefficients, or an
/// m3dc1_complex_field's real and imaginary ones and its toroidal wave number
/// kphi = 2 pi ntor / period.
struct NativeField
{
  Coefficients real;
  /// empty for a real field (or a complex one of toroidal mode 0, which
  /// evaluates to its real part)
  Coefficients imag;
  Real kphi = 0.;

  // called in kernels
  [[nodiscard]] KOKKOS_INLINE_FUNCTION bool Empty() const noexcept
  {
    return real.extent(0) == 0;
  }
};

/// m3dc1_field::eval or m3dc1_complex_field::eval of `field` in element e at
/// (xi, eta) and the toroidal angle phi (the point's minus the phase): val as
/// EvaluatePolynomial, pval the phi derivatives of its entries (OP_DP,
/// OP_DRP, OP_DZP, ...) and ppval the second of the value (OP_DPP); the phi
/// derivatives are zero for real fields, as in 1.x.
KOKKOS_INLINE_FUNCTION void EvaluateField(const NativeField& field, LO e,
                                          Real xi, Real eta, Real co, Real sn,
                                          Real phi, int order, Real* val,
                                          Real* pval, Real& ppval)
{
  EvaluatePolynomial(&field.real(e, 0), xi, eta, co, sn, order, val);
  if (field.imag.extent(0) == 0) {
    for (int j = 0; j < 6; ++j) {
      pval[j] = 0.;
    }
    ppval = 0.;
    return;
  }
  Real vr[6];
  Real vi[6];
  for (int j = 0; j < 6; ++j) {
    vr[j] = val[j];
  }
  EvaluatePolynomial(&field.imag(e, 0), xi, eta, co, sn, order, vi);
  const Real c = Kokkos::cos(phi * field.kphi);
  const Real s = Kokkos::sin(phi * field.kphi);
  for (int j = 0; j < 6; ++j) {
    val[j] = vr[j] * c - vi[j] * s;
    pval[j] = -field.kphi * (vi[j] * c + vr[j] * s);
  }
  ppval = -field.kphi * field.kphi * val[0];
}

/// A native quantity. A scalar is (linfac * field + equilibrium - offset) *
/// factor, as m3dc1_scalar_field::eval, and its derivatives the same without
/// the offset (m3dc1_scalar_field::eval_deriv). The magnetic field is
/// m3dc1_magnetic_field::eval with psi the field, I the current and f the
/// toroidal term:
///   B_R   = -(dpsi/dZ) / r - d2f/dRdphi
///   B_phi =  I / r
///   B_Z   =  (dpsi/dR) / r - d2f/dZdphi
/// (r = R for toroidal files, 1 otherwise; with ifprime, f is f' and its R
/// and Z derivatives replace those), each term of the perturbation times
/// linfac and the equilibrium's psi and I terms added, times factor (B0).
/// Fields are evaluated at the point's phi minus the phase.
struct NativeQuantity
{
  bool magnetic_field = false;
  /// the scalar's stored field, or the magnetic field's psi
  NativeField field;
  /// `field`'s equilibrium when it is added (1.x's eqsub), otherwise empty
  NativeField equilibrium;
  /// the magnetic field's I, and its equilibrium when it is added
  NativeField current;
  NativeField current_equilibrium;
  /// the magnetic field's f (or f') when 1.x takes it (use_f), else empty
  NativeField toroidal_term;
  bool fprime = false;
  Real linfac = 1.;
  Real offset = 0.;
  Real factor = 1.;
  Real phase = 0.;
  bool toroidal = true;
};

using NativeFieldData = ExternalFieldData<NativeQuantity>;

/// M3DC1Binding::Locate's kernels. They are free functions because nvcc
/// does not take an extended lambda in a private member function.
/// ProjectToPlane writes the (R, Z) and phi of the n points.
template <typename Points, typename RZ, typename Phi>
void ProjectToPlane(LO n, const Points& points, const RZ& rz, const Phi& phi)
{
  ForEach(
    "M3DC1Binding::Project", n, KOKKOS_LAMBDA(const LO i) {
      rz(i, 0) = points(i, 0);
      rz(i, 1) = points(i, 2);
      phi(i) = points(i, 1);
    });
}

/// The found mask of n points of which `missing` are not found.
template <typename Missing>
Kokkos::View<bool*, DeviceMemorySpace> FoundMask(LO n, const Missing& missing)
{
  Kokkos::View<bool*, DeviceMemorySpace> found(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "m3dc1_found"), n);
  ForEach(
    "M3DC1Binding::Found", n, KOKKOS_LAMBDA(const LO i) { found(i) = true; });
  ForEach(
    "M3DC1Binding::Missing", static_cast<LO>(missing.extent(0)),
    KOKKOS_LAMBDA(const LO k) { found(missing(k)) = false; });
  return found;
}

/// Points of M3DC1Geometry: the (R, Z) projection of the points bound on the
/// native mesh's (R, Z) geometry, and each point's phi. The found points are
/// the projection's.
class M3DC1Binding final : public BoundSource
{
public:
  M3DC1Binding(std::shared_ptr<const Geometry> geometry,
               const BindRequest& request,
               std::shared_ptr<const Geometry> plane)
    : BoundSource(std::move(geometry), request), plane_(std::move(plane))
  {
    Locate();
  }

  /// The projection's binding: ElementIds() per found point, and
  /// LocatedPointValues() the (R, Z) of every point.
  [[nodiscard]] const OmegaHMeshBinding& Plane() const
  {
    return static_cast<const OmegaHMeshBinding&>(*plane_bound_);
  }

  /// phi of every point
  [[nodiscard]] Kokkos::View<const Real*, DeviceMemorySpace> Phi()
    const noexcept
  {
    return Kokkos::subview(phi_, Kokkos::make_pair(LO{0}, NumPoints()));
  }

protected:
  void RebindImpl() override { Locate(); }

  [[nodiscard]] bool UnsharedImpl() const noexcept override
  {
    return rz_.use_count() <= 1 && phi_.use_count() <= 1 &&
           plane_bound_.use_count() <= 1;
  }

private:
  // Projects the points to (R, Z), binds (or rebinds) the projection and
  // takes its found points.
  void Locate()
  {
    const LO n = NumPoints();
    const auto points = LocatedPointValues();
    pcms::detail::ReserveRows(rz_, "m3dc1_rz", static_cast<size_t>(n), 2);
    pcms::detail::ReserveRows(phi_, "m3dc1_phi", static_cast<size_t>(n));
    const auto rz =
      Kokkos::subview(rz_, Kokkos::make_pair(LO{0}, n), Kokkos::ALL);
    ProjectToPlane(n, points, rz, phi_);
    const CoordinateView<DeviceMemorySpace> projected(
      plane_->GetCoordinateSystem(), MakeRank2View(rz));
    plane_bound_ = plane_bound_ == nullptr
                     ? plane_->Bind(projected, Policy())
                     : plane_->Rebind(std::move(plane_bound_), projected);
    const auto missing = plane_bound_->MissingPoints();
    if (missing.extent(0) == 0) {
      return;
    }
    SetFound(FoundMask(n, missing));
  }

  std::shared_ptr<const Geometry> plane_;
  std::shared_ptr<const BoundSource> plane_bound_;
  Kokkos::View<Real**, DeviceMemorySpace> rz_;
  Kokkos::View<Real*, DeviceMemorySpace> phi_;
};

/// (R, phi, Z) points on the native mesh of an axisymmetric M3D-C1 domain:
/// CylindricalRThetaZ, as Fusion-IO's (R, phi, Z), with phi = theta. Bind
/// locates the points' (R, Z) projection on the mesh's (R, Z) geometry and
/// keeps phi. KnownOwners are ignored (they would name the mesh's elements,
/// which this geometry's discretization does not expose), and
/// NEAREST_BOUNDARY is not supported.
class M3DC1Geometry final : public Geometry
{
public:
  explicit M3DC1Geometry(std::shared_ptr<const M3DC1NativeMesh> mesh)
    : mesh_(std::move(mesh)),
      discretization_(std::make_shared<EmptyDiscretization>()),
      system_(csys::CylindricalRThetaZ::Create())
  {
  }

  [[nodiscard]] std::shared_ptr<const Discretization> GetDiscretization()
    const noexcept override
  {
    return discretization_;
  }

  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>&
  GetCoordinateSystem() const noexcept override
  {
    return system_;
  }

protected:
  [[nodiscard]] std::shared_ptr<BoundSource> BindImpl(
    const BindRequest& request) const override
  {
    return std::make_shared<M3DC1Binding>(shared_from_this(), request,
                                          mesh_->GetGeometry());
  }

private:
  std::shared_ptr<const M3DC1NativeMesh> mesh_;
  std::shared_ptr<const Discretization> discretization_;
  std::shared_ptr<const CoordinateSystem> system_;
};

/// The M3DC1NativeEvaluator kernel: `q` at the found points of a binding
/// (rz and phis of every point, the element of each found one), the value
/// (order 0), gradient (1) or Hessian (2). A free function because nvcc does
/// not take an extended lambda in a protected member function.
template <typename Out, typename RZ, typename Phi, typename Found,
          typename Elements, typename Frames>
void EvaluateNative(const NativeQuantity& q, int order, const Out& out,
                    const RZ& rz, const Phi& phis, const Found& found,
                    const Elements& elements, const Frames& frames)
{
  ForEach(
    "M3DC1NativeEvaluator", static_cast<LO>(found.extent(0)),
    KOKKOS_LAMBDA(const LO k) {
      const LO i = found(k);
      const LO e = elements(k);
      const Real co = frames(e, FrameCo);
      const Real sn = frames(e, FrameSn);
      const Real R = rz(i, 0);
      // m3dc1_mesh::global_to_local
      const Real dx = R - frames(e, FrameX);
      const Real dz = rz(i, 1) - frames(e, FrameZ);
      const Real xi = dx * co + dz * sn - frames(e, FrameB);
      const Real eta = -dx * sn + dz * co;
      const Real phi = phis(i) - q.phase;
      const bool eqsub = !q.equilibrium.Empty();
      Real v1[6], p1[6], pp1;
      Real v0[6], p0[6], pp0;
      if (q.magnetic_field) {
        const Real r = q.toroidal ? R : 1.;
        Real b[3];
        EvaluateField(q.field, e, xi, eta, co, sn, phi, 1, v1, p1, pp1);
        b[0] = -q.linfac * v1[2] / r;
        b[2] = q.linfac * v1[1] / r;
        EvaluateField(q.current, e, xi, eta, co, sn, phi, 0, v1, p1, pp1);
        b[1] = q.linfac * v1[0] / r;
        if (!q.toroidal_term.Empty()) {
          EvaluateField(q.toroidal_term, e, xi, eta, co, sn, phi, 1, v1, p1,
                        pp1);
          b[0] -= q.linfac * (q.fprime ? v1[1] : p1[1]);
          b[2] -= q.linfac * (q.fprime ? v1[2] : p1[2]);
        }
        if (eqsub) {
          EvaluateField(q.equilibrium, e, xi, eta, co, sn, phi, 1, v0, p0, pp0);
          b[0] -= v0[2] / r;
          b[2] += v0[1] / r;
          EvaluateField(q.current_equilibrium, e, xi, eta, co, sn, phi, 0, v0,
                        p0, pp0);
          b[1] += v0[0] / r;
        }
        for (int c = 0; c < 3; ++c) {
          out(i, c) = b[c] * q.factor;
        }
        return;
      }
      EvaluateField(q.field, e, xi, eta, co, sn, phi, order, v1, p1, pp1);
      if (eqsub) {
        EvaluateField(q.equilibrium, e, xi, eta, co, sn, phi, order, v0, p0,
                      pp0);
      }
      // m3dc1_scalar_field: linfac times the field plus the equilibrium,
      // times the factor
      const auto scaled = [&](Real perturbed, Real equilibrium) {
        Real v = q.linfac * perturbed;
        if (eqsub) {
          v += equilibrium;
        }
        return v * q.factor;
      };
      if (order == 0) {
        Real v = q.linfac * v1[0];
        if (eqsub) {
          v += v0[0];
        }
        v -= q.offset;
        v *= q.factor;
        out(i, 0) = v;
        return;
      }
      const Real f_r = scaled(v1[1], v0[1]);
      const Real f_phi = scaled(p1[0], p0[0]);
      const Real f_z = scaled(v1[2], v0[2]);
      if (order == 1) {
        // physical components: (1/R) d/dphi, as the wrapped evaluator
        out(i, 0) = f_r;
        out(i, 1) = f_phi / R;
        out(i, 2) = f_z;
        return;
      }
      // The Hessian's physical components in (R, phi, Z): the coordinate
      // second derivatives less the connection's terms, over the scale
      // factors (1, R, 1).
      const Real f_rr = scaled(v1[3], v0[3]);
      const Real f_rz = scaled(v1[4], v0[4]);
      const Real f_zz = scaled(v1[5], v0[5]);
      const Real f_rphi = scaled(p1[1], p0[1]);
      const Real f_zphi = scaled(p1[2], p0[2]);
      const Real f_phiphi = scaled(pp1, pp0);
      const Real h_rphi = f_rphi / R - f_phi / (R * R);
      const Real h_phiz = f_zphi / R;
      const Real h_phiphi = f_phiphi / (R * R) + f_r / R;
      const Real h[9] = {f_rr,   h_rphi, f_rz,   h_rphi, h_phiphi,
                         h_phiz, f_rz,   h_phiz, f_zz};
      for (int c = 0; c < 9; ++c) {
        out(i, c) = h[c];
      }
    });
}

/// Evaluates native quantities at the points of an M3DC1Geometry binding:
/// the value, or for scalars the gradient (d/dR, (1/R) d/dphi, d/dZ) or the
/// Hessian (rows and columns R, phi, Z), physical cylindrical components.
class M3DC1NativeEvaluator final : public PointEvaluator<Real>
{
public:
  M3DC1NativeEvaluator(std::shared_ptr<const M3DC1Binding> bound,
                       DifferentialOperator op,
                       std::shared_ptr<const FieldLayout> layout,
                       M3DC1NativeMesh::Frames frames)
    : PointEvaluator<Real>(bound, op, std::move(layout)),
      bound_(std::move(bound)),
      frames_(std::move(frames))
  {
  }

  [[nodiscard]] ValueBasis OutputBasis(const ValueBasis& stored) const override
  {
    if (Operator() != DifferentialOperator::Hessian) {
      return PointEvaluator<Real>::OutputBasis(stored);
    }
    if (stored.Rank() != 0) {
      throw pcms_error("M3DC1NativeEvaluator: the Hessian of a rank-" +
                       std::to_string(stored.Rank()) +
                       " field is not implemented");
    }
    return ValueBasis{
      bound_->GetGeometry()->GetCoordinateSystem(), ComponentScaling::Physical,
      values::Of({Variance::Covariant, Variance::Covariant}), stored.parity};
  }

protected:
  void EvaluateImpl(const Field<Real>& field,
                    ValueView<Real, DeviceMemorySpace> values) const override
  {
    PCMS_FUNCTION_TIMER;
    const auto* data = dynamic_cast<const NativeFieldData*>(&field.GetData());
    if (data == nullptr) {
      throw pcms_error("M3DC1NativeEvaluator: the field is not a native "
                       "M3D-C1 quantity");
    }
    const NativeQuantity q = data->quantity;
    const auto out = values.GetValues();
    const auto rz = bound_->Plane().LocatedPointValues();
    const auto phis = bound_->Phi();
    const auto found = bound_->FoundPoints();
    const auto elements = bound_->Plane().ElementIds();
    const int order = Operator() == DifferentialOperator::Value      ? 0
                      : Operator() == DifferentialOperator::Gradient ? 1
                                                                     : 2;
    EvaluateNative(q, order, out, rz, phis, found, elements, frames_);
  }

private:
  std::shared_ptr<const M3DC1Binding> bound_;
  M3DC1NativeMesh::Frames frames_;
};

/// Creates the native evaluators of the quantities of one representation on
/// an M3DC1Geometry: scalars with Gradient and Hessian, the magnetic field
/// with Value only.
class M3DC1NativeEvaluatorFactory final
  : public TypedPointEvaluatorFactory<Real, M3DC1Binding>
{
public:
  M3DC1NativeEvaluatorFactory(
    std::shared_ptr<const M3DC1Geometry> geometry,
    M3DC1NativeMesh::Frames frames, std::shared_ptr<const FieldLayout> layout,
    std::initializer_list<DifferentialOperator> supported)
    : TypedPointEvaluatorFactory(std::move(geometry), supported),
      frames_(std::move(frames)),
      layout_(std::move(layout))
  {
  }

protected:
  std::unique_ptr<PointEvaluator<Real>> CreateEvaluator(
    std::shared_ptr<const M3DC1Binding> bound,
    DifferentialOperator op) const override
  {
    return std::make_unique<M3DC1NativeEvaluator>(std::move(bound), op, layout_,
                                                  frames_);
  }

private:
  M3DC1NativeMesh::Frames frames_;
  std::shared_ptr<const FieldLayout> layout_;
};

/// A space of native quantities with `num_components` components.
std::shared_ptr<FunctionSpace> CreateNativeSpace(
  std::shared_ptr<const M3DC1Geometry> geometry, const M3DC1NativeMesh& mesh,
  int num_components, std::initializer_list<DifferentialOperator> supported)
{
  auto layout = std::make_shared<EmptyFieldLayout>(
    geometry->GetCoordinateSystem(), num_components);
  auto evaluator = std::make_shared<M3DC1NativeEvaluatorFactory>(
    std::move(geometry), mesh.GetFrames(), layout, supported);
  return std::make_shared<FunctionSpace>(std::move(layout), CreateNoFieldData,
                                         std::move(evaluator));
}

/// Whether two M3D-C1 meshes have the same elements.
bool SameElements(const m3dc1_mesh& a, const m3dc1_mesh& b)
{
  const auto n = static_cast<size_t>(a.nelms);
  const auto same = [n](const double* x, const double* y) {
    return std::equal(x, x + n, y);
  };
  return a.nelms == b.nelms && same(a.a, b.a) && same(a.b, b.b) &&
         same(a.c, b.c) && same(a.co, b.co) && same(a.sn, b.sn) &&
         same(a.x, b.x) && same(a.z, b.z);
}

/// The Omega_h mesh of an M3D-C1 mesh's elements, see M3DC1NativeMesh.
std::unique_ptr<Omega_h::Mesh> BuildMesh(Omega_h::Library& library,
                                         const m3dc1_mesh& m3dc1)
{
  const auto nelms = static_cast<size_t>(m3dc1.nelms);
  // the corners of each element: local (-b, 0), (a, 0), (0, c)
  std::vector<std::array<Real, 2>> corners(3 * nelms);
  for (size_t e = 0; e < nelms; ++e) {
    const int i = static_cast<int>(e);
    const Real local[3][2] = {
      {-m3dc1.b[e], 0.}, {m3dc1.a[e], 0.}, {0., m3dc1.c[e]}};
    for (int k = 0; k < 3; ++k) {
      Real phi = 0.;
      m3dc1.local_to_global(i, local[k][0], 0., local[k][1],
                            &corners[3 * e + k][0], &phi,
                            &corners[3 * e + k][1]);
    }
  }
  // M3D-C1 files store the frames in single precision, so the corners elements
  // share agree to about 1e-7 of the domain only. 1.x takes corners within
  // 1e-4 of an element's a + b + c for the same (m3dc1_mesh::shared_nodes);
  // the smallest element's bound merges them and nothing else.
  Real smallest = m3dc1.a[0] + m3dc1.b[0] + m3dc1.c[0];
  for (size_t e = 0; e < nelms; ++e) {
    smallest = std::min(smallest, m3dc1.a[e] + m3dc1.b[e] + m3dc1.c[e]);
  }
  const Real tolerance = 1e-4 * smallest;
  // Merge the corners within the tolerance: sorted by R, the corners within
  // the tolerance of one precede it within a band of R.
  std::vector<size_t> order(corners.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::sort(order.begin(), order.end(),
            [&](size_t i, size_t j) { return corners[i] < corners[j]; });
  std::vector<Omega_h::LO> vertex_of(corners.size(), -1);
  std::vector<Real> coords;
  for (size_t k = 0; k < order.size(); ++k) {
    const auto& corner = corners[order[k]];
    for (size_t j = k; j-- > 0;) {
      const auto& other = corners[order[j]];
      if (corner[0] - other[0] > tolerance) {
        break;
      }
      if (std::abs(corner[1] - other[1]) <= tolerance) {
        vertex_of[order[k]] = vertex_of[order[j]];
        break;
      }
    }
    if (vertex_of[order[k]] < 0) {
      vertex_of[order[k]] = static_cast<Omega_h::LO>(coords.size() / 2);
      coords.insert(coords.end(), {corner[0], corner[1]});
    }
  }
  for (size_t e = 0; e < nelms; ++e) {
    const auto* v = &vertex_of[3 * e];
    if (v[0] == v[1] || v[1] == v[2] || v[0] == v[2]) {
      throw pcms_error("fusionio: M3D-C1 element " + std::to_string(e) +
                       " is degenerate at the merge tolerance");
    }
  }
  Omega_h::HostWrite<Omega_h::LO> elems2verts(
    static_cast<Omega_h::LO>(vertex_of.size()));
  for (size_t i = 0; i < vertex_of.size(); ++i) {
    elems2verts[static_cast<Omega_h::LO>(i)] = vertex_of[i];
  }
  Omega_h::HostWrite<Omega_h::Real> vertex_coords(
    static_cast<Omega_h::LO>(coords.size()));
  for (size_t i = 0; i < coords.size(); ++i) {
    vertex_coords[static_cast<Omega_h::LO>(i)] = coords[i];
  }
  auto mesh = std::make_unique<Omega_h::Mesh>(&library);
  Omega_h::build_from_elems_and_coords(mesh.get(), OMEGA_H_SIMPLEX, 2,
                                       Omega_h::LOs(elems2verts.write()),
                                       Omega_h::Reals(vertex_coords.write()));
  // every entity classified on its own dimension, model entity 0
  for (Omega_h::Int dim = 0; dim <= mesh->dim(); ++dim) {
    mesh->add_tag<Omega_h::I8>(
      dim, "class_dim", 1,
      Omega_h::Read<Omega_h::I8>(mesh->nents(dim), Omega_h::I8(dim)));
    mesh->add_tag<Omega_h::ClassId>(
      dim, "class_id", 1,
      Omega_h::Read<Omega_h::ClassId>(mesh->nents(dim), Omega_h::ClassId(0)));
  }
  return mesh;
}

/// The stored field and factors Fusion-IO 1.x reads for a scalar quantity
/// (m3dc1_source::get_field), and whether it is the equilibrium only.
struct ScalarEntry
{
  std::string name;
  Real factor = 1.;
  Real offset = 0.;
  bool equilibrium_only = false;
};

/// The options of a field as m3dc1_fio_field::load resolves them.
struct ResolvedOptions
{
  int time = 0;
  Real linfac = 1.;
  bool eqsub = false;
  bool use_f = false;
};

/// An M3D-C1 source evaluated natively, see CreateM3DC1NativeSource.
class M3DC1NativeSource final : public Source
{
public:
  M3DC1NativeSource(std::shared_ptr<FusionIOSource> source,
                    std::shared_ptr<Source> wrapped)
    : source_(std::move(source)), wrapped_(std::move(wrapped))
  {
    m3dc1_ = dynamic_cast<m3dc1_source*>(&source_->Get());
    const m3dc1_mesh* mesh = source_->DomainMesh();
    if (m3dc1_ == nullptr || mesh == nullptr || m3dc1_->i3d == 1) {
      throw pcms_error("fusionio: native evaluation needs an M3D-C1 source "
                       "with a 2D mesh");
    }
    domain_mesh_ = mesh;
    mesh_ = std::make_shared<M3DC1NativeMesh>(*mesh);
    geometry_ = std::make_shared<M3DC1Geometry>(mesh_);
    scalars_ = CreateNativeSpace(
      geometry_, *mesh_, 1,
      {DifferentialOperator::Gradient, DifferentialOperator::Hessian});
    vectors_ = CreateNativeSpace(geometry_, *mesh_, 3, {});
  }

  [[nodiscard]] std::shared_ptr<const Geometry> GetGeometry() const override
  {
    return geometry_;
  }

  [[nodiscard]] std::vector<QuantityInfo> Quantities() const override
  {
    return wrapped_->Quantities();
  }

  [[nodiscard]] std::vector<Real> Times() const override
  {
    return wrapped_->Times();
  }

  [[nodiscard]] Real ToroidalPeriod() const override
  {
    return wrapped_->ToroidalPeriod();
  }

  [[nodiscard]] std::optional<Real> EvaluateSeries(Series series,
                                                   Real time) const override
  {
    return wrapped_->EvaluateSeries(series, time);
  }

  [[nodiscard]] Function<Real> CreateFunction(
    Quantity quantity, const Options& options) const override
  {
    if (quantity == Quantity::MagneticField) {
      return CreateMagneticField(options);
    }
    const ScalarEntry entry = Entry(quantity, options);
    const ResolvedOptions resolved = Resolve(options, entry.equilibrium_only);
    NativeQuantity scalar;
    scalar.field = Load(entry.name, resolved.time);
    if (resolved.eqsub) {
      scalar.equilibrium = Load(entry.name, -1);
    }
    scalar.linfac = resolved.linfac;
    scalar.offset = entry.offset;
    scalar.factor = entry.factor;
    scalar.phase = options.phase.value_or(0.);
    return scalars_->CreateFunction<Real>(
      std::string(Name(quantity)),
      std::make_unique<NativeFieldData>(std::move(scalar), ValueBasis{}));
  }

private:
  // The magnetic field, as m3dc1_magnetic_field::load and eval.
  Function<Real> CreateMagneticField(const Options& options) const
  {
    const m3dc1_source& s = *m3dc1_;
    const ResolvedOptions resolved = Resolve(options, false);
    NativeQuantity b;
    b.magnetic_field = true;
    if (resolved.use_f) {
      b.fprime = s.ifprime != 0;
      b.toroidal_term = Load(b.fprime ? "fp" : "f", resolved.time);
    }
    b.field = Load("psi", resolved.time);
    b.current = Load("I", resolved.time);
    if (resolved.eqsub) {
      b.equilibrium = Load("psi", -1);
      b.current_equilibrium = Load("I", -1);
    }
    b.linfac = resolved.linfac;
    b.factor = s.B0;
    b.phase = options.phase.value_or(0.);
    b.toroidal = s.itor == 1;
    return vectors_->CreateFunction<Real>(
      std::string(Name(Quantity::MagneticField)),
      std::make_unique<NativeFieldData>(std::move(b), QuantityBasis(true)));
  }

  // The stored field of a scalar quantity, as m3dc1_source::get_field.
  ScalarEntry Entry(Quantity quantity, const Options& options) const
  {
    const m3dc1_source& s = *m3dc1_;
    const auto species = [&]() {
      if (!options.species) {
        throw pcms_error("fusionio: \"" + std::string(Name(quantity)) +
                         "\" needs the species option");
      }
      return *options.species;
    };
    switch (quantity) {
      case Quantity::PoloidalFlux:
        return {"psi", 2. * std::numbers::pi * s.B0 * s.L0 * s.L0};
      case Quantity::PoloidalFluxNorm: return FluxNormEntry(options);
      case Quantity::TotalPressure: return {"P", s.p0};
      case Quantity::Pressure:
        if (species() == Species::Electron) {
          return {"Pe", s.p0};
        }
        break;
      case Quantity::Density:
        if (species() == Species::Electron) {
          return s.version < 23 ? ScalarEntry{"den", s.n0 * s.z_ion}
                                : ScalarEntry{"ne", s.n0};
        }
        return {"den", s.n0};
      case Quantity::Temperature:
        return {species() == Species::Electron ? "te" : "ti", s.temp0};
      case Quantity::Resistivity: return {"eta", s.eta0, 0., true};
      case Quantity::TotalRadiation: return {"kprad_rad", s.p0 / s.t0};
      case Quantity::BootstrapCurrent: return {"JpdotB", s.J0};
      case Quantity::BootstrapCurrentDnDpsi: return {"JpdotB_dndpsi", s.J0};
      case Quantity::BootstrapCurrentDteDpsi: return {"JpdotB_dtedpsi", s.J0};
      case Quantity::BootstrapCurrentDtiDpsi: return {"JpdotB_dtidpsi", s.J0};
      case Quantity::BootstrapCurrentL31: return {"Jbs_L31", 1.};
      case Quantity::BootstrapCurrentL32: return {"Jbs_L32", 1.};
      case Quantity::BootstrapCurrentL34: return {"Jbs_L34", 1.};
      case Quantity::BootstrapCurrentAlpha: return {"Jbs_alpha", 1.};
      default: break;
    }
    throw pcms_error("fusionio: \"" + std::string(Name(quantity)) +
                     "\" is not evaluated natively");
  }

  // The normalized flux: (psi - psi_axis) / (psi_lcfs - psi_axis) at the
  // time of the requested slice, without the offset for the perturbed part.
  ScalarEntry FluxNormEntry(const Options& options) const
  {
    fio_source& s = source_->Get();
    const int timeslice = options.timeslice.value_or(0);
    Real time = 0.;
    if (s.get_slice_time(timeslice, &time) != FIO_SUCCESS) {
      throw pcms_error("fusionio: no time slice " + std::to_string(timeslice));
    }
    const auto series = [&](series_type type) {
      fio_series* opened = nullptr;
      const int status = s.get_series(type, &opened);
      const std::unique_ptr<fio_series> owned(opened);
      Real value = 0.;
      if (status != FIO_SUCCESS || owned->eval(time, &value) != FIO_SUCCESS) {
        throw pcms_error("fusionio: could not evaluate the flux series of "
                         "the normalized flux");
      }
      return value;
    };
    const Real axis = series(FIO_MAGAXIS_PSI);
    const Real lcfs = series(FIO_LCFS_PSI);
    const bool perturbed = options.part == Part::Perturbed;
    return {"psi", 1. / (lcfs - axis), perturbed ? 0. : axis};
  }

  // m3dc1_fio_field::load's resolution of the time slice, linear scale and
  // equilibrium subtraction.
  ResolvedOptions Resolve(const Options& options, bool equilibrium_only) const
  {
    const m3dc1_source& s = *m3dc1_;
    if (s.extsubtract == 1 && s.version < 8) {
      throw pcms_error("fusionio: native evaluation of files with subtracted "
                       "external fields is not supported");
    }
    ResolvedOptions resolved;
    resolved.time = options.timeslice.value_or(0);
    resolved.linfac = options.linear_scale.value_or(1.);
    const Part part =
      equilibrium_only ? Part::Equilibrium : options.part.value_or(Part::Total);
    if (part == Part::Equilibrium && s.eqsubtract == 1) {
      resolved.time = -1;
    }
    resolved.eqsub = s.eqsubtract == 1 && part != Part::Perturbed;
    resolved.use_f = s.i3d == 1 || s.icomplex == 1;
    if (resolved.time == -1) {
      resolved.eqsub = false;
      resolved.use_f = false;
    }
    if (resolved.linfac != 1. && (s.linear == 0 || resolved.time == -1)) {
      resolved.linfac = 1.;
    }
    return resolved;
  }

  // The coefficients of a stored field on the device, read once.
  NativeField Load(const std::string& name, int time) const
  {
    auto& loaded = fields_[{name, time}];
    if (!loaded.Empty()) {
      return loaded;
    }
    m3dc1_field* field = m3dc1_->file.load_field(name.c_str(), time);
    if (field == nullptr) {
      throw pcms_error("fusionio: could not read the M3D-C1 field \"" + name +
                       "\" of time slice " + std::to_string(time));
    }
    // 2D fields: real ones and complex ones (a toroidal mode); not 3D,
    // stellarator or compound fields
    const auto* complex = dynamic_cast<const m3dc1_complex_field*>(field);
    if (typeid(*field) != typeid(m3dc1_field) &&
        typeid(*field) != typeid(m3dc1_complex_field)) {
      throw pcms_error("fusionio: the M3D-C1 field \"" + name +
                       "\" of time slice " + std::to_string(time) +
                       " is not a 2D field; only those are evaluated "
                       "natively");
    }
    if (field->mesh != domain_mesh_ &&
        !SameElements(*field->mesh, *domain_mesh_)) {
      throw pcms_error("fusionio: the M3D-C1 field \"" + name +
                       "\" of time slice " + std::to_string(time) +
                       " is on another mesh");
    }
    loaded.real = Upload(field->data, name);
    // A complex field of toroidal mode 0 evaluates to its real part
    // (m3dc1_complex_field::eval: cos(0) = 1, sin(0) = 0).
    if (complex != nullptr && complex->ntor != 0) {
      loaded.imag = Upload(complex->data_i, name + "_i");
      loaded.kphi = 2. * std::numbers::pi * static_cast<Real>(complex->ntor) /
                    field->mesh->period;
    }
    return loaded;
  }

  // nelms x 20 coefficients copied to the device
  Coefficients Upload(const double* data, const std::string& name) const
  {
    const Kokkos::View<const Real**, Kokkos::LayoutRight, HostMemorySpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host(data, static_cast<size_t>(mesh_->NumElements()), NumBasis);
    Kokkos::View<Real**, Kokkos::LayoutRight, DeviceMemorySpace> device(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "m3dc1_coefficients_" + name),
      host.extent(0), NumBasis);
    Kokkos::deep_copy(device, host);
    return device;
  }

  std::shared_ptr<FusionIOSource> source_;
  std::shared_ptr<Source> wrapped_;
  m3dc1_source* m3dc1_ = nullptr;
  const m3dc1_mesh* domain_mesh_ = nullptr;
  std::shared_ptr<const M3DC1NativeMesh> mesh_;
  std::shared_ptr<const M3DC1Geometry> geometry_;
  std::shared_ptr<FunctionSpace> scalars_;
  std::shared_ptr<FunctionSpace> vectors_;
  mutable std::map<std::pair<std::string, int>, NativeField> fields_;
};

} // namespace

M3DC1NativeMesh::M3DC1NativeMesh(const m3dc1_mesh& mesh)
  : library_(
      std::make_unique<Omega_h::Library>(nullptr, nullptr, MPI_COMM_SELF)),
    mesh_(BuildMesh(*library_, mesh)),
    geometry_(CreateOmegaHGeometry(*mesh_, csys::CylindricalRZ::Create())),
    period_(mesh.period)
{
  const auto n = static_cast<size_t>(mesh.nelms);
  Kokkos::View<Real**, Kokkos::LayoutRight, DeviceMemorySpace> frames(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "m3dc1_frames"), n,
    NumFrameColumns);
  const auto host = Kokkos::create_mirror_view(frames);
  for (size_t e = 0; e < n; ++e) {
    host(e, FrameX) = mesh.x[e];
    host(e, FrameZ) = mesh.z[e];
    host(e, FrameCo) = mesh.co[e];
    host(e, FrameSn) = mesh.sn[e];
    host(e, FrameB) = mesh.b[e];
  }
  Kokkos::deep_copy(frames, host);
  frames_ = frames;
}

std::shared_ptr<const Geometry> M3DC1NativeMesh::GetGeometry() const
{
  // shares ownership of this object, which holds the mesh
  return {shared_from_this(), geometry_.get()};
}

std::shared_ptr<Source> CreateM3DC1NativeSource(
  std::shared_ptr<FusionIOSource> source, std::shared_ptr<Source> wrapped)
{
  return std::make_shared<M3DC1NativeSource>(std::move(source),
                                             std::move(wrapped));
}

} // namespace fusionio::detail
