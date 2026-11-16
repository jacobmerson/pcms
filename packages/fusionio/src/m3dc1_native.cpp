#include "m3dc1_native.hpp"
#include "fusion_io_source.hpp"
#include "host_evaluation.hpp"
#include "pcms/field/coordinate_systems/cylindrical.hpp"
#include "pcms/field/evaluator/omega_h_geometry.hpp"
#include "pcms/field/layout/empty.h"
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

/// A scalar quantity: (linfac * field + equilibrium - offset) * factor, as
/// m3dc1_scalar_field::eval; derivatives without the offset
/// (m3dc1_scalar_field::eval_deriv).
struct NativeScalar
{
  std::shared_ptr<const M3DC1NativeMesh> mesh;
  Coefficients field;
  /// empty unless the equilibrium is added (1.x's eqsub)
  Coefficients equilibrium;
  Real linfac = 1.;
  Real offset = 0.;
  Real factor = 1.;
};

using NativeFieldData = ExternalFieldData<NativeScalar>;

/// Evaluates native scalar quantities at the points of an Omega_h binding of
/// the native mesh's geometry: the value, the gradient (d/dR, d/dZ) or the
/// Hessian (d2/dR2, d2/dRdZ, d2/dZdR, d2/dZ2), physical components in
/// CylindricalRZ.
class M3DC1NativeEvaluator final : public PointEvaluator<Real>
{
public:
  M3DC1NativeEvaluator(std::shared_ptr<const OmegaHMeshBinding> bound,
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
    const NativeScalar& q = data->quantity;
    const auto out = values.GetValues();
    const auto points = bound_->LocatedPointValues();
    const auto found = bound_->FoundPoints();
    const auto elements = bound_->ElementIds();
    const auto frames = frames_;
    const Coefficients f1 = q.field;
    const Coefficients f0 = q.equilibrium;
    const bool eqsub = f0.extent(0) > 0;
    const Real linfac = q.linfac;
    const Real offset = q.offset;
    const Real factor = q.factor;
    const int order = Operator() == DifferentialOperator::Value      ? 0
                      : Operator() == DifferentialOperator::Gradient ? 1
                                                                     : 2;
    Kokkos::parallel_for(
      "M3DC1NativeEvaluator",
      Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(
        0, static_cast<LO>(found.extent(0))),
      KOKKOS_LAMBDA(const LO k) {
        const LO i = found(k);
        const LO e = elements(k);
        const Real co = frames(e, FrameCo);
        const Real sn = frames(e, FrameSn);
        // m3dc1_mesh::global_to_local
        const Real dx = points(i, 0) - frames(e, FrameX);
        const Real dz = points(i, 1) - frames(e, FrameZ);
        const Real xi = dx * co + dz * sn - frames(e, FrameB);
        const Real eta = -dx * sn + dz * co;
        Real v1[6];
        Real v0[6];
        EvaluatePolynomial(&f1(e, 0), xi, eta, co, sn, order, v1);
        if (eqsub) {
          EvaluatePolynomial(&f0(e, 0), xi, eta, co, sn, order, v0);
        }
        if (order == 0) {
          Real v = linfac * v1[0];
          if (eqsub) {
            v += v0[0];
          }
          v -= offset;
          v *= factor;
          out(i, 0) = v;
          return;
        }
        // (d/dR, d/dZ) or (d2/dR2, d2/dRdZ, d2/dZdR, d2/dZ2) from val's
        // OP_DR, OP_DZ or OP_DRR, OP_DRZ, OP_DZZ
        const int from[4] = {order == 1 ? 1 : 3, order == 1 ? 2 : 4, 4, 5};
        const int n = order == 1 ? 2 : 4;
        for (int c = 0; c < n; ++c) {
          Real d = linfac * v1[from[c]];
          if (eqsub) {
            d += v0[from[c]];
          }
          d *= factor;
          out(i, c) = d;
        }
      });
  }

private:
  std::shared_ptr<const OmegaHMeshBinding> bound_;
  M3DC1NativeMesh::Frames frames_;
};

/// Creates the native evaluators of the scalar quantities on the native
/// mesh's geometry: Value, Gradient and Hessian.
class M3DC1NativeEvaluatorFactory final
  : public TypedPointEvaluatorFactory<Real, OmegaHMeshBinding>
{
public:
  M3DC1NativeEvaluatorFactory(std::shared_ptr<const M3DC1NativeMesh> mesh,
                              std::shared_ptr<const FieldLayout> layout)
    : TypedPointEvaluatorFactory(
        mesh->GetGeometry(),
        {DifferentialOperator::Gradient, DifferentialOperator::Hessian}),
      mesh_(std::move(mesh)),
      layout_(std::move(layout))
  {
  }

protected:
  std::unique_ptr<PointEvaluator<Real>> CreateEvaluator(
    std::shared_ptr<const OmegaHMeshBinding> bound,
    DifferentialOperator op) const override
  {
    return std::make_unique<M3DC1NativeEvaluator>(std::move(bound), op, layout_,
                                                  mesh_->GetFrames());
  }

private:
  std::shared_ptr<const M3DC1NativeMesh> mesh_;
  std::shared_ptr<const FieldLayout> layout_;
};

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
    auto layout =
      std::make_shared<EmptyFieldLayout>(csys::CylindricalRZ::Create(), 1);
    auto evaluator =
      std::make_shared<M3DC1NativeEvaluatorFactory>(mesh_, layout);
    scalars_ = std::make_shared<FunctionSpace>(
      std::move(layout), CreateNoFieldData, std::move(evaluator));
  }

  [[nodiscard]] std::shared_ptr<const Geometry> GetGeometry() const override
  {
    return mesh_->GetGeometry();
  }

  [[nodiscard]] std::vector<QuantityInfo> Quantities() const override
  {
    return wrapped_->Quantities();
  }

  [[nodiscard]] std::vector<Real> Times() const override
  {
    return wrapped_->Times();
  }

  [[nodiscard]] Real Period() const override { return wrapped_->Period(); }

  [[nodiscard]] std::optional<Real> EvaluateSeries(Series series,
                                                   Real time) const override
  {
    return wrapped_->EvaluateSeries(series, time);
  }

  [[nodiscard]] Function<Real> CreateFunction(
    Quantity quantity, const Options& options) const override
  {
    const ScalarEntry entry = Entry(quantity, options);
    const ResolvedOptions resolved = Resolve(options, entry.equilibrium_only);
    NativeScalar scalar{mesh_,
                        Load(entry.name, resolved.time),
                        resolved.eqsub ? Load(entry.name, -1) : Coefficients(),
                        resolved.linfac,
                        entry.offset,
                        entry.factor};
    return scalars_->CreateFunction<Real>(
      std::string(Name(quantity)),
      std::make_unique<NativeFieldData>(std::move(scalar), ValueBasis{}));
  }

private:
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
    if (resolved.time == -1) {
      resolved.eqsub = false;
    }
    if (resolved.linfac != 1. && (s.linear == 0 || resolved.time == -1)) {
      resolved.linfac = 1.;
    }
    return resolved;
  }

  // The coefficients of a stored field on the device, read once.
  Coefficients Load(const std::string& name, int time) const
  {
    auto& loaded = coefficients_[{name, time}];
    if (loaded.extent(0) > 0) {
      return loaded;
    }
    m3dc1_field* field = m3dc1_->file.load_field(name.c_str(), time);
    if (field == nullptr) {
      throw pcms_error("fusionio: could not read the M3D-C1 field \"" + name +
                       "\" of time slice " + std::to_string(time));
    }
    const auto* complex = dynamic_cast<const m3dc1_complex_field*>(field);
    const bool axisymmetric =
      typeid(*field) == typeid(m3dc1_field) ||
      (typeid(*field) == typeid(m3dc1_complex_field) && complex->ntor == 0);
    if (!axisymmetric) {
      throw pcms_error("fusionio: the M3D-C1 field \"" + name +
                       "\" of time slice " + std::to_string(time) +
                       " is not axisymmetric; only 2D fields are evaluated "
                       "natively");
    }
    if (field->mesh != domain_mesh_ &&
        !SameElements(*field->mesh, *domain_mesh_)) {
      throw pcms_error("fusionio: the M3D-C1 field \"" + name +
                       "\" of time slice " + std::to_string(time) +
                       " is on another mesh");
    }
    // A complex field of toroidal mode 0 evaluates to its real part
    // (m3dc1_complex_field::eval: cos(0) = 1, sin(0) = 0).
    const Kokkos::View<const Real**, Kokkos::LayoutRight, HostMemorySpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host(field->data, static_cast<size_t>(mesh_->NumElements()), NumBasis);
    Kokkos::View<Real**, Kokkos::LayoutRight, DeviceMemorySpace> device(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "m3dc1_coefficients_" + name),
      host.extent(0), NumBasis);
    Kokkos::deep_copy(device, host);
    loaded = device;
    return loaded;
  }

  std::shared_ptr<FusionIOSource> source_;
  std::shared_ptr<Source> wrapped_;
  m3dc1_source* m3dc1_ = nullptr;
  const m3dc1_mesh* domain_mesh_ = nullptr;
  std::shared_ptr<const M3DC1NativeMesh> mesh_;
  std::shared_ptr<FunctionSpace> scalars_;
  mutable std::map<std::pair<std::string, int>, Coefficients> coefficients_;
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
