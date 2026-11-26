// The product of function spaces: the basis phi_i(a) psi_k(b) ..., DOF
// (i, k, ...). It is evaluated factor by factor: at scattered points by a
// contraction over the entries of the factors' rows (point_rows.hpp), at the
// tensor product of the factors' points mode by mode.

#include "pcms/field/detail/field_ops.hpp"
#include "pcms/field/function_spaces.hpp"
#include "pcms/field/point_rows.hpp"
#include "pcms/geometry/product_geometry.hpp"
#include "pcms/utility/assert.hpp"
#include "product_binding.hpp"
#include <algorithm>
#include <optional>
#include <set>
#include <vector>

namespace pcms
{

namespace
{

using detail::MaxFactors;
using detail::PointRows;
using Scratch = Kokkos::View<Real**, DeviceMemorySpace>;
using ConstValues = Rank2View<const Real, DeviceMemorySpace>;
using Extents = Kokkos::Array<LO, MaxFactors>;

/// Columns [offset, offset + dimension) of `out`: the coordinates `x` of each
/// holder's digit of stride `stride` among `n`.
void ScatterColumns(Scratch out, ConstValues x, LO stride, LO n, int offset)
{
  const int d = static_cast<int>(x.extent(1));
  detail::ForEach(
    "pcms::ProductCoordinates", static_cast<LO>(out.extent(0)),
    KOKKOS_LAMBDA(const LO h) {
      const LO i = (h / stride) % n;
      for (int j = 0; j < d; ++j) {
        out(h, offset + j) = x(i, j);
      }
    });
}

/// The layout of a product: holder (i_1, ..., i_N) is the mixed-radix index
/// of the factors' holders, the first factor slowest.
class ProductLayout final : public FieldLayout
{
public:
  ProductLayout(std::shared_ptr<const Geometry> geometry,
                std::vector<std::shared_ptr<const FieldLayout>> factors)
    : factors_(std::move(factors))
  {
    for (int k = static_cast<int>(factors_.size()) - 1; k >= 0; --k) {
      if (factors_[k]->IsDistributed()) {
        throw pcms_error("CreateProductSpace: the factors' layouts must be on "
                         "one rank (the product is serial)");
      }
      strides_[k] = num_holders_;
      num_holders_ *= factors_[k]->GetNumDofHolder();
      num_global_ *= factors_[k]->GetNumGlobalDofHolder();
    }
    SetCoordinateSystem(geometry->GetCoordinateSystem());
    SetGeometry(std::move(geometry));
  }

  int GetNumComponents() const override { return 1; }
  LO GetNumDofHolder() const override { return num_holders_; }
  GO GetNumGlobalDofHolder() const override { return num_global_; }
  MPI_Comm GetCommunicator() const override { return MPI_COMM_SELF; }
  Rank1View<const bool, HostMemorySpace> GetOwnedHost() const override
  {
    if (owned_.extent(0) != static_cast<std::size_t>(num_holders_)) {
      owned_ =
        Kokkos::View<bool*, HostMemorySpace>("product_owned", num_holders_);
      Kokkos::deep_copy(owned_, true);
    }
    return MakeRank1View(owned_);
  }
  GlobalIDView<HostMemorySpace> GetGidsHost() const override
  {
    if (gids_.extent(0) != static_cast<std::size_t>(num_holders_)) {
      gids_ = Kokkos::View<GO*, HostMemorySpace>("product_gids", num_holders_);
      for (LO h = 0; h < num_holders_; ++h) {
        GO gid = 0;
        for (std::size_t k = 0; k < factors_.size(); ++k) {
          const LO i = (h / strides_[k]) % factors_[k]->GetNumDofHolder();
          gid = gid * factors_[k]->GetNumGlobalDofHolder() +
                factors_[k]->GetGidsHost()[i];
        }
        gids_(h) = gid;
      }
    }
    return GlobalIDView<HostMemorySpace>(gids_.data(), gids_.extent(0));
  }
  EntOffsetsArray GetEntOffsets() const override
  {
    const auto n = static_cast<std::size_t>(num_holders_);
    return {0, n, n, n, n};
  }
  CoordinateView<DeviceMemorySpace> GetDOFHolderCoordinates() const override
  {
    if (coordinates_.extent(0) != static_cast<std::size_t>(num_holders_)) {
      coordinates_ = Scratch("product_coordinates", num_holders_,
                             static_cast<std::size_t>(GetDimension()));
      int offset = 0;
      for (std::size_t k = 0; k < factors_.size(); ++k) {
        const auto x = factors_[k]->GetDOFHolderCoordinates().GetValues();
        ScatterColumns(coordinates_, x, strides_[k],
                       factors_[k]->GetNumDofHolder(), offset);
        offset += static_cast<int>(x.extent(1));
      }
    }
    return CoordinateView<DeviceMemorySpace>(GetCoordinateSystem(),
                                             MakeConstRank2View(coordinates_));
  }
  int GetDimension() const override
  {
    int dimension = 0;
    for (const auto& factor : factors_) {
      dimension += factor->GetDimension();
    }
    return dimension;
  }
  Rank1View<const LO, HostMemorySpace>
  GetDOFHolderClassificationDimensionsHost() const override
  {
    return MakeRank1View(Zeros());
  }
  Rank1View<const LO, HostMemorySpace> GetDOFHolderClassificationIdsHost()
    const override
  {
    return MakeRank1View(Zeros());
  }

  /// @return the holder stride of each factor (the last is 1)
  [[nodiscard]] const Extents& Strides() const noexcept { return strides_; }
  [[nodiscard]] const std::vector<std::shared_ptr<const FieldLayout>>& Factors()
    const noexcept
  {
    return factors_;
  }

private:
  const Kokkos::View<LO*, HostMemorySpace>& Zeros() const
  {
    if (zeros_.extent(0) != static_cast<std::size_t>(num_holders_)) {
      zeros_ = Kokkos::View<LO*, HostMemorySpace>("zeros", num_holders_);
    }
    return zeros_;
  }

  std::vector<std::shared_ptr<const FieldLayout>> factors_;
  Extents strides_ = {};
  LO num_holders_ = 1;
  GO num_global_ = 1;
  mutable Kokkos::View<bool*, HostMemorySpace> owned_;
  mutable Kokkos::View<GO*, HostMemorySpace> gids_;
  mutable Kokkos::View<LO*, HostMemorySpace> zeros_;
  mutable Scratch coordinates_;
};

/// The rows of every factor, passed to the kernels by value.
struct FactorRows
{
  Kokkos::Array<PointRows, MaxFactors> value, gradient;
  /// holder strides; with an evaluate-only factor, of the other factors'
  /// holders
  Extents stride = {};
  /// for a tensor binding, the points of each factor
  Extents extent = {};
  Kokkos::Array<int, MaxFactors> dimension = {};
  int num_factors = 0;
  /// the evaluate-only factor, -1 if none
  int opaque = -1;
  /// the points of the evaluate-only factor
  LO opaque_points = 0;
  /// the holders of the factors after the evaluate-only one
  LO right = 1;
  bool tensor = false;
};

/// The entries of each factor's row for one output at one point.
struct RowEntries
{
  /// each factor's own point
  LO q[MaxFactors];
  LO first[MaxFactors];
  LO count[MaxFactors];
  /// the differentiated factor (-1 for a value) and its partial
  int g;
  int local;
  /// the number of entries of the product of the rows
  LO total;
};

KOKKOS_INLINE_FUNCTION RowEntries Entries(const FactorRows& f, int num_out,
                                          LO p, int r)
{
  RowEntries e{};
  e.g = num_out > 1 ? 0 : -1;
  e.local = r;
  while (e.g >= 0 && e.local >= f.dimension[e.g]) {
    e.local -= f.dimension[e.g];
    ++e.g;
  }
  e.total = 1;
  LO rest = p;
  for (int k = f.num_factors - 1; k >= 0; --k) {
    e.q[k] = f.tensor ? rest % f.extent[k] : p;
    rest = f.tensor ? rest / f.extent[k] : rest;
    if (k == f.opaque) {
      e.first[k] = 0;
      e.count[k] = 1;
      continue;
    }
    const PointRows& rows = k == e.g ? f.gradient[k] : f.value[k];
    const LO row = k == e.g ? e.q[k] * f.dimension[k] + e.local : e.q[k];
    e.first[k] = rows.offsets(row);
    e.count[k] = rows.offsets(row + 1) - e.first[k];
    e.total *= e.count[k];
  }
  return e;
}

/// A walk over the entries of the product of the rows, the last factor
/// fastest, keeping the products of the factors before each.
struct EntryWalk
{
  Real weight[MaxFactors + 1];
  /// of the other factors' holders with an evaluate-only factor
  LO holder[MaxFactors + 1];
  LO index[MaxFactors];

  /// @brief Starts at the first entry; there must be one
  KOKKOS_INLINE_FUNCTION void Start(const FactorRows& f, const RowEntries& e)
  {
    weight[0] = 1;
    holder[0] = 0;
    for (int k = 0; k < f.num_factors; ++k) {
      index[k] = 0;
    }
    From(f, e, 0);
  }

  /// @return whether there is a next entry, which the walk moves to
  KOKKOS_INLINE_FUNCTION bool Next(const FactorRows& f, const RowEntries& e)
  {
    int k = f.num_factors - 1;
    while (k >= 0 && ++index[k] == e.count[k]) {
      index[k] = 0;
      --k;
    }
    if (k < 0) {
      return false;
    }
    From(f, e, k);
    return true;
  }

  /// @brief The products from factor k on
  KOKKOS_INLINE_FUNCTION void From(const FactorRows& f, const RowEntries& e,
                                   int k)
  {
    for (; k < f.num_factors; ++k) {
      weight[k + 1] = weight[k];
      holder[k + 1] = holder[k];
      if (k != f.opaque) {
        const PointRows& rows = k == e.g ? f.gradient[k] : f.value[k];
        const LO j = e.first[k] + index[k];
        weight[k + 1] *= rows.values(j);
        holder[k + 1] += rows.holders(j) * f.stride[k];
      }
    }
  }
};

/// @return the index in [left][points][m][right] of output mm of the
///         evaluate-only factor at its point q, for the other factors' holder
KOKKOS_INLINE_FUNCTION LO OpaqueIndex(const FactorRows& f, LO holder, LO q,
                                      int m, int mm)
{
  const LO l = holder / f.right;
  const LO r = holder % f.right;
  return ((l * f.opaque_points + q) * m + mm) * f.right + r;
}

/// @return entry i of `v` in row-major order
KOKKOS_INLINE_FUNCTION Real Flat(const ConstValues& v, LO i)
{
  const auto cols = static_cast<LO>(v.extent(1));
  return v(i / cols, i % cols);
}

/// out(p, r) = the sum over the entries of the factors' rows at p of their
/// weights times the DOFs at the combined holder; with an evaluate-only
/// factor, times its values z (zg for its partials).
void ApplyFactored(const FactorRows& f,
                   Kokkos::View<const bool*, DeviceMemorySpace> found,
                   int num_out, ConstValues dofs,
                   Kokkos::View<const Real**, DeviceMemorySpace> z,
                   Kokkos::View<const Real**, DeviceMemorySpace> zg,
                   Rank2View<Real, DeviceMemorySpace> out)
{
  detail::ForEach(
    "pcms::ApplyFactored", static_cast<LO>(out.extent(0)),
    KOKKOS_LAMBDA(const LO p) {
      const bool missing = found.extent(0) > 0 && !found(p);
      for (int r = 0; r < num_out; ++r) {
        const RowEntries e = Entries(f, num_out, p, r);
        Real sum = 0;
        if (!missing && e.total > 0) {
          EntryWalk walk;
          walk.Start(f, e);
          do {
            const Real weight = walk.weight[f.num_factors];
            const LO holder = walk.holder[f.num_factors];
            if (f.opaque < 0) {
              sum += weight * dofs(holder, 0);
            } else if (e.g == f.opaque) {
              sum += weight * zg(OpaqueIndex(f, holder, e.q[f.opaque],
                                             f.dimension[f.opaque], e.local),
                                 0);
            } else {
              sum += weight * z(OpaqueIndex(f, holder, e.q[f.opaque], 1, 0), 0);
            }
          } while (walk.Next(f, e));
        }
        out(p, r) = sum;
      }
    });
}

/// total += the transpose of ApplyFactored's Value applied to `in`; with an
/// evaluate-only factor, `total` is indexed as its values z.
void ApplyFactoredTranspose(const FactorRows& f,
                            Kokkos::View<const bool*, DeviceMemorySpace> found,
                            ConstValues in, Scratch total)
{
  detail::ForEach(
    "pcms::ApplyFactoredTranspose", static_cast<LO>(in.extent(0)),
    KOKKOS_LAMBDA(const LO p) {
      if (found.extent(0) > 0 && !found(p)) {
        return;
      }
      const RowEntries e = Entries(f, 1, p, 0);
      if (e.total == 0) {
        return;
      }
      EntryWalk walk;
      walk.Start(f, e);
      do {
        const LO holder = walk.holder[f.num_factors];
        const LO i =
          f.opaque < 0 ? holder : OpaqueIndex(f, holder, e.q[f.opaque], 1, 0);
        Kokkos::atomic_add(&total(i, 0), walk.weight[f.num_factors] * in(p, 0));
      } while (walk.Next(f, e));
    });
}

/// The Kronecker rows of the factors' rows: row p * num_out + r.
PointRows KroneckerRows(const FactorRows& f,
                        Kokkos::View<const bool*, DeviceMemorySpace> found,
                        LO num_points, int num_out)
{
  const LO num_rows = num_points * num_out;
  Kokkos::View<LO*, DeviceMemorySpace> offsets("product_offsets", num_rows + 1);
  Kokkos::parallel_scan(
    "pcms::ProductRowOffsets",
    Kokkos::RangePolicy<DefaultExecutionSpace>(0, num_rows + 1),
    KOKKOS_LAMBDA(const LO r, LO& partial, const bool final) {
      if (final) {
        offsets(r) = partial;
      }
      if (r < num_rows && (found.extent(0) == 0 || found(r / num_out))) {
        partial += Entries(f, num_out, r / num_out, r % num_out).total;
      }
    });
  LO nnz = 0;
  Kokkos::deep_copy(nnz, Kokkos::subview(offsets, num_rows));
  Kokkos::View<LO*, DeviceMemorySpace> holders("product_holders", nnz);
  Kokkos::View<Real*, DeviceMemorySpace> values("product_values", nnz);
  detail::ForEach(
    "pcms::ProductRows", num_rows, KOKKOS_LAMBDA(const LO r) {
      LO k = offsets(r);
      if (k == offsets(r + 1)) {
        return;
      }
      const RowEntries e = Entries(f, num_out, r / num_out, r % num_out);
      EntryWalk walk;
      walk.Start(f, e);
      do {
        values(k) = walk.weight[f.num_factors];
        holders(k) = walk.holder[f.num_factors];
        ++k;
      } while (walk.Next(f, e));
    });
  return {offsets, holders, values, num_out};
}

/// out(l, q, s) = sum_j A(q, j) in(l, j, s): the first `num_rows` rows A
/// applied along the middle mode of `in` seen as [left][n][right] in
/// row-major order.
void ApplyMode(const PointRows& rows, LO num_rows, LO left, LO n, LO right,
               ConstValues in, Scratch out)
{
  const auto offsets = rows.offsets;
  const auto holders = rows.holders;
  const auto values = rows.values;
  detail::ForEach(
    "pcms::ApplyMode", left * num_rows * right, KOKKOS_LAMBDA(const LO i) {
      const LO s = i % right;
      const LO q = (i / right) % num_rows;
      const LO l = i / right / num_rows;
      Real sum = 0;
      for (LO j = offsets(q); j < offsets(q + 1); ++j) {
        sum += values(j) * Flat(in, (l * n + holders(j)) * right + s);
      }
      out(i, 0) = sum;
    });
}

/// out(l, j, s) += sum_q A(q, j) in(l, q, s): the transpose of ApplyMode.
void ApplyModeTranspose(const PointRows& rows, LO num_rows, LO left, LO n,
                        LO right, ConstValues in, Scratch out)
{
  const auto offsets = rows.offsets;
  const auto holders = rows.holders;
  const auto values = rows.values;
  detail::ForEach(
    "pcms::ApplyModeTranspose", left * num_rows * right,
    KOKKOS_LAMBDA(const LO i) {
      const LO s = i % right;
      const LO q = (i / right) % num_rows;
      const LO l = i / right / num_rows;
      const Real v = Flat(in, i);
      for (LO j = offsets(q); j < offsets(q + 1); ++j) {
        Kokkos::atomic_add(&out((l * n + holders(j)) * right + s, 0),
                           values(j) * v);
      }
    });
}

/// slice(j, 0) = in(l, j, s) of `in` seen as [left][n][right].
void GatherSlice(ConstValues in, LO l, LO n, LO right, LO s, Scratch slice)
{
  detail::ForEach(
    "pcms::ProductGatherSlice", n, KOKKOS_LAMBDA(const LO j) {
      slice(j, 0) = Flat(in, (l * n + j) * right + s);
    });
}

/// out(l, i, s) = `slice` in row-major order, of `out` seen as
/// [left][entries of slice][right].
void ScatterSlice(ConstValues slice, LO l, LO right, LO s, Scratch out)
{
  const auto n = static_cast<LO>(slice.extent(0) * slice.extent(1));
  detail::ForEach(
    "pcms::ProductScatterSlice", n, KOKKOS_LAMBDA(const LO i) {
      out((l * n + i) * right + s, 0) = Flat(slice, i);
    });
}

/// out(p, offset + j) = the entry of `data`, seen as
/// [q_1]...[q_g][d]...[q_N], at the digits of p and j.
void ScatterChain(ConstValues data, const Extents& extent, int num_factors,
                  int g, int d, int offset,
                  Rank2View<Real, DeviceMemorySpace> out)
{
  detail::ForEach(
    "pcms::ProductScatterChain", static_cast<LO>(out.extent(0)),
    KOKKOS_LAMBDA(const LO p) {
      for (int j = 0; j < d; ++j) {
        LO row = 0;
        LO stride = 1;
        LO rest = p;
        for (int k = num_factors - 1; k >= 0; --k) {
          const LO q = rest % extent[k];
          rest /= extent[k];
          row += (k == g ? q * d + j : q) * stride;
          stride *= k == g ? extent[k] * d : extent[k];
        }
        out(p, offset + j) = Flat(data, row);
      }
    });
}

/// out [left][points][m][right] = `evaluator` applied along the middle mode
/// of `in` seen as [left][n][right], one slice at a time through `scratch`.
void ApplyOpaque(const PointEvaluator<Real>& evaluator, Field<Real>& scratch,
                 LO left, LO n, LO right, ConstValues in, Scratch out)
{
  const LO q = evaluator.NumPoints();
  const auto basis = evaluator.OutputBasis(scratch.GetData().GetValueBasis());
  Scratch dofs("product_slice", n, 1);
  Scratch values("product_slice_values", q,
                 out.extent(0) / static_cast<std::size_t>(left * q * right));
  for (LO l = 0; l < left; ++l) {
    for (LO s = 0; s < right; ++s) {
      GatherSlice(in, l, n, right, s, dofs);
      scratch.SetDOFHolderDataUnchecked(MakeConstRank2View(dofs));
      evaluator.Evaluate(scratch, ValueView<Real, DeviceMemorySpace>(
                                    basis, MakeRank2View(values)));
      ScatterSlice(MakeConstRank2View(values), l, right, s, out);
    }
  }
}

/// out [left][n][right] = the transpose of ApplyOpaque's Value applied to
/// `in` seen as [left][points][right].
void ApplyOpaqueTranspose(const PointEvaluator<Real>& evaluator,
                          Field<Real>& scratch, LO left, LO n, LO right,
                          ConstValues in, Scratch out)
{
  const LO q = evaluator.NumPoints();
  const auto basis = scratch.GetData().GetValueBasis();
  Scratch values("product_slice_values", q, 1);
  Scratch zeros("product_slice", n, 1);
  for (LO l = 0; l < left; ++l) {
    for (LO s = 0; s < right; ++s) {
      GatherSlice(in, l, q, right, s, values);
      scratch.SetDOFHolderDataUnchecked(MakeConstRank2View(zeros));
      evaluator.EvaluateTranspose(ValueView<const Real, DeviceMemorySpace>(
                                    basis, MakeConstRank2View(values)),
                                  scratch);
      ScatterSlice(scratch.GetDOFHolderData().GetValues(), l, right, s, out);
    }
  }
}

/// One leaf factor at its bound points: its rows, or its evaluators if it
/// provides no matrix.
struct FactorAtPoints
{
  std::shared_ptr<const FunctionSpace> space;
  PointRows value;
  PointRows gradient;
  std::unique_ptr<PointEvaluator<Real>> value_evaluator;
  std::unique_ptr<PointEvaluator<Real>> gradient_evaluator;
};

/// Evaluates a product space factor by factor at a ProductBinding: mode by
/// mode at the tensor product of the factors' points, else by ApplyFactored.
class ProductPointEvaluator final : public PointEvaluator<Real>
{
public:
  /// @throws pcms_error at scattered points if more than one factor
  ///         provides no matrix
  ProductPointEvaluator(std::shared_ptr<const BoundPoints> bound,
                        std::shared_ptr<const ProductLayout> layout,
                        DifferentialOperator op,
                        const detail::LeafBinding& leaves,
                        std::vector<FactorAtPoints> factors)
    : PointEvaluator<Real>(bound, op, layout),
      layout_(std::move(layout)),
      factors_(std::move(factors))
  {
    if (bound->MissingPoints().extent(0) > 0) {
      found_ = detail::FoundMask(bound->NumPoints(), {bound->MissingPoints()});
    }
    chain_ = leaves.tensor && found_.extent(0) == 0;
    rows_.num_factors = static_cast<int>(factors_.size());
    rows_.tensor = leaves.tensor;
    rows_.extent = leaves.extent;
    rows_.stride = layout_->Strides();
    int opaque = -1;
    int num_opaque = 0;
    for (int k = 0; k < rows_.num_factors; ++k) {
      rows_.value[k] = factors_[k].value;
      rows_.gradient[k] = factors_[k].gradient;
      rows_.dimension[k] =
        leaves.factors[k]->GetGeometry()->GetCoordinateSystem()->Dimension();
      if (factors_[k].value_evaluator != nullptr) {
        opaque = k;
        order_.insert(order_.begin() + num_opaque, k);
        ++num_opaque;
      } else {
        order_.push_back(k);
      }
    }
    if (chain_ || opaque < 0) {
      return;
    }
    if (num_opaque > 1) {
      throw pcms_error("CreatePointEvaluator: a product at scattered points "
                       "needs the matrix of all factors but one");
    }
    rows_.opaque = opaque;
    rows_.right = rows_.stride[opaque];
    rows_.opaque_points = factors_[opaque].value_evaluator->NumPoints();
    for (int k = 0; k < opaque; ++k) {
      rows_.stride[k] /= Holders(opaque);
    }
  }

  [[nodiscard]] bool SupportsTranspose() const noexcept override
  {
    return std::all_of(factors_.begin(), factors_.end(),
                       [](const FactorAtPoints& f) {
                         return f.value_evaluator == nullptr ||
                                f.value_evaluator->SupportsTranspose();
                       });
  }
  [[nodiscard]] bool SupportsMatrix() const noexcept override
  {
    return std::all_of(
      factors_.begin(), factors_.end(),
      [](const FactorAtPoints& f) { return f.value_evaluator == nullptr; });
  }

  /// @throws pcms_error for Coordinates values on a periodic axis
  void EvaluateImpl(const Field<Real>& field,
                    ValueView<Real, DeviceMemorySpace> values) const override
  {
    const auto periods =
      detail::CoordinatePeriods(field.GetData().GetValueBasis());
    if (periods[0] > 0 || periods[1] > 0 || periods[2] > 0) {
      throw pcms_error("PointEvaluator::Evaluate: a product space does not "
                       "evaluate Coordinates on a periodic axis");
    }
    const auto dofs = field.GetDOFHolderData().GetValues();
    const auto out = values.GetValues();
    if (chain_) {
      if (Operator() != DifferentialOperator::Gradient) {
        ScatterChain(MakeConstRank2View(Chain(dofs, -1)), rows_.extent,
                     rows_.num_factors, -1, 1, 0, out);
        return;
      }
      int offset = 0;
      for (int g = 0; g < rows_.num_factors; ++g) {
        ScatterChain(MakeConstRank2View(Chain(dofs, g)), rows_.extent,
                     rows_.num_factors, g, rows_.dimension[g], offset, out);
        offset += rows_.dimension[g];
      }
      return;
    }
    Scratch z;
    Scratch zg;
    if (rows_.opaque >= 0) {
      const auto& factor = factors_[rows_.opaque];
      z = Opaque(*factor.value_evaluator, 1, dofs);
      if (factor.gradient_evaluator != nullptr) {
        zg = Opaque(*factor.gradient_evaluator, rows_.dimension[rows_.opaque],
                    dofs);
      }
    }
    ApplyFactored(rows_, found_, static_cast<int>(out.extent(1)), dofs, z, zg,
                  out);
  }

  void EvaluateTransposeImpl(Rank2View<const Real, DeviceMemorySpace> values,
                             Field<Real>& functional) const override
  {
    detail::CheckLinear(functional.GetData().GetValueBasis());
    if (chain_) {
      Extents current = rows_.extent;
      ConstValues in = values;
      Scratch data;
      Scratch previous;
      for (auto k = order_.rbegin(); k != order_.rend(); ++k) {
        const LO left = Left(current, *k);
        const LO right = Right(current, *k);
        data = Scratch("product_chain",
                       static_cast<std::size_t>(left * Holders(*k) * right), 1);
        const auto& factor = factors_[*k];
        if (factor.value_evaluator != nullptr) {
          auto scratch = factor.space->CreateFunction<Real>();
          ApplyOpaqueTranspose(*factor.value_evaluator, scratch, left,
                               Holders(*k), right, in, data);
        } else {
          ApplyModeTranspose(factor.value, current[*k], left, Holders(*k),
                             right, in, data);
        }
        current[*k] = Holders(*k);
        previous = data;
        in = MakeConstRank2View(previous);
      }
      detail::AddToField(functional, data);
      return;
    }
    Scratch total("product_transpose", layout_->GetNumDofHolder(), 1);
    if (rows_.opaque < 0) {
      ApplyFactoredTranspose(rows_, found_, values, total);
    } else {
      const LO n = Holders(rows_.opaque);
      const LO left = layout_->GetNumDofHolder() / (n * rows_.right);
      Scratch z(
        "product_opaque_transpose",
        static_cast<std::size_t>(left * rows_.opaque_points * rows_.right), 1);
      ApplyFactoredTranspose(rows_, found_, values, z);
      const auto& factor = factors_[rows_.opaque];
      auto scratch = factor.space->CreateFunction<Real>();
      ApplyOpaqueTranspose(*factor.value_evaluator, scratch, left, n,
                           rows_.right, MakeConstRank2View(z), total);
    }
    detail::AddToField(functional, total);
  }

  /// @brief The Kronecker rows of all factors, built on the first call
  [[nodiscard]] PointMatrix<Real> GetMatrixImpl() const override
  {
    if (!kronecker_.has_value()) {
      int num_out = 1;
      if (Operator() == DifferentialOperator::Gradient) {
        num_out = 0;
        for (int k = 0; k < rows_.num_factors; ++k) {
          num_out += rows_.dimension[k];
        }
      }
      kronecker_ = KroneckerRows(rows_, found_, NumPoints(), num_out);
    }
    return detail::ToPointMatrix(*kronecker_, detail::DeviceGids(*layout_));
  }

private:
  [[nodiscard]] LO Holders(int k) const
  {
    return layout_->Factors()[k]->GetNumDofHolder();
  }

  /// @return the product of the extents before factor k
  [[nodiscard]] static LO Left(const Extents& extent, int k)
  {
    LO left = 1;
    for (int m = 0; m < k; ++m) {
      left *= extent[m];
    }
    return left;
  }
  /// @return the product of the extents after factor k
  [[nodiscard]] LO Right(const Extents& extent, int k) const
  {
    LO right = 1;
    for (int m = k + 1; m < rows_.num_factors; ++m) {
      right *= extent[m];
    }
    return right;
  }

  /// @return the evaluate-only factor's `evaluator`, of `m` outputs, applied
  ///         along its mode of the DOFs: [left][points][m][right]
  [[nodiscard]] Scratch Opaque(const PointEvaluator<Real>& evaluator, int m,
                               ConstValues dofs) const
  {
    const int k = rows_.opaque;
    const LO left = layout_->GetNumDofHolder() / (Holders(k) * rows_.right);
    Scratch out(
      "product_opaque",
      static_cast<std::size_t>(left * rows_.opaque_points * m * rows_.right),
      1);
    auto scratch = factors_[k].space->CreateFunction<Real>();
    ApplyOpaque(evaluator, scratch, left, Holders(k), rows_.right, dofs, out);
    return out;
  }

  /// @return the values (g < 0) or the partials of factor g at the tensor
  ///         points: [q_1]...[q_g][d_g]...[q_N]
  [[nodiscard]] Scratch Chain(ConstValues dofs, int g) const
  {
    Extents current = {};
    for (int k = 0; k < rows_.num_factors; ++k) {
      current[k] = Holders(k);
    }
    ConstValues in = dofs;
    Scratch data;
    Scratch previous;
    for (const int k : order_) {
      const LO rows = rows_.extent[k] * (k == g ? rows_.dimension[k] : 1);
      const LO left = Left(current, k);
      const LO right = Right(current, k);
      data = Scratch(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "product_chain"),
        static_cast<std::size_t>(left * rows * right), 1);
      const auto& factor = factors_[k];
      if (factor.value_evaluator != nullptr) {
        auto scratch = factor.space->CreateFunction<Real>();
        ApplyOpaque(k == g ? *factor.gradient_evaluator
                           : *factor.value_evaluator,
                    scratch, left, current[k], right, in, data);
      } else {
        ApplyMode(k == g ? factor.gradient : factor.value, rows, left,
                  current[k], right, in, data);
      }
      current[k] = rows;
      previous = data;
      in = MakeConstRank2View(previous);
    }
    return data;
  }

  std::shared_ptr<const ProductLayout> layout_;
  std::vector<FactorAtPoints> factors_;
  FactorRows rows_;
  Kokkos::View<bool*, DeviceMemorySpace> found_;
  // the modes of a tensor chain in order: evaluate-only factors first
  std::vector<int> order_;
  bool chain_ = false;
  mutable std::optional<PointRows> kronecker_;
};

/// The evaluator source of a product space: an evaluator of each leaf at its
/// bound points, their rows where they provide a matrix.
class ProductEvaluatorFactory final
  : public detail::TypedPointEvaluatorFactory<Real, ProductBinding>
{
public:
  ProductEvaluatorFactory(
    std::shared_ptr<const Geometry> geometry,
    std::shared_ptr<const ProductLayout> layout,
    std::vector<std::shared_ptr<const FunctionSpace>> leaves,
    std::set<DifferentialOperator> supported)
    : TypedPointEvaluatorFactory<Real, ProductBinding>(std::move(geometry),
                                                       std::move(supported)),
      layout_(std::move(layout)),
      leaves_(std::move(leaves))
  {
    for (const auto& leaf : leaves_) {
      geometries_.push_back(leaf->GetGeometry());
    }
  }

  [[nodiscard]] bool HasNodalDOFs() const noexcept override
  {
    return std::all_of(leaves_.begin(), leaves_.end(),
                       [](const auto& leaf) { return leaf->HasNodalDOFs(); });
  }

protected:
  std::unique_ptr<PointEvaluator<Real>> CreateEvaluator(
    std::shared_ptr<const ProductBinding> bound,
    DifferentialOperator op) const override
  {
    const auto leaves = detail::Flatten(*bound, geometries_);
    std::vector<FactorAtPoints> factors(leaves_.size());
    for (std::size_t k = 0; k < leaves_.size(); ++k) {
      auto& f = factors[k];
      f.space = leaves_[k];
      auto value = f.space->CreatePointEvaluator<Real>(leaves.factors[k]);
      std::unique_ptr<PointEvaluator<Real>> partials;
      if (op == DifferentialOperator::Gradient) {
        partials = f.space->CreatePointEvaluator<Real>(
          leaves.factors[k], DifferentialOperator::Gradient);
      }
      if (!value->SupportsMatrix()) {
        f.value_evaluator = std::move(value);
        f.gradient_evaluator = std::move(partials);
        continue;
      }
      const auto m = value->GetMatrix();
      f.value = {m.row_offsets, m.holders, m.values};
      if (partials != nullptr) {
        const auto g = partials->GetMatrix();
        f.gradient = {
          g.row_offsets, g.holders, g.values,
          leaves.factors[k]->GetGeometry()->GetCoordinateSystem()->Dimension()};
      }
    }
    return std::make_unique<ProductPointEvaluator>(
      std::move(bound), layout_, op, leaves, std::move(factors));
  }

private:
  std::shared_ptr<const ProductLayout> layout_;
  std::vector<std::shared_ptr<const FunctionSpace>> leaves_;
  std::vector<std::shared_ptr<const Geometry>> geometries_;
};

/// A product space, which keeps its leaf factors; its quadrature is the
/// tensor product of its factors'.
class ProductSpace final : public FunctionSpace
{
public:
  ProductSpace(std::shared_ptr<const FieldLayout> layout,
               std::shared_ptr<const PointEvaluatorFactory> source,
               std::shared_ptr<const ProductGeometry> geometry,
               std::shared_ptr<const FunctionSpace> a,
               std::shared_ptr<const FunctionSpace> b,
               std::vector<std::shared_ptr<const FunctionSpace>> leaves)
    : FunctionSpace(std::move(layout), std::move(source), geometry),
      geometry_(std::move(geometry)),
      a_(std::move(a)),
      b_(std::move(b)),
      leaves_(std::move(leaves))
  {
  }

  [[nodiscard]] const std::vector<std::shared_ptr<const FunctionSpace>>&
  Leaves() const noexcept
  {
    return leaves_;
  }

protected:
  [[nodiscard]] Quadrature CreateQuadrature() const override
  {
    return geometry_->TensorQuadrature(a_->GetQuadrature(),
                                       b_->GetQuadrature());
  }

private:
  std::shared_ptr<const ProductGeometry> geometry_;
  std::shared_ptr<const FunctionSpace> a_;
  std::shared_ptr<const FunctionSpace> b_;
  std::vector<std::shared_ptr<const FunctionSpace>> leaves_;
};

/// @return the leaf factors of `space` if it is a product space, else {space}
std::vector<std::shared_ptr<const FunctionSpace>> LeafFactors(
  const std::shared_ptr<const FunctionSpace>& space)
{
  const auto* product = dynamic_cast<const ProductSpace*>(space.get());
  return product != nullptr
           ? product->Leaves()
           : std::vector<std::shared_ptr<const FunctionSpace>>{space};
}

} // namespace

std::shared_ptr<const FunctionSpace> CreateProductSpace(
  std::shared_ptr<const FunctionSpace> a,
  std::shared_ptr<const FunctionSpace> b,
  std::shared_ptr<const Geometry> geometry)
{
  if (a == nullptr || b == nullptr) {
    throw pcms_error("CreateProductSpace: the factors must not be null");
  }
  if (geometry == nullptr) {
    geometry = CreateProductGeometry(a->GetGeometry(), b->GetGeometry());
  }
  const auto product =
    std::dynamic_pointer_cast<const ProductGeometry>(geometry);
  if (product == nullptr || product->FactorA() != a->GetGeometry() ||
      product->FactorB() != b->GetGeometry()) {
    throw pcms_error("CreateProductSpace: the geometry must be the product "
                     "geometry of the factors' geometries");
  }
  if (a->GetLayout()->GetNumComponents() != 1 ||
      b->GetLayout()->GetNumComponents() != 1) {
    throw pcms_error("CreateProductSpace: the factors must be scalar");
  }
  auto leaves = LeafFactors(a);
  const auto right = LeafFactors(b);
  leaves.insert(leaves.end(), right.begin(), right.end());
  if (leaves.size() > static_cast<std::size_t>(MaxFactors)) {
    throw pcms_error("CreateProductSpace: a product has at most 6 factors");
  }
  std::vector<std::shared_ptr<const FieldLayout>> layouts;
  std::set<DifferentialOperator> supported = {DifferentialOperator::Gradient};
  for (const auto& leaf : leaves) {
    layouts.push_back(leaf->GetLayout());
    if (!leaf->Supports(DifferentialOperator::Gradient)) {
      supported.clear();
    }
  }
  auto layout = std::make_shared<ProductLayout>(geometry, std::move(layouts));
  auto source = std::make_shared<ProductEvaluatorFactory>(
    geometry, layout, leaves, std::move(supported));
  return std::make_shared<ProductSpace>(std::move(layout), std::move(source),
                                        product, a, b, std::move(leaves));
}

} // namespace pcms
