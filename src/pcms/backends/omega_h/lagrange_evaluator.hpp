#ifndef PCMS_BACKENDS_OMEGA_H_LAGRANGE_EVALUATOR_HPP
#define PCMS_BACKENDS_OMEGA_H_LAGRANGE_EVALUATOR_HPP

#include "pcms/backends/omega_h/omega_h_geometry.hpp"
#include "pcms/backends/omega_h/lagrange_layout.hpp"
#include "pcms/field/point_evaluator_factory.hpp"
#include "pcms/field/point_evaluator.hpp"
#include "pcms/field/field_data.hpp"
#include "pcms/backends/omega_h/field_data.hpp"
#include "pcms/utility/assert.hpp"
#include "pcms/utility/arrays.hpp"
#include "pcms/utility/detail/profile.hpp"
#include "pcms/utility/types.hpp"

#include <Omega_h_mesh.hpp>
#include <Kokkos_Core.hpp>
#include <memory>
#include <utility>

namespace pcms::omega_h
{

// ---------------------------------------------------------------------------
// PointEvaluator
// ---------------------------------------------------------------------------

namespace detail
{
// DOF-holder values read from an Omega_h tag: one value per component per
// mesh entity, entity-major.
template <typename T, typename TagValue>
struct OmegaHLagrangeTagValues
{
  Omega_h::Read<TagValue> values;
  int num_components;
  KOKKOS_INLINE_FUNCTION T operator()(LO holder, int c) const
  {
    return static_cast<T>(values[holder * num_components + c]);
  }
};

/// The basis of an order 0 or 1 Lagrange field at the points of a binding of a
/// mesh of dimension Dim: row p of the evaluation matrix, `stride` entries per
/// point (1 for order 0, Dim + 1 for order 1). Order 0 takes the value of the
/// lowest element adjacent to the entity of the point's record, with weight
/// 1; order 1 interpolates on the vertices of that entity itself, with the
/// weights its reduced coordinates give, and picks no element. Entries of
/// the vertices an entity of lower dimension lacks, and of missing points,
/// have weight zero.
struct OmegaHLagrangeBasis
{
  int stride = 0;
  Kokkos::View<LO*, DeviceMemorySpace> holders; // [points * stride]
  Kokkos::View<Real*, DeviceMemorySpace> weights; // [points * stride]
};

/// (A free function: CUDA does not allow device lambdas in the public virtual
/// function that dispatches on the dimension.)
template <int Dim>
OmegaHLagrangeBasis BuildOmegaHLagrangeBasis(
  Omega_h::Mesh& mesh, int order,
  const pcms::detail::OmegaHMeshBinding& binding)
{
  const auto found = binding.FoundPoints();
  const auto results = binding.Results<Dim>();
  const auto n_valid = static_cast<LO>(found.extent(0));
  const auto up = UpAdjacencies<Dim>(mesh);
  const auto verts_of = VerticesOfEntities<Dim>(mesh);
  OmegaHLagrangeBasis basis;
  basis.stride = order == 0 ? 1 : Dim + 1;
  const int stride = basis.stride;
  const auto num_points = static_cast<std::size_t>(binding.NumPoints());
  basis.holders = Kokkos::View<LO*, DeviceMemorySpace>(
    "omega_h_lagrange_holders", num_points * stride);
  basis.weights = Kokkos::View<Real*, DeviceMemorySpace>(
    "omega_h_lagrange_weights", num_points * stride);
  const auto holders = basis.holders;
  const auto weights_out = basis.weights;
  pcms::detail::ForEach(
    "OmegaHLagrangePointEvaluator::BuildBasis", n_valid,
    KOKKOS_LAMBDA(const LO k) {
      const LO row = found(k);
      if (order == 0) {
        holders(row) = ElementOfResult<Dim>(up, results(row));
        weights_out(row) = 1.0;
        return;
      }
      LO vertices[Dim + 1];
      Real weights[Dim + 1];
      const int nv =
        EntityVertexWeights<Dim>(verts_of, results(row), vertices, weights);
      for (int v = 0; v < nv; ++v) {
        holders(row * stride + v) = vertices[v];
        weights_out(row * stride + v) = weights[v];
      }
    });
  return basis;
}

/// Evaluates an order 0 or 1 Lagrange field at the found points: the values
/// are the cached weights applied to the DOFs of the cached holders.
template <typename T, typename Dof, typename Values>
void EvaluateOmegaHLagrange(const OmegaHLagrangeBasis& basis,
                            const Kokkos::View<const LO*, DeviceMemorySpace>& found,
                            const Dof& dof_data, int n_comp,
                            const Values& values)
{
  const auto n_valid = static_cast<LO>(found.extent(0));
  const auto holders = basis.holders;
  const auto weights = basis.weights;
  const int stride = basis.stride;
  pcms::detail::ForEach(
    "OmegaHLagrangePointEvaluator::Evaluate", n_valid,
    KOKKOS_LAMBDA(const LO k) {
      const LO row = found(k);
      for (int c = 0; c < n_comp; ++c) {
        T val = T{};
        for (int v = 0; v < stride; ++v) {
          val += static_cast<T>(weights(row * stride + v)) *
                 dof_data(holders(row * stride + v), c);
        }
        values(row, c) = val;
      }
    });
}

/// The transpose of EvaluateOmegaHLagrange: adds each point's values, with the
/// cached weights, to the DOF holders it reads. Concurrent points that share a
/// holder add atomically.
template <typename T, typename Dofs, typename Values>
void EvaluateTransposeOmegaHLagrange(
  const OmegaHLagrangeBasis& basis,
  const Kokkos::View<const LO*, DeviceMemorySpace>& found, const Values& values,
  const Dofs& dofs, int n_comp)
{
  const auto n_valid = static_cast<LO>(found.extent(0));
  const auto holders = basis.holders;
  const auto weights = basis.weights;
  const int stride = basis.stride;
  pcms::detail::ForEach(
    "OmegaHLagrangePointEvaluator::EvaluateTranspose", n_valid,
    KOKKOS_LAMBDA(const LO k) {
      const LO row = found(k);
      for (int c = 0; c < n_comp; ++c) {
        for (int v = 0; v < stride; ++v) {
          Kokkos::atomic_add(&dofs(holders(row * stride + v), c),
                             static_cast<T>(weights(row * stride + v)) *
                               values(row, c));
        }
      }
    });
}

/// The values at the owned DOF holders of the contributions `dofs` of this
/// rank to every DOF holder, ghosts included, summed over the ranks
/// (Omega_h's reduction of the copies into their owners).
template <typename T>
Kokkos::View<T**, DeviceMemorySpace> SumGhostsIntoOwners(
  const OmegaHLagrangeLayout& layout,
  const Kokkos::View<T**, DeviceMemorySpace>& dofs)
{
  if (!layout.HasGhosts()) {
    return dofs;
  }
  using TagValue = utility::OmegaHValue_t<T>;
  const int n_comp = layout.GetNumComponents();
  Omega_h::Write<TagValue> contributions(
    static_cast<LO>(dofs.extent(0)) * n_comp);
  Kokkos::parallel_for(
    "SumGhostsIntoOwners::pack",
    Kokkos::RangePolicy<DefaultExecutionSpace>(0, static_cast<LO>(dofs.extent(0))),
    KOKKOS_LAMBDA(const LO i) {
      for (int c = 0; c < n_comp; ++c) {
        contributions[i * n_comp + c] = static_cast<TagValue>(dofs(i, c));
      }
    });
  const Omega_h::Read<TagValue> reduced = layout.GetMesh().reduce_array(
    layout.GetDOFHolderEntityDim(), Omega_h::Read<TagValue>(contributions),
    n_comp, OMEGA_H_SUM);
  const auto owned_indices = layout.GetOwnedIndices();
  Kokkos::View<T**, DeviceMemorySpace> owned("omega_h_lagrange_owned",
                                             owned_indices.extent(0), n_comp);
  Kokkos::parallel_for(
    "SumGhostsIntoOwners::unpack",
    Kokkos::RangePolicy<DefaultExecutionSpace>(
      0, static_cast<LO>(owned_indices.extent(0))),
    KOKKOS_LAMBDA(const LO o) {
      for (int c = 0; c < n_comp; ++c) {
        owned(o, c) = static_cast<T>(reduced[owned_indices(o) * n_comp + c]);
      }
    });
  return owned;
}

/// owned[o][c] += addend[o][c]
template <typename T>
Kokkos::View<T**, DeviceMemorySpace> AddOwned(
  const Rank2View<const T, DeviceMemorySpace>& current,
  const Kokkos::View<T**, DeviceMemorySpace>& addend)
{
  Kokkos::View<T**, DeviceMemorySpace> sum(
    Kokkos::view_alloc(Kokkos::WithoutInitializing, "omega_h_lagrange_sum"),
    addend.extent(0), addend.extent(1));
  const auto n = static_cast<int>(sum.extent(0));
  const auto nc = static_cast<int>(sum.extent(1));
  Kokkos::parallel_for(
    "OmegaHLagrangePointEvaluator::AddOwned",
    Kokkos::RangePolicy<DefaultExecutionSpace>(0, n * nc),
    KOKKOS_LAMBDA(const int k) {
      sum(k / nc, k % nc) = current(k / nc, k % nc) + addend(k / nc, k % nc);
    });
  return sum;
}

} // namespace detail

// OmegaHLagrangePointEvaluator<T> implements PointEvaluator<T> for simplex
// meshes backed by Omega_h, using the search records of the BoundPoints.
//
// Output shape: [num_query_points][num_components].
//
// The values are read from the field's Omega_h tag
// (OmegaHFieldData::GetTagName). Its owned entries are always current; its
// ghost entries, which elements on the partition boundary read, are correct
// only after the field's last write was followed by Synchronize.
template <typename T,
          typename LayoutPolicy =
            pcms::detail::default_layout_for_memory_space_t<DeviceMemorySpace>>
class OmegaHLagrangePointEvaluator : public PointEvaluator<T, LayoutPolicy>
{
public:
  /// @param layout Omega_h Lagrange layout of the evaluated fields
  /// @param binding the search records of the query points
  OmegaHLagrangePointEvaluator(
    std::shared_ptr<const OmegaHLagrangeLayout> layout,
    std::shared_ptr<const pcms::detail::OmegaHMeshBinding> binding)
    : PointEvaluator<T, LayoutPolicy>(binding, DifferentialOperator::Value,
                                      layout),
      layout_(std::move(layout)),
      binding_(std::move(binding))
  {
    // The basis at the points is computed once and serves Evaluate,
    // EvaluateTranspose and GetMatrix.
    Omega_h::Mesh& mesh = const_cast<Omega_h::Mesh&>(layout_->GetMesh());
    basis_ = mesh.dim() == 2 ? detail::BuildOmegaHLagrangeBasis<2>(
                                 mesh, layout_->GetOrder(), *binding_)
                             : detail::BuildOmegaHLagrangeBasis<3>(
                                 mesh, layout_->GetOrder(), *binding_);
  }

  // Public because CUDA forbids extended lambdas in non-public member
  // functions. User call Evaluate.
  // TODO: add passkey or other mechanism to protect this
  void EvaluateImpl(
    const Field<T>& field,
    ValueView<T, DeviceMemorySpace, LayoutPolicy> out) const override
  {
    PCMS_FUNCTION_TIMER;
    const auto values = out.GetValues();
    // Elements read the values of all their DOF holders, ghosts included, so
    // read the field's Omega_h tag rather than the owned values.
    const auto& data = dynamic_cast<const OmegaHFieldData<T>&>(field.GetData());
    using TagValue = typename OmegaHFieldData<T>::tag_value_type;
    const auto dof_data = detail::OmegaHLagrangeTagValues<T, TagValue>{
      layout_->GetMesh().get_array<TagValue>(layout_->GetDOFHolderEntityDim(),
                                             data.GetTagName()),
      layout_->GetNumComponents()};
    detail::EvaluateOmegaHLagrange<T>(basis_, binding_->FoundPoints(),
                                      dof_data, layout_->GetNumComponents(),
                                      values);
  }

  [[nodiscard]] bool SupportsTranspose() const noexcept override
  {
    return true;
  }

  // Public for the same reason as EvaluateImpl. The contributions are
  // accumulated per DOF holder, ghosts included, then the ghosts' are summed
  // into their owners (serial: nothing to sum).
  void EvaluateTransposeImpl(
    Rank2View<const T, DeviceMemorySpace, LayoutPolicy> values,
    Field<T>& functional) const override
  {
    PCMS_FUNCTION_TIMER;
    const int n_comp = layout_->GetNumComponents();
    Kokkos::View<T**, DeviceMemorySpace> dofs(
      "omega_h_lagrange_transpose",
      static_cast<std::size_t>(layout_->GetNumDofHolderWithGhosts()), n_comp);
    detail::EvaluateTransposeOmegaHLagrange<T>(
      basis_, binding_->FoundPoints(), values, dofs, n_comp);
    const auto owned = detail::SumGhostsIntoOwners<T>(*layout_, dofs);
    const auto sum =
      detail::AddOwned<T>(functional.GetDOFHolderData().GetValues(), owned);
    functional.SetDOFHolderDataUnchecked(MakeConstRank2View(sum));
    functional.Synchronize();
  }

  [[nodiscard]] bool SupportsMatrix() const noexcept override { return true; }

  PointMatrix<T> GetMatrixImpl() const override
  {
    const auto num_points = static_cast<LO>(binding_->NumPoints());
    const int stride = basis_.stride;
    Kokkos::View<LO*, DeviceMemorySpace> offsets("omega_h_lagrange_offsets",
                                                 num_points + 1);
    Kokkos::View<GO*, DeviceMemorySpace> columns("omega_h_lagrange_columns",
                                                 basis_.holders.extent(0));
    const auto holders = basis_.holders;
    const auto gids = layout_->GetGids();
    Kokkos::parallel_for(
      "OmegaHLagrangePointEvaluator::GetMatrix",
      Kokkos::RangePolicy<DefaultExecutionSpace>(0, num_points + 1),
      KOKKOS_LAMBDA(const LO p) {
        offsets(p) = p * stride;
        if (p < num_points) {
          for (int v = 0; v < stride; ++v) {
            columns(p * stride + v) = gids[holders(p * stride + v)];
          }
        }
      });
    Kokkos::View<T*, DeviceMemorySpace> values("omega_h_lagrange_values",
                                               basis_.weights.extent(0));
    const auto weights = basis_.weights;
    Kokkos::parallel_for(
      "OmegaHLagrangePointEvaluator::GetMatrixValues",
      Kokkos::RangePolicy<DefaultExecutionSpace>(
        0, static_cast<LO>(values.extent(0))),
      KOKKOS_LAMBDA(const LO i) { values(i) = static_cast<T>(weights(i)); });
    return PointMatrix<T>{offsets, columns, values};
  }

private:
  std::shared_ptr<const OmegaHLagrangeLayout> layout_;
  std::shared_ptr<const pcms::detail::OmegaHMeshBinding> binding_;
  detail::OmegaHLagrangeBasis basis_;
};

// ---------------------------------------------------------------------------
// EvaluatorFactory
// ---------------------------------------------------------------------------

/// @brief Point evaluation of Lagrange fields on an Omega_h simplex mesh.
/// @tparam T field value type
template <typename T>
class OmegaHLagrangeEvaluatorFactory
  : public pcms::detail::TypedPointEvaluatorFactory<
      T, pcms::detail::OmegaHMeshBinding>
{
public:
  /// @param geometry geometry of the layout's mesh
  /// @param layout Omega_h Lagrange layout of the evaluated fields
  OmegaHLagrangeEvaluatorFactory(
    std::shared_ptr<const OmegaHGeometry> geometry,
    std::shared_ptr<const OmegaHLagrangeLayout> layout)
    : pcms::detail::TypedPointEvaluatorFactory<T,
                                               pcms::detail::OmegaHMeshBinding>(
        std::move(geometry)),
      layout_(std::move(layout))
  {
  }

protected:
  std::unique_ptr<PointEvaluator<T>> CreateEvaluator(
    std::shared_ptr<const pcms::detail::OmegaHMeshBinding> bound,
    DifferentialOperator /*op*/) const override
  {
    return std::make_unique<OmegaHLagrangePointEvaluator<T>>(layout_,
                                                             std::move(bound));
  }

private:
  std::shared_ptr<const OmegaHLagrangeLayout> layout_;
};

} // namespace pcms::omega_h

#endif // PCMS_BACKENDS_OMEGA_H_LAGRANGE_EVALUATOR_HPP
