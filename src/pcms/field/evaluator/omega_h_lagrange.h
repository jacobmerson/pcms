#ifndef PCMS_OMEGA_H_LAGRANGE_EVALUATOR_FACTORY_H
#define PCMS_OMEGA_H_LAGRANGE_EVALUATOR_FACTORY_H

#include "pcms/field/layout/omega_h_lagrange.h"
#include "pcms/field/point_evaluator_factory.hpp"
#include "pcms/field/point_evaluator.h"
#include "pcms/field/field_data.h"
#include "pcms/field/data/omega_h.h"
#include "pcms/localization/point_search.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/arrays.h"
#include "pcms/utility/profile.h"
#include "pcms/utility/types.h"

#include <Omega_h_mesh.hpp>
#include <Kokkos_Core.hpp>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <variant>

namespace pcms
{

namespace detail
{

/// @brief Barycentric coordinates of points in their owning elements.
/// @tparam Dim mesh dimension
/// @param mesh simplex mesh
/// @param points query points, one row per point
/// @param rows points to compute, as indices into `points`
/// @param owning_ids owning element of each point
/// @return owning element and barycentric coordinates ([row][vertex]) of each
///         row
template <int Dim>
std::pair<Kokkos::View<LO*, DeviceMemorySpace>,
          Kokkos::View<Real**, DeviceMemorySpace>>
BarycentricInElements(Omega_h::Mesh& mesh,
                      Kokkos::View<const Real**, DeviceMemorySpace> points,
                      Kokkos::View<const LO*, DeviceMemorySpace> rows,
                      Kokkos::View<const LO*, DeviceMemorySpace> owning_ids)
{
  const auto n = static_cast<LO>(rows.extent(0));
  Kokkos::View<LO*, DeviceMemorySpace> elem_ids("omega_h_elem_ids", n);
  Kokkos::View<Real**, DeviceMemorySpace> bary("omega_h_bary", n, Dim + 1);
  const auto elem_verts = mesh.ask_elem_verts();
  const auto mesh_coords = mesh.coords();
  Kokkos::parallel_for(
    "BarycentricInElements",
    Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, n),
    KOKKOS_LAMBDA(const LO k) {
      const LO i = rows(k);
      const LO elem = owning_ids(i);
      elem_ids(k) = elem;
      Omega_h::Few<Omega_h::Vector<Dim>, Dim + 1> vertex_coords;
      for (int v = 0; v < Dim + 1; ++v) {
        vertex_coords[v] = Omega_h::get_vector<Dim>(
          mesh_coords, elem_verts[elem * (Dim + 1) + v]);
      }
      Omega_h::Vector<Dim> point;
      for (int d = 0; d < Dim; ++d) {
        point[d] = points(i, d);
      }
      const auto local =
        Omega_h::barycentric_from_global<Dim, Dim>(point, vertex_coords);
      for (int v = 0; v < Dim + 1; ++v) {
        bary(k, v) = local[v];
      }
    });
  return {elem_ids, bary};
}

inline std::variant<GridPointSearch2D, GridPointSearch3D> MakeSearch(
  Omega_h::Mesh& mesh)
{
  if (mesh.dim() == 2) {
    const auto n = DivisionsPerAxisForMesh<2>(mesh.nelems());
    return GridPointSearch2D(mesh, n, n);
  } else if (mesh.dim() == 3) {
    const auto n = DivisionsPerAxisForMesh<3>(mesh.nelems());
    return GridPointSearch3D(mesh, n, n, n);
  }
  throw std::invalid_argument(
    "OmegaHLagrangeEvaluatorFactory: only 2D and 3D meshes are supported");
}

} // namespace detail

/// @brief Omega_h bound source: the element and barycentric coordinates of
///        each found query point.
class OmegaHLagrangeBinding final : public BoundSource
{
public:
  /// @param request query points and out-of-bounds policy
  /// @param search containing-element search over `mesh`
  /// @param mesh simplex mesh of the evaluated fields
  OmegaHLagrangeBinding(const BindRequest& request,
                        const GridPointSearchVariant& search,
                        Omega_h::Mesh& mesh)
    : BoundSource(request)
  {
    std::visit(
      [&](const auto& s) {
        constexpr int Dim = std::decay_t<decltype(s)>::DIM;
        auto [found, owning_ids] = FindOwningElements(s, PointValues());
        SetFound(found);
        std::tie(elem_ids_, bary_) = detail::BarycentricInElements<Dim>(
          mesh, PointValues(), FoundPoints(), owning_ids);
      },
      search);
  }

  /// @return element of each found point, in FoundPoints() order
  [[nodiscard]] Kokkos::View<const LO*, DeviceMemorySpace> ElementIds()
    const noexcept
  {
    return elem_ids_;
  }

  /// @return barycentric coordinates of each found point in its element
  ///         ([found point][vertex]), in FoundPoints() order
  [[nodiscard]] Kokkos::View<const Real**, DeviceMemorySpace> Barycentric()
    const noexcept
  {
    return bary_;
  }

private:
  Kokkos::View<LO*, DeviceMemorySpace> elem_ids_;
  Kokkos::View<Real**, DeviceMemorySpace> bary_;
};

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
} // namespace detail

// OmegaHLagrangePointEvaluator<T> implements PointEvaluator<T> for simplex
// meshes backed by Omega_h, using the element IDs and barycentric coordinates
// of the bound source.
//
// Output shape: [num_query_points][num_components].
//
// The values are read from the field's Omega_h tag
// (OmegaHFieldData::GetTagName). Its owned entries are always current; its
// ghost entries, which elements on the partition boundary read, are correct
// only after the field's last write was followed by Synchronize.
template <typename T,
          typename LayoutPolicy =
            detail::default_layout_for_memory_space_t<DeviceMemorySpace>>
class OmegaHLagrangePointEvaluator : public PointEvaluator<T, LayoutPolicy>
{
public:
  /// @param layout Omega_h Lagrange layout of the evaluated fields
  /// @param binding elements and barycentric coordinates of the query points
  OmegaHLagrangePointEvaluator(
    std::shared_ptr<const OmegaHLagrangeLayout> layout,
    std::shared_ptr<const OmegaHLagrangeBinding> binding)
    : PointEvaluator<T, LayoutPolicy>(binding, DifferentialOperator::Value,
                                      layout),
      layout_(std::move(layout)),
      binding_(std::move(binding))
  {
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
    const auto found = binding_->FoundPoints();
    const auto elem_ids = binding_->ElementIds();
    const auto bary = binding_->Barycentric();
    LO n_valid = static_cast<LO>(found.extent(0));
    int n_comp = layout_->GetNumComponents();

    if (layout_->GetOrder() == 0) {
      Kokkos::parallel_for(
        "OmegaHLagrangePointEvaluator::EvaluateOrder0",
        Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, n_valid),
        KOKKOS_LAMBDA(LO k) {
          for (int c = 0; c < n_comp; ++c) {
            values(found(k), c) = dof_data(elem_ids(k), c);
          }
        });

    } else {
      // Order-1: barycentric interpolation over element vertices
      Omega_h::Mesh& mesh = const_cast<Omega_h::Mesh&>(layout_->GetMesh());
      int mesh_dim = mesh.dim();
      int nvpe = mesh_dim + 1;
      auto elem_verts = mesh.ask_elem_verts();

      Kokkos::parallel_for(
        "OmegaHLagrangePointEvaluator::EvaluateOrder1",
        Kokkos::RangePolicy<DeviceMemorySpace::execution_space>(0, n_valid),
        KOKKOS_LAMBDA(LO k) {
          const LO elem = elem_ids(k);
          for (int c = 0; c < n_comp; ++c) {
            T val = T{};
            for (int v = 0; v < nvpe; ++v) {
              LO vert = elem_verts[elem * nvpe + v];
              val += static_cast<T>(bary(k, v)) * dof_data(vert, c);
            }
            values(found(k), c) = val;
          }
        });
    }
  }

private:
  std::shared_ptr<const OmegaHLagrangeLayout> layout_;
  std::shared_ptr<const OmegaHLagrangeBinding> binding_;
};

// ---------------------------------------------------------------------------
// EvaluatorFactory
// ---------------------------------------------------------------------------

/// @brief Point evaluation of Lagrange fields on an Omega_h simplex mesh.
/// @tparam T field value type
template <typename T>
class OmegaHLagrangeEvaluatorFactory
  : public TypedPointEvaluatorFactory<T, OmegaHLagrangeBinding>
{
  using Bound = OmegaHLagrangeBinding;

public:
  /// @param layout Omega_h Lagrange layout of the evaluated fields
  explicit OmegaHLagrangeEvaluatorFactory(
    std::shared_ptr<const OmegaHLagrangeLayout> layout)
    : TypedPointEvaluatorFactory<T, Bound>(layout->GetCoordinateSystem()),
      layout_(std::move(layout)),
      search_(detail::MakeSearch(layout_->GetMesh()))
  {
  }

  /// The containing-element search over this layout's mesh. Exposed so that
  /// operators which must locate points on the same mesh, such as the
  /// conservative transfer's intersection search, reuse it rather than build
  /// and hold a second copy of the candidate map.
  [[nodiscard]] const GridPointSearchVariant& GetSearch() const noexcept
  {
    return search_;
  }

protected:
  std::shared_ptr<Bound> BindPoints(const BindRequest& request) const override
  {
    PCMS_FUNCTION_TIMER;
    return std::make_shared<Bound>(
      request, search_, const_cast<Omega_h::Mesh&>(layout_->GetMesh()));
  }

  std::unique_ptr<PointEvaluator<T>> CreateEvaluator(
    std::shared_ptr<const Bound> bound,
    DifferentialOperator /*op*/) const override
  {
    return std::make_unique<OmegaHLagrangePointEvaluator<T>>(layout_,
                                                             std::move(bound));
  }

private:
  std::shared_ptr<const OmegaHLagrangeLayout> layout_;
  GridPointSearchVariant search_;
};

} // namespace pcms

#endif // PCMS_OMEGA_H_LAGRANGE_EVALUATOR_FACTORY_H
