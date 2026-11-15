#include "pcms/field/function_space/lagrange.h"
#include "pcms/configuration.h"
#include "pcms/utility/assert.h"
#include "pcms/utility/common.h"
#ifdef PCMS_ENABLE_MESHFIELDS
#include "pcms/field/layout/mesh_fields.h"
#include "pcms/field/evaluator/mesh_fields.h"
#include "pcms/field/data/mesh_fields.h"
#endif
#include "pcms/field/evaluator/omega_h_geometry.hpp"
#include "pcms/field/evaluator/uniform_grid_geometry.hpp"
#include "pcms/field/layout/omega_h_lagrange.h"
#include "pcms/field/evaluator/omega_h_lagrange.h"
#include "pcms/field/layout/uniform_grid.h"
#include "pcms/field/evaluator/uniform_grid.h"
#include "pcms/field/data/omega_h.h"
#include "pcms/field/data/simple.h"
#include "pcms/utility/uniform_grid.h"

#include <array>
#include <stdexcept>
#include <string>

namespace pcms
{

namespace
{

template <typename T>
void ValidateLagrangeWrappedFieldData(const FieldLayout& layout,
                                      const FieldData<T>& data)
{
#ifdef PCMS_ENABLE_MESHFIELDS
  if (dynamic_cast<const MeshFieldsAdapterLayout*>(&layout) != nullptr) {
    if (dynamic_cast<const MeshFieldsFieldData<T>*>(&data) == nullptr) {
      throw pcms_error(
        "LagrangeFunctionSpace::CreateField: MeshFields layout requires "
        "MeshFieldsFieldData");
    }
  } else
#endif
    if (dynamic_cast<const OmegaHLagrangeLayout*>(&layout) != nullptr) {
    if (dynamic_cast<const OmegaHFieldData<T>*>(&data) == nullptr) {
      throw pcms_error(
        "LagrangeFunctionSpace::CreateField: OmegaH layout requires "
        "OmegaHFieldData");
    }
  } else {
    if (dynamic_cast<const SimpleFieldData<T>*>(&data) == nullptr) {
      throw pcms_error(
        "LagrangeFunctionSpace::CreateField: this backend requires "
        "SimpleFieldData");
    }
  }

  if (data.GetDOFAccess()->GetValues().size() !=
      detail::ExpectedFlatFieldDataSize(layout)) {
    throw pcms_error(
      "LagrangeFunctionSpace::CreateField: field data size does not match "
      "layout");
  }
}

using Backend = LagrangeFunctionSpace::Backend;

// The discretization of an Omega_h geometry, which the layouts of the spaces
// on it share (OmegaHGeometry creates an OmegaHDiscretization).
std::shared_ptr<const OmegaHDiscretization> GeometryDiscretization(
  const OmegaHGeometry& geometry)
{
  return std::static_pointer_cast<const OmegaHDiscretization>(
    geometry.GetDiscretization());
}

/// Checks that `backend` supports Lagrange order `order` (and an owned mask
/// if `owned_mask`) before anything is built.
/// @param caller name of the calling factory, for the messages
/// @return the backend to use: MeshFields falls back to OmegaH for order 0
Backend CheckOrderAndBackend(const std::string& caller, int order,
                             Backend backend, bool owned_mask)
{
  // https://github.com/SCOREC/meshFields/issues/88
  if (backend == Backend::MeshFields && order == 0 && !owned_mask) {
    // MeshFields does not support order-0 fields; fall back to OmegaH.
    backend = Backend::OmegaH;
  }
  if (backend == Backend::MeshFields) {
    if (owned_mask) {
      throw pcms_error("LagrangeFunctionSpace::" + caller +
                       ": owned-mask construction is only supported by the "
                       "OmegaH backend");
    }
#ifdef PCMS_ENABLE_MESHFIELDS
    if (order != 1 && order != 2) {
      throw pcms_error("Unimplemented Lagrange order");
    }
#else
    throw pcms_error("LagrangeFunctionSpace::" + caller +
                     ": MeshFields backend requested but "
                     "PCMS_ENABLE_MESHFIELDS is not set");
#endif
    return backend;
  }
  if (order > 1) {
    throw pcms_error("LagrangeFunctionSpace::" + caller +
                     ": OmegaH backend only supports Lagrange orders 0 and 1");
  }
  return backend;
}

} // namespace

LagrangeFunctionSpace::LagrangeFunctionSpace(
  Key, std::shared_ptr<const FieldLayout> layout,
  std::function<FieldDataVariant(Type, ValueBasis)> create_field_data_fn,
  std::shared_ptr<const PointEvaluatorFactory> evaluator) noexcept
  : FunctionSpace(std::move(evaluator)),
    layout_(std::move(layout)),
    create_field_data_fn_(std::move(create_field_data_fn))
{
}

std::shared_ptr<LagrangeFunctionSpace> LagrangeFunctionSpace::FromGeometry(
  std::shared_ptr<const OmegaHGeometry> geometry, int order, int num_components,
  std::string global_id_name, Backend backend, std::string layout_name)
{
  PCMS_ALWAYS_ASSERT(geometry != nullptr);
  backend =
    CheckOrderAndBackend("FromGeometry", order, backend, /*owned_mask=*/false);
  Omega_h::Mesh& mesh = geometry->GetMesh();
  const auto& coordinate_system = geometry->GetCoordinateSystem();
  if (backend == Backend::MeshFields) {
#ifdef PCMS_ENABLE_MESHFIELDS
    const std::array<int, 4> nodes_per_dim = order == 1
                                               ? std::array<int, 4>{1, 0, 0, 0}
                                               : std::array<int, 4>{1, 1, 0, 0};
    auto mesh_layout = std::make_shared<MeshFieldsAdapterLayout>(
      mesh, nodes_per_dim, num_components, coordinate_system,
      GeometryDiscretization(*geometry), std::move(global_id_name));
    mesh_layout->SetName(layout_name);
    auto eval_factory =
      std::make_shared<MeshFieldsEvaluatorFactory<Real>>(geometry, mesh_layout);
    return std::make_shared<LagrangeFunctionSpace>(
      Key{}, mesh_layout,
      [mesh_layout](Type t, ValueBasis basis) -> FieldDataVariant {
        if (t == Type::Float) {
          if constexpr (std::is_same_v<MeshField::Real4, float> ||
                        std::is_same_v<MeshField::Real8, float>) {
            return std::make_unique<MeshFieldsFieldData<float>>(
              mesh_layout, std::move(basis));
          }
          throw pcms_error(
            "LagrangeFunctionSpace: MeshFields backend does not support "
            "float in this build");
        }
        if (t == Type::Real) {
          return std::make_unique<MeshFieldsFieldData<double>>(
            mesh_layout, std::move(basis));
        }
        throw pcms_error(
          "LagrangeFunctionSpace: MeshFields backend only supports the "
          "MeshFields scalar types enabled in this build");
      },
      std::move(eval_factory));
#endif
  }
  auto layout = std::make_shared<OmegaHLagrangeLayout>(
    mesh, order, num_components, coordinate_system,
    GeometryDiscretization(*geometry), std::move(global_id_name));
  layout->SetName(std::move(layout_name));
  auto eval_factory = std::make_shared<OmegaHLagrangeEvaluatorFactory<Real>>(
    std::move(geometry), layout);
  return std::make_shared<LagrangeFunctionSpace>(
    Key{}, layout,
    [layout](Type t, ValueBasis basis) -> FieldDataVariant {
      return apply_to_type(t, [&](auto tag) -> FieldDataVariant {
        using T = typename decltype(tag)::type;
        if constexpr (std::is_same_v<T, int8_t>) {
          throw pcms_error("LagrangeFunctionSpace: int8_t is not supported");
        } else {
          return std::make_unique<OmegaHFieldData<T>>(layout, basis);
        }
      });
    },
    std::move(eval_factory));
}

std::shared_ptr<LagrangeFunctionSpace> LagrangeFunctionSpace::FromGeometry(
  std::shared_ptr<const OmegaHGeometry> geometry, int order, int num_components,
  Omega_h::Read<Omega_h::I8> owned_mask, std::string global_id_name,
  Backend backend, std::string layout_name)
{
  PCMS_ALWAYS_ASSERT(geometry != nullptr);
  CheckOrderAndBackend("FromGeometry", order, backend, /*owned_mask=*/true);
  Omega_h::Mesh& mesh = geometry->GetMesh();
  const auto& coordinate_system = geometry->GetCoordinateSystem();
  auto layout = std::make_shared<OmegaHLagrangeLayout>(
    mesh, order, num_components, coordinate_system, std::move(owned_mask),
    GeometryDiscretization(*geometry), std::move(global_id_name));
  layout->SetName(std::move(layout_name));
  auto eval_factory = std::make_shared<OmegaHLagrangeEvaluatorFactory<Real>>(
    std::move(geometry), layout);
  return std::make_shared<LagrangeFunctionSpace>(
    Key{}, layout,
    [layout](Type t, ValueBasis basis) -> FieldDataVariant {
      return apply_to_type(t, [&](auto tag) -> FieldDataVariant {
        using T = typename decltype(tag)::type;
        if constexpr (std::is_same_v<T, int8_t>) {
          throw pcms_error("LagrangeFunctionSpace: int8_t is not supported");
        } else {
          return std::make_unique<OmegaHFieldData<T>>(layout, basis);
        }
      });
    },
    std::move(eval_factory));
}

std::shared_ptr<LagrangeFunctionSpace> LagrangeFunctionSpace::FromGeometry(
  std::shared_ptr<const UniformGridGeometry<2>> geometry, int order,
  int num_components, std::string layout_name)
{
  PCMS_ALWAYS_ASSERT(geometry != nullptr);
  if (order != 0 && order != 1) {
    throw std::invalid_argument("LagrangeFunctionSpace::FromGeometry: only "
                                "orders 0 and 1 are supported");
  }
  auto ug_layout = std::make_shared<UniformGridFieldLayout<2>>(
    geometry->GetGrid(), num_components, geometry->GetCoordinateSystem(),
    order);
  ug_layout->SetName(std::move(layout_name));
  auto eval_factory = std::make_shared<UniformGridEvaluatorFactory<2>>(
    std::move(geometry), ug_layout);
  return std::make_shared<LagrangeFunctionSpace>(
    Key{}, ug_layout,
    [ug_layout](Type t, ValueBasis basis) -> FieldDataVariant {
      return apply_to_type(t, [&](auto tag) -> FieldDataVariant {
        using T = typename decltype(tag)::type;
        if constexpr (std::is_same_v<T, int8_t>) {
          throw pcms_error("LagrangeFunctionSpace: int8_t is not supported");
        } else {
          return std::make_unique<SimpleFieldData<T>>(ug_layout, basis);
        }
      });
    },
    std::move(eval_factory));
}

std::shared_ptr<LagrangeFunctionSpace> LagrangeFunctionSpace::FromGeometry(
  std::shared_ptr<const UniformGridGeometry<3>> geometry, int order,
  int num_components, std::string layout_name)
{
  PCMS_ALWAYS_ASSERT(geometry != nullptr);
  if (order != 0 && order != 1) {
    throw std::invalid_argument("LagrangeFunctionSpace::FromGeometry: only "
                                "orders 0 and 1 are supported");
  }
  auto ug_layout = std::make_shared<UniformGridFieldLayout<3>>(
    geometry->GetGrid(), num_components, geometry->GetCoordinateSystem(),
    order);
  ug_layout->SetName(std::move(layout_name));
  auto eval_factory = std::make_shared<UniformGridEvaluatorFactory<3>>(
    std::move(geometry), ug_layout);
  return std::make_shared<LagrangeFunctionSpace>(
    Key{}, ug_layout,
    [ug_layout](Type t, ValueBasis basis) -> FieldDataVariant {
      return apply_to_type(t, [&](auto tag) -> FieldDataVariant {
        using T = typename decltype(tag)::type;
        if constexpr (std::is_same_v<T, int8_t>) {
          throw pcms_error("LagrangeFunctionSpace: int8_t is not supported");
        } else {
          return std::make_unique<SimpleFieldData<T>>(ug_layout, basis);
        }
      });
    },
    std::move(eval_factory));
}

std::shared_ptr<const FieldLayout> LagrangeFunctionSpace::GetLayout()
  const noexcept
{
  return layout_;
}

FieldVariant LagrangeFunctionSpace::CreateFieldImpl(Type storage_type,
                                                    ValueBasis basis) const
{
  auto field_data = create_field_data_fn_(storage_type, std::move(basis));
  return std::visit(
    [this](auto&& fd) -> FieldVariant {
      using FD = std::decay_t<decltype(fd)>;
      using T = typename FD::element_type::value_type;
      return WrapField<T>(layout_, std::forward<decltype(fd)>(fd));
    },
    std::move(field_data));
}

FieldVariant LagrangeFunctionSpace::CreateFieldImpl(FieldDataVariant data) const
{
  return std::visit(
    [this](auto&& fd) -> FieldVariant {
      using FD = std::decay_t<decltype(fd)>;
      using T = typename FD::element_type::value_type;
      PCMS_ALWAYS_ASSERT(fd != nullptr);
      if constexpr (std::is_same_v<T, int8_t>) {
        throw pcms_error(
          "LagrangeFunctionSpace: int8_t is not a supported field type");
      } else {
        ValidateLagrangeWrappedFieldData(*layout_, *fd);
        return WrapField<T>(layout_, std::forward<decltype(fd)>(fd));
      }
    },
    std::move(data));
}

} // namespace pcms
