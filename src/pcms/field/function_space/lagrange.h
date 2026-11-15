#ifndef PCMS_LAGRANGE_FIELD_FACTORY_H
#define PCMS_LAGRANGE_FIELD_FACTORY_H

#include <Omega_h_mesh.hpp>
#include "pcms/configuration.h"
#include "pcms/field/field.h"
#include "pcms/field/field_layout.h"
#include "pcms/field/function_space.h"
#include "pcms/field/field_data.h"
#include "pcms/field/value_view.hpp"
#include "pcms/field/point_evaluator.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/coordinate_system.hpp"
#include "pcms/utility/arrays.h"
#include "pcms/utility/types.h"
#include "pcms/utility/memory_spaces.h"
#include "pcms/utility/uniform_grid.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace pcms
{

class OmegaHGeometry;
template <unsigned Dim>
class UniformGridGeometry;

class LagrangeFunctionSpace : public FunctionSpace
{
  // Passkey: only members (the From* factories) can create a Key, so only they
  // can construct a space, while std::make_shared can still call the ctor.
  struct Key
  {
    explicit Key() = default;
  };

public:
  LagrangeFunctionSpace(
    Key, std::shared_ptr<const FieldLayout> layout,
    std::function<FieldDataVariant(Type, ValueBasis)> create_field_data_fn,
    std::shared_ptr<const PointEvaluatorFactory> evaluator) noexcept;

  enum class Backend
  {
    MeshFields,
    OmegaH
  };

#ifdef PCMS_ENABLE_MESHFIELDS
  static constexpr Backend DefaultBackend = Backend::MeshFields;
#else
  static constexpr Backend DefaultBackend = Backend::OmegaH;
#endif

  /// @brief A Lagrange space on the mesh of an Omega_h geometry; dispatches
  ///        to MeshFields or native Omega_h backend. Spaces on one geometry
  ///        accept each other's bound sources.
  [[nodiscard]] static std::shared_ptr<LagrangeFunctionSpace> FromGeometry(
    std::shared_ptr<const OmegaHGeometry> geometry, int order,
    int num_components, std::string global_id_name = "global",
    Backend backend = DefaultBackend, std::string layout_name = "");

  [[nodiscard]] static std::shared_ptr<LagrangeFunctionSpace> FromGeometry(
    std::shared_ptr<const OmegaHGeometry> geometry, int order,
    int num_components, Omega_h::Read<Omega_h::I8> owned_mask,
    std::string global_id_name = "global", Backend backend = DefaultBackend,
    std::string layout_name = "");

  /// @brief A Lagrange space of order 0 or 1 on the grid of a uniform-grid
  ///        geometry; the arguments are in the order of the Omega_h overload.
  [[nodiscard]] static std::shared_ptr<LagrangeFunctionSpace> FromGeometry(
    std::shared_ptr<const UniformGridGeometry<2>> geometry, int order,
    int num_components, std::string layout_name = "");

  [[nodiscard]] static std::shared_ptr<LagrangeFunctionSpace> FromGeometry(
    std::shared_ptr<const UniformGridGeometry<3>> geometry, int order,
    int num_components, std::string layout_name = "");

  [[nodiscard]] std::shared_ptr<const FieldLayout> GetLayout()
    const noexcept override;

protected:
  [[nodiscard]] FieldVariant CreateFieldImpl(Type storage_type,
                                             ValueBasis basis) const override;

  [[nodiscard]] FieldVariant CreateFieldImpl(
    FieldDataVariant data) const override;

private:
  std::shared_ptr<const FieldLayout> layout_;
  std::function<FieldDataVariant(Type, ValueBasis)> create_field_data_fn_;
};

} // namespace pcms

#endif // PCMS_LAGRANGE_FIELD_FACTORY_H
