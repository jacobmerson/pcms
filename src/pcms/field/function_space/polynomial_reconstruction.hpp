#ifndef PCMS_POLYNOMIAL_RECONSTRUCTION_FUNCTION_SPACE_H
#define PCMS_POLYNOMIAL_RECONSTRUCTION_FUNCTION_SPACE_H

#include "pcms/field/field.h"
#include "pcms/field/field_layout.h"
#include "pcms/field/field_data.h"
#include "pcms/field/value_view.hpp"
#include "pcms/field/function_space.h"
#include "pcms/field/out_of_bounds_policy.h"
#include "pcms/field/coordinate_view.hpp"
#include "pcms/field/evaluator/mls_options.h"
#include "pcms/geometry/geometry.hpp"
#include "pcms/utility/arrays.h"
#include "pcms/utility/memory_spaces.h"

#include <Omega_h_mesh.hpp>
#include <memory>

namespace pcms
{

class PointCloudGeometry;

class PolynomialReconstructionFunctionSpace : public FunctionSpace
{
  // Passkey: only members (the Create/FromMesh/FromGeometry factories) can
  // create a Key, so
  // only they can construct a space, while std::make_shared can call the ctor.
  struct Key
  {
    explicit Key() = default;
  };

public:
  PolynomialReconstructionFunctionSpace(
    Key, std::shared_ptr<const FieldLayout> layout,
    std::shared_ptr<const PointEvaluatorFactory> evaluator) noexcept;

  /// @brief A space on a new point-cloud geometry of `coords`.
  /// @param options the reconstruction
  /// @param neighbourhood the support search of the new geometry
  /// @throws pcms_error if the coordinate system does not have the identity
  ///         metric (MLS distances are Euclidean)
  [[nodiscard]] static std::shared_ptr<PolynomialReconstructionFunctionSpace>
  Create(CoordinateView<HostMemorySpace> coords, MLSOptions options = {},
         NeighbourhoodOptions neighbourhood = {});

  /// @brief A space on a new geometry of the mesh entities of one dimension.
  /// @param options the reconstruction
  /// @param neighbourhood the support search of the new geometry
  /// @throws pcms_error if source_entity_dim is out of range or the
  ///         coordinate system does not have the identity metric
  [[nodiscard]] static std::shared_ptr<PolynomialReconstructionFunctionSpace>
  FromMesh(Omega_h::Mesh& mesh, int source_entity_dim,
           std::shared_ptr<const CoordinateSystem> coordinate_system,
           MLSOptions options = {}, NeighbourhoodOptions neighbourhood = {});

  /// @brief A space on an existing geometry, whose sites are its DOF
  ///        holders. Spaces on one geometry accept each other's bound
  ///        sources; the supports come from the geometry's search options.
  /// @param options the reconstruction
  /// @throws pcms_error if `geometry` is null
  [[nodiscard]] static std::shared_ptr<PolynomialReconstructionFunctionSpace>
  FromGeometry(std::shared_ptr<const PointCloudGeometry> geometry,
               MLSOptions options = {});

  [[nodiscard]] std::shared_ptr<const FieldLayout> GetLayout()
    const noexcept override;

protected:
  [[nodiscard]] FieldVariant CreateFieldImpl(Type storage_type,
                                             ValueBasis basis) const override;

  [[nodiscard]] FieldVariant CreateFieldImpl(
    FieldDataVariant data) const override;

private:
  std::shared_ptr<const FieldLayout> layout_;
};

} // namespace pcms

#endif // PCMS_POLYNOMIAL_RECONSTRUCTION_FUNCTION_SPACE_H
