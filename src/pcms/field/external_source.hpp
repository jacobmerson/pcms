#ifndef PCMS_FIELD_EXTERNAL_SOURCE_HPP
#define PCMS_FIELD_EXTERNAL_SOURCE_HPP

#include "pcms/field/coordinate_system.hpp"
#include "pcms/field/data/handle.hpp"
#include "pcms/field/field.h"
#include "pcms/field/field_factory.h"
#include "pcms/field/layout/empty.h"
#include "pcms/field/point_evaluator_factory.hpp"
#include "pcms/field/value_view.hpp"
#include "pcms/utility/types.h"
#include <memory>
#include <string>
#include <utility>

namespace pcms
{

/// Base for sources evaluated by an external library that holds the field
/// values itself (e.g. NIMROD's nimeval, Fusion-IO). Such a source is a
/// PointEvaluatorFactory, so it plugs into Interpolator, the Monte Carlo
/// projections, and TransformedSource directly. It is also a FieldFactory, but
/// only to mint the Fields that name its quantities: each Field carries
/// HandleFieldData on an EmptyFieldLayout, because PCMS holds none of its
/// values. Derived sources expose a quantity-specific way to create those
/// Fields (via MakeField); the generic CreateField overloads throw because a
/// value type and basis alone cannot name a quantity.
///
/// Derived sources implement CreatePointEvaluatorImpl, typically with a
/// HostPointEvaluator. Concrete sources are shared-owned so that a
/// TransformedSource, or Field handles that keep library state alive, can
/// co-own them.
class ExternalSource
  : public PointEvaluatorFactory
  , public FieldFactory
{
public:
  [[nodiscard]] const std::shared_ptr<const CoordinateSystem>&
  GetCoordinateSystem() const override
  {
    return system_;
  }

  /// Layout of the source as a whole: no DOF holders, one component. Fields
  /// minted by MakeField carry a layout with their own component count.
  [[nodiscard]] std::shared_ptr<const FieldLayout> GetLayout()
    const noexcept override
  {
    return layout_;
  }

protected:
  /// @param system coordinate system query points must be expressed in
  /// @param name label for error messages (e.g. "NimrodSource")
  ExternalSource(std::shared_ptr<const CoordinateSystem> system,
                 std::string name);

  /// Creates the Field naming one quantity of this source.
  /// @param name field name
  /// @param handle names the quantity inside the external library
  /// @param basis value basis of the quantity's components
  /// @param num_components components per value
  template <typename T, typename Handle>
  [[nodiscard]] Field<T> MakeField(std::string name, Handle handle,
                                   ValueBasis basis, int num_components) const
  {
    detail::ValidateValueSemantics(basis, num_components);
    return WrapField<T>(
      std::move(name),
      std::make_shared<EmptyFieldLayout>(system_, num_components),
      std::make_unique<HandleFieldData<T, Handle>>(
        std::move(handle), std::move(basis), num_components));
  }

  [[nodiscard]] const std::string& GetName() const noexcept { return name_; }

  FieldVariant CreateFieldImpl(Type storage_type,
                               ValueBasis basis) const override;
  FieldVariant CreateFieldImpl(FieldDataVariant data) const override;

private:
  std::shared_ptr<const CoordinateSystem> system_;
  std::shared_ptr<const EmptyFieldLayout> layout_;
  std::string name_;
};

} // namespace pcms

#endif // PCMS_FIELD_EXTERNAL_SOURCE_HPP
