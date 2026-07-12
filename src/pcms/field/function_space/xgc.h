#ifndef PCMS_XGC_FUNCTION_SPACE_H
#define PCMS_XGC_FUNCTION_SPACE_H

#include "pcms/field/data/xgc.h"
#include "pcms/field/field.h"
#include "pcms/field/function_space.h"
#include "pcms/field/field_metadata.h"
#include "pcms/utility/common.h"
#include <functional>
#include <memory>

namespace pcms
{

class XGCFunctionSpace : public FunctionSpace
{
public:
  XGCFunctionSpace(const ReverseClassificationVertex& reverse_classification,
                   std::function<int8_t(int, int)> in_overlap,
                   LO num_plane_nodes, std::string layout_name = "")
    : layout_(std::make_shared<XGCFieldLayout>(
        reverse_classification, std::move(in_overlap), num_plane_nodes))
  {
    layout_->SetName(std::move(layout_name));
  }

  // Set the layout name after construction. Const because the name is metadata
  // that does not affect the (const) discretization; the layout pointee is
  // mutable through the shared_ptr. Used by the capi, which learns a field's
  // name only when it is added, after the space is built.
  void SetLayoutName(std::string name) const
  {
    layout_->SetName(std::move(name));
  }

  [[nodiscard]] std::shared_ptr<const FieldLayout> GetLayout()
    const noexcept override
  {
    return layout_;
  }

  [[nodiscard]] CoordinateSystem GetCoordinateSystem() const noexcept override
  {
    return CoordinateSystem::XGC;
  }

protected:
  [[nodiscard]] FieldVariant CreateFieldImpl(
    Type value_type, FieldMetadata metadata) const override
  {
    return apply_to_type(value_type, [&](auto tag) -> FieldVariant {
      using T = typename decltype(tag)::type;
      return WrapField<T>(layout_,
                          std::make_unique<XGCFieldData<T>>(layout_, metadata));
    });
  }

  [[nodiscard]] FieldVariant CreateFieldImpl(
    FieldDataVariant data) const override
  {
    return std::visit(
      [this](auto&& fd) -> FieldVariant {
        using FD = std::decay_t<decltype(fd)>;
        using T = typename FD::element_type::value_type;
        PCMS_ALWAYS_ASSERT(fd != nullptr);
        if (dynamic_cast<const XGCFieldData<T>*>(fd.get()) == nullptr) {
          throw pcms_error(
            "XGCFunctionSpace::CreateField: requires XGCFieldData");
        }
        if (fd->GetDOFHolderDataHost().size() !=
            static_cast<size_t>(layout_->GetFullDataSize())) {
          throw pcms_error(
            "XGCFunctionSpace::CreateField: field data size does not match "
            "layout");
        }
        return WrapField<T>(layout_, std::move(fd));
      },
      std::move(data));
  }

  [[nodiscard]] PointEvaluatorVariant CreatePointEvaluatorImpl(
    Type /*value_type*/, const EvaluationRequest& /*request*/) const override
  {
    throw pcms_error("XGCFunctionSpace does not support point evaluation");
  }

public:
  [[nodiscard]] std::shared_ptr<const XGCFieldLayout> GetXGCLayout()
    const noexcept
  {
    return layout_;
  }

private:
  std::shared_ptr<XGCFieldLayout> layout_;
};

} // namespace pcms

#endif // PCMS_XGC_FUNCTION_SPACE_H
