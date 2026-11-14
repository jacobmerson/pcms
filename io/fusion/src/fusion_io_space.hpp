#ifndef FUSION_IO_SRC_FUSION_IO_SPACE_HPP
#define FUSION_IO_SRC_FUSION_IO_SPACE_HPP

#include "fusion_io/fusion_io.hpp"
#include "pcms/field/function_space.h"
#include "pcms/field/layout/empty.h"
#include <memory>
#include <string>

namespace fio::detail
{

/// An open Fusion-IO file. Opaque: Fusion-IO is a private dependency.
class FusionIOSource;

/// Opens a Fusion-IO file, e.g. an M3D-C1 C1.h5 or a geqdsk file. Only
/// sources with cylindrical geometry are supported. The spaces and fields
/// created from it share and keep it open.
std::shared_ptr<FusionIOSource> OpenFusionIOSource(Format format,
                                                   const std::string& filename);

/// Fields of a Fusion-IO source, evaluated by Fusion-IO at the query points.
/// Points are (R, phi, Z), which is CylindricalRThetaZ, and vector values are
/// physical cylindrical components. PCMS holds none of the values: functions
/// can be evaluated and used as transfer sources, but not read or written.
///
/// Fusion-IO's (R, phi, Z) is right-handed, like CylindricalRThetaZ, so phi is
/// theta.
///
/// A space holds quantities with one component count (1 for scalars, 3 for
/// vectors); create one space per count over the same source.
///
/// Evaluators of derivative order 1 give the gradient of scalar quantities as
/// physical cylindrical components (d/dR, (1/R) d/dphi, d/dZ). Fusion-IO
/// provides them for M3D-C1 and geqdsk sources; for other formats Evaluate
/// throws.
class FusionIOFunctionSpace final : public pcms::FunctionSpace
{
public:
  static std::shared_ptr<FusionIOFunctionSpace> Create(
    std::shared_ptr<FusionIOSource> source, int num_components);

  std::shared_ptr<const pcms::FieldLayout> GetLayout() const noexcept override
  {
    return layout_;
  }

  /// Function for a quantity of the source, whose component count must be
  /// the space's.
  pcms::Function<Real> CreateFunction(std::string name, Quantity quantity,
                                      const Options& options = {}) const;

protected:
  pcms::FieldVariant CreateFieldImpl(pcms::Type storage_type,
                                     pcms::ValueBasis basis) const override;
  pcms::FieldVariant CreateFieldImpl(
    pcms::FieldDataVariant data) const override;

private:
  FusionIOFunctionSpace(std::shared_ptr<FusionIOSource> source,
                        int num_components);

  std::shared_ptr<FusionIOSource> source_;
  std::shared_ptr<const pcms::EmptyFieldLayout> layout_;
};

} // namespace fio::detail

#endif // FUSION_IO_SRC_FUSION_IO_SPACE_HPP
