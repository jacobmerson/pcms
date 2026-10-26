#ifndef PCMS_FIELD_FUNCTION_SPACE_NIMROD_HPP
#define PCMS_FIELD_FUNCTION_SPACE_NIMROD_HPP

#include "pcms/field/function_space.h"
#include "pcms/field/layout/empty.h"
#include <memory>
#include <string>
#include <vector>

namespace pcms
{

/// A NIMROD dump opened for evaluation. Opaque: NIMROD is a private
/// dependency.
class NimrodSource;

struct NimrodOptions
{
  enum class Part
  {
    Total,
    Perturbed,
    Equilibrium
  };

  Part part = Part::Total;
};

/// Opens a NIMROD dump for evaluation of the fields named by the
/// single-character ids in `fields`, e.g. "bvn". NIMROD's evaluator holds one
/// dump per process, so this throws while another source is open; the spaces
/// and fields created from a source share and keep it open. Only toroidal
/// cases (geom='tor') are supported. Not thread safe: it changes the current
/// directory while NIMROD opens the dump.
///
/// @param case_dir directory holding the case's nimrod.in (default: the
/// current directory). It must be writable: NIMROD writes scratch files there
/// while opening the dump. `dump` is relative to the current directory.
std::shared_ptr<NimrodSource> OpenNimrodSource(
  const std::string& dump, const std::string& fields,
  const std::string& case_dir = "");

/// NIMROD's names for the quantities `source` can evaluate, e.g. "b" or "ti".
std::vector<std::string> AvailableQuantities(const NimrodSource& source);

/// Fields of a NIMROD dump, evaluated by NIMROD at the query points. Points
/// are (R, theta, Z), which is CylindricalRThetaZ, and vector values are
/// physical cylindrical components. PCMS holds none of the values: functions
/// can be evaluated and used as transfer sources, but not read or written.
///
/// NIMROD orders coordinates (R, Z, phi), right-handed, so its phi is -theta
/// and vectors are mapped accordingly.
///
/// A space holds quantities with one component count (e.g. 1 for pressure, 3
/// for vectors); create one space per count over the same source.
class NimrodFunctionSpace final : public FunctionSpace
{
public:
  static std::shared_ptr<NimrodFunctionSpace> Create(
    std::shared_ptr<NimrodSource> source, int num_components);

  std::shared_ptr<const FieldLayout> GetLayout() const noexcept override
  {
    return layout_;
  }

  /// Function for a NIMROD quantity, whose component count must be the
  /// space's.
  /// @param quantity NIMROD's name for it (see AvailableQuantities)
  Function<Real> CreateFunction(std::string name, const std::string& quantity,
                                const NimrodOptions& options = {}) const;

protected:
  FieldVariant CreateFieldImpl(Type storage_type,
                               ValueBasis basis) const override;
  FieldVariant CreateFieldImpl(FieldDataVariant data) const override;
  std::shared_ptr<BoundSource> BindImpl(
    const EvaluationRequest& request) const override;
  PointEvaluatorVariant CreatePointEvaluatorImpl(
    Type value_type, std::shared_ptr<const BoundSource> bound,
    int derivative_order) const override;

private:
  NimrodFunctionSpace(std::shared_ptr<NimrodSource> source, int num_components);

  std::shared_ptr<NimrodSource> source_;
  std::shared_ptr<const EmptyFieldLayout> layout_;
};

} // namespace pcms

#endif // PCMS_FIELD_FUNCTION_SPACE_NIMROD_HPP
