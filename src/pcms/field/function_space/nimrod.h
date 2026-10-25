#ifndef PCMS_NIMROD_FUNCTION_SPACE_H
#define PCMS_NIMROD_FUNCTION_SPACE_H

// Evaluation-only PCMS backend for NIMROD fields.
//
// NimrodFunctionSpace reads a NIMROD dump file (via NIMROD's own `nimeval`
// shared library, declared in nimrod_ffi.h) and exposes point evaluation of
// NIMROD fields through the standard PCMS FunctionSpace -> PointEvaluator API.
// Because we call the same library nimpy wraps, results agree with nimpy.
//
// Scope (see plan): real-space, per-plane evaluation of a named NIMROD field at
// arbitrary (R,Z,Phi) points. Send/receive coupling and complex/per-mode
// evaluation are future milestones. Only the `double` value type is supported.
//
// Global-state caveat: NIMROD's c_nimfield_init keeps a single active dump per
// process; constructing a second NimrodFunctionSpace re-initializes over the
// first. Evaluators created from a superseded space throw on use.

#include "pcms/field/field.h"
#include "pcms/field/field_data.h"
#include "pcms/field/field_metadata.h"
#include "pcms/field/function_space.h"
#include "pcms/field/layout/empty.h"
#include "pcms/field/point_evaluator.h"
#include "pcms/utility/assert.h"

#include <cstdint>
#include <memory>
#include <string>

namespace pcms
{

// Subset of nimrod.in traits needed to size NIMROD fields. Mirrors the values
// nimpy reads in EvalNimrod.__init__ / __get_nqty. Supply these from the run's
// nimrod.in (a namelist parser to fill this automatically is a future step).
struct NimrodInputTraits
{
  int nisp = 0;    // number of ion species; density 'n' has nqty = nisp + 1
  int nnsp = 0;    // number of neutral species
  int ds_nqty = 1; // number of diffusivity-shape components (field 'd')
  int zimp = 0;    // impurity charge; field 'nz' has nqty = zimp + 1
  int nmodes = 1;  // number of Fourier modes (used by complex evaluation)

  // Number of components a single-character NIMROD field id carries at dmode=0.
  // Mirrors nimpy EvalNimrod.__get_nqty.
  [[nodiscard]] int NqtyForField(char field_id) const;
};

// Identifies which NIMROD field a PCMS Field refers to, plus the evaluation
// options. NIMROD holds the coefficient state internally, so this FieldData
// carries a descriptor rather than DOF coefficients (permitted by the
// FieldData contract, which allows non-spatial parameter data).
struct NimrodFieldDescriptor
{
  std::string field_id; // NUL-terminated NIMROD field id, e.g. "b", "n", "p".
  int nqty = 1;         // number of components produced at dmode = 0.
  int eq = 2;           // equilibrium/perturbation selector (see nimrod_ffi.h).
  int dmode = 0;        // derivative mode (0 only, for now).
};

template <typename T>
class NimrodFieldData : public FieldData<T>
{
public:
  NimrodFieldData(NimrodFieldDescriptor descriptor, FieldMetadata metadata)
    : descriptor_(std::move(descriptor)), metadata_(metadata)
  {
  }

  [[nodiscard]] const NimrodFieldDescriptor& descriptor() const noexcept
  {
    return descriptor_;
  }

  const FieldMetadata& GetMetadata() const override { return metadata_; }

  // NIMROD owns the coefficient state; this backend exposes no DOF coefficients.
  Rank1View<const T, HostMemorySpace> GetDOFHolderDataHost() const override
  {
    return Rank1View<const T, HostMemorySpace>(static_cast<const T*>(nullptr),
                                               0);
  }
  void SetDOFHolderDataHost(Rank1View<const T, HostMemorySpace>) override
  {
    throw pcms_error(
      "NimrodFieldData is evaluation-only; coefficient data cannot be set");
  }
  Rank1View<const T, DeviceMemorySpace> GetDOFHolderData() const override
  {
    return Rank1View<const T, DeviceMemorySpace>(static_cast<const T*>(nullptr),
                                                 0);
  }
  void SetDOFHolderData(Rank1View<const T, DeviceMemorySpace>) override
  {
    throw pcms_error(
      "NimrodFieldData is evaluation-only; coefficient data cannot be set");
  }

private:
  NimrodFieldDescriptor descriptor_;
  FieldMetadata metadata_;
};

class NimrodFunctionSpace : public FunctionSpace
{
public:
  // Open a NIMROD dump for evaluation. Calls c_nimfield_init once (requires
  // nimrod.in in the working directory) and becomes the process-active dump.
  //   dumpfile:  path to the NIMROD dump file.
  //   traits:    field-sizing traits from nimrod.in.
  //   fieldlist: NIMROD field ids to initialize (default "bvnpf").
  //   eq:        default equilibrium/perturbation selector for MakeField.
  [[nodiscard]] static NimrodFunctionSpace FromDump(
    const std::string& dumpfile, NimrodInputTraits traits,
    const std::string& fieldlist = "bvnpf", int eq = 2);

  // Build a Field naming a NIMROD field to evaluate (e.g. "b", "n", "p").
  // nqty is resolved from the traits; value type is always double.
  [[nodiscard]] Field<double> MakeField(const std::string& field_id) const;
  [[nodiscard]] Field<double> MakeField(const std::string& field_id,
                                        int eq) const;

  [[nodiscard]] const NimrodInputTraits& GetTraits() const noexcept
  {
    return traits_;
  }
  [[nodiscard]] std::uint64_t GetActivationId() const noexcept
  {
    return activation_id_;
  }

  [[nodiscard]] std::shared_ptr<const FieldLayout> GetLayout()
    const noexcept override
  {
    return layout_;
  }

  [[nodiscard]] CoordinateSystem GetCoordinateSystem() const noexcept override
  {
    return CoordinateSystem::Cylindrical;
  }

protected:
  [[nodiscard]] FieldVariant CreateFieldImpl(
    Type value_type, FieldMetadata metadata) const override;
  [[nodiscard]] FieldVariant CreateFieldImpl(
    FieldDataVariant data) const override;
  [[nodiscard]] PointEvaluatorVariant CreatePointEvaluatorImpl(
    Type value_type, const EvaluationRequest& request) const override;

private:
  NimrodFunctionSpace(NimrodInputTraits traits, int eq,
                      std::uint64_t activation_id);

  std::shared_ptr<const FieldLayout> layout_;
  NimrodInputTraits traits_;
  int eq_;
  std::uint64_t activation_id_;
};

} // namespace pcms

#endif // PCMS_NIMROD_FUNCTION_SPACE_H
