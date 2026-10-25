#include "pcms/field/function_space/nimrod.h"
#include "pcms/field/function_space/nimrod_ffi.h"
#include "pcms/utility/arrays.h"

#include <Kokkos_Core.hpp>

#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

namespace pcms
{

int NimrodInputTraits::NqtyForField(char field_id) const
{
  // Mirror nimpy EvalNimrod.__get_nqty (dmode == 0).
  switch (field_id) {
    case 'b': // magnetic field
    case 'v': // velocity
    case 'j': // current density
    case 'f': // distortion / force
    case 'e': // electric field
      return 3;
    case 'n': // number density
      return nisp + 1;
    case 'd': // diffusivity shape
      return ds_nqty;
    case 'r':
      return 4;
    default: // scalar fields: pressure 'p', temperatures 't', etc.
      return 1;
  }
}

namespace
{

// Process-global bookkeeping for NIMROD's single-active-dump C interface.
// c_nimfield_init installs one dump per process; the most recent FromDump wins.
std::mutex& NimrodInitMutex()
{
  static std::mutex m;
  return m;
}
std::atomic<std::uint64_t>& NimrodActivationCounter()
{
  static std::atomic<std::uint64_t> counter{0};
  return counter;
}
std::atomic<std::uint64_t>& NimrodActiveActivation()
{
  static std::atomic<std::uint64_t> active{0};
  return active;
}

using DeviceLayout = detail::default_layout_for_memory_space_t<DeviceMemorySpace>;
using KokkosDeviceLayout =
  std::conditional_t<std::is_same_v<DeviceLayout, Kokkos::layout_left>,
                     Kokkos::LayoutLeft, Kokkos::LayoutRight>;

// Point evaluator bound to a fixed set of query coordinates. Localization is
// delegated to NIMROD (the xyguess seed marches between successive points).
class NimrodPointEvaluator : public PointEvaluator<double>
{
public:
  NimrodPointEvaluator(CoordinateView<DeviceMemorySpace> coords,
                       std::uint64_t activation_id)
    : activation_id_(activation_id)
  {
    auto values = coords.GetValues();
    const std::size_t npts = values.extent(0);
    const std::size_t ndim = values.extent(1);
    npts_ = npts;

    // Copy query coordinates to host: NIMROD evaluation is host/CPU Fortran.
    Kokkos::View<const Real**, KokkosDeviceLayout, DeviceMemorySpace,
                 Kokkos::MemoryUnmanaged>
      device_coords(values.data_handle(), npts, ndim);
    auto host_coords =
      Kokkos::create_mirror_view_and_copy(HostMemorySpace(), device_coords);

    coords_.assign(npts * 3, 0.0);
    for (std::size_t i = 0; i < npts; ++i) {
      for (std::size_t d = 0; d < ndim && d < 3; ++d) {
        coords_[i * 3 + d] = host_coords(i, d);
      }
    }
  }

  void Evaluate(const Field<double>& field,
                Rank2View<double, DeviceMemorySpace, DeviceLayout> values)
    const override
  {
    if (NimrodActiveActivation().load() != activation_id_) {
      throw pcms_error(
        "NimrodPointEvaluator: the NIMROD dump this evaluator was built from is "
        "no longer active (another dump was opened)");
    }

    const auto* data =
      dynamic_cast<const NimrodFieldData<double>*>(&field.GetData());
    if (data == nullptr) {
      throw pcms_error(
        "NimrodPointEvaluator::Evaluate: field was not created by "
        "NimrodFunctionSpace::MakeField");
    }
    const NimrodFieldDescriptor& desc = data->descriptor();

    const std::size_t npts = values.extent(0);
    const std::size_t nqty = values.extent(1);
    if (npts != npts_) {
      throw pcms_error(
        "NimrodPointEvaluator::Evaluate: output point count does not match the "
        "query coordinates");
    }
    if (static_cast<int>(nqty) != desc.nqty) {
      throw pcms_error(
        "NimrodPointEvaluator::Evaluate: output component count does not match "
        "the NIMROD field");
    }

    Kokkos::View<double**, KokkosDeviceLayout, HostMemorySpace> host_out(
      "nimrod_eval_host_out", npts, nqty);

    int c_nqty = desc.nqty;
    int c_dmode = desc.dmode;
    int c_eq = desc.eq;
    std::vector<double> fval(static_cast<std::size_t>(desc.nqty) * 10, 0.0);
    double xyguess[2] = {-1.0, -1.0};
    const std::string& fname = desc.field_id;

    for (std::size_t i = 0; i < npts; ++i) {
      double rzp[3] = {coords_[i * 3 + 0], coords_[i * 3 + 1],
                       coords_[i * 3 + 2]};
      int ifail = 0;
      c_eval_field_dm(rzp, fname.c_str(), fval.data(), &c_nqty, xyguess, &ifail,
                      &c_dmode, &c_eq);
      if (ifail != 0) {
        // Out of bounds: mark NaN. Do NOT reset xyguess to a cold (-1,-1);
        // keep the (NIMROD-updated) seed so the next point still benefits from
        // a warm guess. A warm seed lets NIMROD's rz_to_xy converge at boundary
        // points that a cold global find_guess misses, so PCMS covers the same
        // near-boundary domain as nimpy (which also marches a persistent guess
        // and never resets it on failure).
        for (std::size_t q = 0; q < nqty; ++q) {
          host_out(i, q) = std::numeric_limits<double>::quiet_NaN();
        }
      } else {
        for (std::size_t q = 0; q < nqty; ++q) {
          host_out(i, q) = fval[q];
        }
      }
    }

    Kokkos::View<double**, KokkosDeviceLayout, DeviceMemorySpace,
                 Kokkos::MemoryUnmanaged>
      device_out(values.data_handle(), npts, nqty);
    Kokkos::deep_copy(device_out, host_out);
  }

private:
  std::vector<double> coords_; // flattened [npts][3] (R,Z,Phi) on host
  std::size_t npts_ = 0;
  std::uint64_t activation_id_;
};

} // namespace

NimrodFunctionSpace::NimrodFunctionSpace(NimrodInputTraits traits, int eq,
                                         std::uint64_t activation_id)
  : layout_(std::make_shared<EmptyFieldLayout>()),
    traits_(traits),
    eq_(eq),
    activation_id_(activation_id)
{
}

NimrodFunctionSpace NimrodFunctionSpace::FromDump(const std::string& dumpfile,
                                                  NimrodInputTraits traits,
                                                  const std::string& fieldlist,
                                                  int eq)
{
  std::lock_guard<std::mutex> lock(NimrodInitMutex());
  c_nimfield_init(dumpfile.c_str(), fieldlist.c_str());
  const std::uint64_t id = ++NimrodActivationCounter();
  NimrodActiveActivation().store(id);
  return NimrodFunctionSpace(traits, eq, id);
}

Field<double> NimrodFunctionSpace::MakeField(const std::string& field_id) const
{
  return MakeField(field_id, eq_);
}

Field<double> NimrodFunctionSpace::MakeField(const std::string& field_id,
                                             int eq) const
{
  if (field_id.empty()) {
    throw pcms_error("NimrodFunctionSpace::MakeField: empty field id");
  }
  NimrodFieldDescriptor descriptor;
  descriptor.field_id = field_id;
  descriptor.nqty = traits_.NqtyForField(field_id[0]);
  descriptor.eq = eq;
  descriptor.dmode = 0;

  FieldMetadata metadata;
  metadata.value_type =
    descriptor.nqty > 1 ? FieldValueType::Vector : FieldValueType::Scalar;
  metadata.value_coordinate_system = CoordinateSystem::Cylindrical;

  return WrapField<double>(
    layout_, std::make_unique<NimrodFieldData<double>>(std::move(descriptor),
                                                       metadata));
}

FieldVariant NimrodFunctionSpace::CreateFieldImpl(Type /*value_type*/,
                                                  FieldMetadata /*metadata*/)
  const
{
  throw pcms_error(
    "NimrodFunctionSpace: use MakeField(field_id) to name a NIMROD field");
}

FieldVariant NimrodFunctionSpace::CreateFieldImpl(FieldDataVariant data) const
{
  return std::visit(
    [this](auto&& fd) -> FieldVariant {
      using FD = std::decay_t<decltype(fd)>;
      using T = typename FD::element_type::value_type;
      PCMS_ALWAYS_ASSERT(fd != nullptr);
      if (dynamic_cast<const NimrodFieldData<T>*>(fd.get()) == nullptr) {
        throw pcms_error(
          "NimrodFunctionSpace::CreateField: requires NimrodFieldData");
      }
      return WrapField<T>(layout_, std::move(fd));
    },
    std::move(data));
}

PointEvaluatorVariant NimrodFunctionSpace::CreatePointEvaluatorImpl(
  Type value_type, const EvaluationRequest& request) const
{
  if (value_type != Type::Real) {
    throw pcms_error(
      "NimrodFunctionSpace only supports double-valued point evaluation");
  }
  return PointEvaluatorVariant{std::unique_ptr<PointEvaluator<double>>(
    std::make_unique<NimrodPointEvaluator>(request.coords, activation_id_))};
}

} // namespace pcms
