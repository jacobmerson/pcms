#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <redev.h>

#include "pcms/coupler/coupler.hpp"
#include "pcms/field/function_space.hpp"
#include "pcms/pythonapi/numpy_view.hpp"
#include "pcms/pythonapi/py_value_types.hpp"

namespace py = pybind11;

namespace pcms
{

namespace
{
/// @brief Convert a Python communicator to an MPI communicator.
/// @param comm an mpi4py communicator (anything with py2f()) or None
/// @return the communicator, MPI_COMM_WORLD for None
/// @throws py::type_error if comm has no py2f()
MPI_Comm ToMPIComm(const py::object& comm)
{
  if (comm.is_none()) {
    return MPI_COMM_WORLD;
  }
  if (!py::hasattr(comm, "py2f")) {
    throw py::type_error(
      "comm must be an mpi4py communicator (with py2f()) or None");
  }
  return MPI_Comm_f2c(comm.attr("py2f")().cast<MPI_Fint>());
}
} // namespace

/// @brief Bind the coupler as the submodule pcms.coupler.
/// @param m the pcms module
void bind_coupler_module(py::module& m)
{
  auto coupler = m.def_submodule(
    "coupler", "Field exchange between applications over ADIOS2 (redev)");

  py::enum_<redev::TransportType>(coupler, "TransportType")
    .value("BP4", redev::TransportType::BP4,
           "ADIOS2 BP4 files: works anywhere, data goes through disk")
    .value("SST", redev::TransportType::SST,
           "ADIOS2 SST: live stream between running processes");

  py::class_<redev::RCBPtn>(coupler, "RCBPartition")
    .def(py::init(
           [](redev::LO dim, std::vector<int> ranks, std::vector<double> cuts) {
             return redev::RCBPtn(dim, ranks, cuts);
           }),
         py::arg("dim"), py::arg("ranks"), py::arg("cuts"),
         "Recursive coordinate bisection partition of the server ranks: "
         "ranks label the leaves left to right, cuts holds the cut "
         "coordinates breadth first from index 1. One rank: "
         "RCBPartition(dim, [0], [0.0]).")
    .def_property_readonly("ranks", &redev::RCBPtn::GetRanks)
    .def_property_readonly("cuts", &redev::RCBPtn::GetCuts);

  py::class_<FieldHandle<Real>>(coupler, "FieldHandle")
    .def(
      "send", [](const FieldHandle<Real>& self) { self.Send(); },
      "Send the field; must be called inside a send phase")
    .def(
      "receive", [](const FieldHandle<Real>& self) { self.Receive(); },
      "Receive into the field; must be called inside a receive phase")
    .def_property_readonly("field", &FieldHandle<Real>::GetField,
                           py::return_value_policy::reference_internal,
                           "The registered field (owned by the application)");

  py::class_<FunctionHandle<Real>, FieldHandle<Real>>(coupler, "FunctionHandle")
    .def(
      "get_space",
      [](const FunctionHandle<Real>& self) {
        return Immutable<FunctionSpace>{self.GetSpacePtr()};
      },
      "The space of the registered function");

  py::class_<DataHandle<Real>>(coupler, "DataHandle")
    .def(
      "send", [](const DataHandle<Real>& self) { self.Send(); },
      "Send the registered array; must be called inside a send phase")
    .def(
      "receive", [](const DataHandle<Real>& self) { self.Receive(); },
      "Receive into the registered array; must be called inside a receive "
      "phase")
    .def_property_readonly("name", &DataHandle<Real>::GetName);

  // Owned by its CouplerComm; Python never deletes it.
  py::class_<ApplicationComm, std::unique_ptr<ApplicationComm, py::nodelete>>(
    coupler, "ApplicationComm")
    .def(
      "add_function",
      [](ApplicationComm& self, std::string name,
         const Immutable<FunctionSpace>& space, bool participates) {
        return self.AddFunction(
          space.ptr->CreateFunction<Real>(std::move(name)), participates);
      },
      py::arg("name"), py::arg("space"), py::arg("participates") = true,
      py::keep_alive<0, 1>(),
      "Create a scalar function named `name` on `space` and register it for "
      "exchange; both sides use the same name and layout name. Returns its "
      "FunctionHandle.")
    .def(
      "add_data",
      [](ApplicationComm& self, std::string name, py::array& data,
         const py::object& comm) {
        auto view = NumpyOutputToHostView<1, Real>(data);
        return self.AddData<Real>(std::move(name), view, ToMPIComm(comm));
      },
      py::arg("name"), py::arg("data"), py::arg("comm") = py::none(),
      py::keep_alive<0, 1>(), py::keep_alive<1, 3>(),
      "Register a C-contiguous float64 array exchanged whole; receive() "
      "overwrites it in place. comm is the CouplerComm's mpi4py communicator "
      "(None: MPI_COMM_WORLD). Returns its DataHandle.")
    .def("begin_send_phase", &ApplicationComm::BeginSendPhase)
    .def("end_send_phase", &ApplicationComm::EndSendPhase)
    .def("begin_receive_phase", &ApplicationComm::BeginReceivePhase)
    .def("end_receive_phase", &ApplicationComm::EndReceivePhase)
    .def("in_send_phase", &ApplicationComm::InSendPhase)
    .def("in_receive_phase", &ApplicationComm::InReceivePhase);

  py::class_<CouplerComm>(coupler, "CouplerComm")
    .def(py::init([](std::string name, const py::object& comm, bool is_server,
                     std::optional<redev::RCBPtn> partition) {
           if (is_server && !partition) {
             throw py::value_error("a server needs a partition");
           }
           redev::Partition ptn =
             partition ? redev::Partition{*partition} : redev::Partition{};
           return std::make_unique<CouplerComm>(
             std::move(name), ToMPIComm(comm), is_server, std::move(ptn));
         }),
         py::arg("name"), py::arg("comm") = py::none(),
         py::arg("is_server") = false, py::arg("partition") = py::none(),
         py::keep_alive<1, 3>(),
         "Connect to the other side; the server passes the partition of its "
         "ranks. comm is an mpi4py communicator (None: MPI_COMM_WORLD).")
    .def(
      "add_application",
      [](CouplerComm& self, std::string name, std::string path,
         redev::TransportType transport,
         std::optional<std::map<std::string, std::string>> params) {
        if (params) {
          return self.AddApplication(std::move(name), std::move(path),
                                     transport, adios2::Params(*params));
        }
        return self.AddApplication(std::move(name), std::move(path), transport);
      },
      py::arg("name"), py::arg("path") = "",
      py::arg("transport") = redev::TransportType::BP4,
      py::arg("params") = py::none(), py::return_value_policy::reference,
      py::keep_alive<0, 1>(),
      "Open the ADIOS2 channel `name` in directory `path` with ADIOS2 engine "
      "parameters `params`. Returns the ApplicationComm.");
}

} // namespace pcms
