#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <mpi.h>
#include <redev.h>
#include "pcms/coupler/coupler.hpp"
#include "pcms/field/function_space.h"
#include "numpy_view.h"

namespace py = pybind11;

namespace pcms
{

namespace
{
// Accepts an mpi4py communicator (anything with py2f()) or None for
// MPI_COMM_WORLD. Going through the Fortran handle avoids a build-time
// dependency on the mpi4py headers.
MPI_Comm ToMPIComm(const py::object& comm)
{
  if (comm.is_none()) {
    return MPI_COMM_WORLD;
  }
  if (!py::hasattr(comm, "py2f")) {
    throw std::invalid_argument(
      "comm must be an mpi4py communicator (with py2f()) or None");
  }
  return MPI_Comm_f2c(comm.attr("py2f")().cast<MPI_Fint>());
}
} // namespace

void bind_coupler_module(py::module& m)
{
  py::enum_<redev::TransportType>(m, "TransportType")
    .value("BP4", redev::TransportType::BP4,
           "ADIOS2 BP4 files: works anywhere, data goes through disk")
    .value("SST", redev::TransportType::SST,
           "ADIOS2 SST: live stream between running processes")
    .export_values();

  // Recursive coordinate bisection partition of the server ranks. A single
  // server rank owning everything is RCBPartition(dim, [0], [0.0]).
  py::class_<redev::RCBPtn>(m, "RCBPartition")
    .def(py::init(
           [](redev::LO dim, std::vector<int> ranks, std::vector<double> cuts) {
             return redev::RCBPtn(dim, ranks, cuts);
           }),
         py::arg("dim"), py::arg("ranks"), py::arg("cuts"),
         "Partition from the cut tree: ranks label the leaves left to right; "
         "cuts holds the non-leaf cut coordinates in breadth-first order "
         "starting at index 1 (index 0 is unused)")
    .def_property_readonly("ranks", &redev::RCBPtn::GetRanks)
    .def_property_readonly("cuts", &redev::RCBPtn::GetCuts);

  py::class_<FieldHandle<Real>>(m, "FieldHandle")
    .def(
      "send", [](const FieldHandle<Real>& self) { self.Send(); },
      "Send the field; must be called inside a send phase")
    .def(
      "receive", [](const FieldHandle<Real>& self) { self.Receive(); },
      "Receive into the field; must be called inside a receive phase")
    .def_property_readonly("field", &FieldHandle<Real>::GetField,
                           py::return_value_policy::reference_internal,
                           "The registered field (owned by the application)");

  py::class_<DataHandle<Real>>(m, "DataHandle")
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
    m, "ApplicationComm")
    .def(
      "add_field",
      [](ApplicationComm& self, std::string name, const FunctionSpace& space) {
        return self.AddField(space.CreateFunction<Real>(std::move(name)));
      },
      py::arg("name"), py::arg("space"), py::keep_alive<0, 1>(),
      "Create a field named `name` on `space` and register it for exchange. "
      "Both sides must use the same field name and the same layout name "
      "(from_mesh(layout_name=...)); fields with the same layout name share "
      "one exchange plan. The field lives in the application; access it via "
      "the returned handle's `field`.")
    .def(
      "add_data",
      [](ApplicationComm& self, std::string name, py::array& data) {
        auto view = NumpyOutputToHostView<1, Real>(data);
        return self.AddData<Real>(std::move(name), view, self.GetMPIComm());
      },
      py::arg("name"), py::arg("data"), py::keep_alive<0, 1>(),
      py::keep_alive<1, 3>(),
      "Register a small float64 array sent as is (not partitioned): send() "
      "transmits its current contents, receive() overwrites it in place. "
      "The array must be C-contiguous and stays registered for the life of "
      "the application.")
    .def("begin_send_phase", &ApplicationComm::BeginSendPhase)
    .def("end_send_phase", &ApplicationComm::EndSendPhase)
    .def("begin_receive_phase", &ApplicationComm::BeginReceivePhase)
    .def("end_receive_phase", &ApplicationComm::EndReceivePhase)
    .def("in_send_phase", &ApplicationComm::InSendPhase)
    .def("in_receive_phase", &ApplicationComm::InReceivePhase);

  py::class_<CouplerComm>(m, "CouplerComm")
    .def(py::init([](std::string name, const py::object& comm, bool is_server,
                     std::optional<redev::RCBPtn> partition) {
           if (is_server && !partition) {
             throw std::invalid_argument("a server needs a partition");
           }
           redev::Partition ptn =
             partition ? redev::Partition{*partition} : redev::Partition{};
           return std::make_unique<CouplerComm>(
             std::move(name), ToMPIComm(comm), is_server, std::move(ptn));
         }),
         py::arg("name"), py::arg("comm") = py::none(),
         py::arg("is_server") = false, py::arg("partition") = py::none(),
         py::keep_alive<1, 3>(),
         "Rendezvous with the other side. The server passes the partition of "
         "its ranks; clients receive it during add_application. comm is an "
         "mpi4py communicator (default MPI_COMM_WORLD).")
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
      "Open the ADIOS2 channel named `name` (in directory `path`). params are "
      "ADIOS2 engine parameters, e.g. {'QueueLimit': '4'} for SST.");
}

} // namespace pcms
