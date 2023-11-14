#include <pybind11/pybind11.h>
#include <pcms/capi/kokkos.h>
#include "../client.h"
#include "../xgc_field_adapter.h"


namespace py = pybind11;
using pcms::Mode;
using pcms::CoupledField;
using pcms::CouplerClient;

namespace pcms{
class PythonFieldAdapterConcept {
      public: 

      using memory_space = HostMemorySpace;
      using value_type = double;
      using coordinate_element_type = double;
      
      virtual int Serialize(ScalarArrayView<value_type, memory_space> buffer,
    ScalarArrayView<const pcms::LO, memory_space> permutation) const = 0;

      virtual void Deserialize(
    ScalarArrayView<const value_type, memory_space> buffer,
    ScalarArrayView<const pcms::LO, memory_space> permutation) const = 0;

    [[nodiscard]] virtual std::vector<GO> GetGids() const = 0;

    [[nodiscard]] virtual ReversePartitionMap GetReversePartitionMap(
    const redev::Partition& partition) const = 0;

    [[nodiscard]] virtual bool RankParticipatesCouplingCommunication() const noexcept = 0;
};


class PythonFieldAdapter {
      public: 

      PythonFieldAdapter(PythonFieldAdapterConcept& c) : adapter_(c) {};

      PythonFieldAdapterConcept& adapter_;
      using memory_space = HostMemorySpace;
      using value_type = double;
      using coordinate_element_type = double;

      int Serialize(ScalarArrayView<value_type, memory_space> buffer,
    ScalarArrayView<const pcms::LO, memory_space> permutation) const {
      adapter_.Serialize(buffer,permutation);
    }

      void Deserialize(
    ScalarArrayView<const value_type, memory_space> buffer,
    ScalarArrayView<const pcms::LO, memory_space> permutation) const {
      adapter_.Deserialize(buffer,permutation);
    }

    [[nodiscard]] std::vector<GO> GetGids() const {
      adapter_.GetGids();
    }

    [[nodiscard]] ReversePartitionMap GetReversePartitionMap(
    const redev::Partition& partition) const {
      adapter_.GetReversePartitionMap(partition);
    }

    [[nodiscard]] bool RankParticipatesCouplingCommunication() const noexcept {
      adapter_.RankParticipatesCouplingCommunication();
    }
};

//Trampoline class
class PythonFieldAdapterTramp : public PythonFieldAdapterConcept {
  public:
    using PythonFieldAdapterConcept::PythonFieldAdapterConcept;

    int Serialize(ScalarArrayView<value_type, memory_space> buffer,
    ScalarArrayView<const pcms::LO, memory_space> permutation) const override { PYBIND11_OVERRIDE_PURE(int, PythonFieldAdapterConcept, Serialize, buffer, permutation); }

    void Deserialize(ScalarArrayView<const value_type, memory_space> buffer,
    ScalarArrayView<const pcms::LO, memory_space> permutation) const override { PYBIND11_OVERRIDE_PURE(void, PythonFieldAdapterConcept, Deserialize, buffer, permutation); }

    std::vector<GO> GetGids() const override { PYBIND11_OVERRIDE_PURE(std::vector<GO>, PythonFieldAdapterConcept, GetGids); }

    ReversePartitionMap GetReversePartitionMap(const redev::Partition& partition) const override { PYBIND11_OVERRIDE_PURE(ReversePartitionMap, PythonFieldAdapterConcept, GetReversePartitionMap, partition); }

    bool RankParticipatesCouplingCommunication() const noexcept override { PYBIND11_OVERRIDE_PURE(bool, PythonFieldAdapterConcept, RankParticipatesCouplingCommunication ); }
};
} // namespace pcms


PYBIND11_MODULE(pcms_python, m) {
  m.doc() = "PCMS Coupling Client C++ Pybind11 Bindings";

  m.def("pcms_kokkos_initialize_without_args", &pcms_kokkos_initialize_without_args);
  m.def("pcms_kokkos_finalize", &pcms_kokkos_finalize);
  
  py::class_<pcms::PythonFieldAdapterConcept, pcms::PythonFieldAdapterTramp>(m, "PythonFieldAdapterConcept")
            .def(py::init<>())
            .def("Serialize", &pcms::PythonFieldAdapterConcept::Serialize)
            .def("Deserialize", &pcms::PythonFieldAdapterConcept::Deserialize)
            .def("GetGids", &pcms::PythonFieldAdapterConcept::GetGids)
            .def("GetReversePartitionMap", &pcms::PythonFieldAdapterConcept::GetReversePartitionMap)
            .def("RankParticipatesCouplingCommunication", &pcms::PythonFieldAdapterConcept::RankParticipatesCouplingCommunication);

  py::class_<pcms::PythonFieldAdapter>(m, "PythonFieldAdapter")
            .def(py::init<pcms::PythonFieldAdapterConcept&>());

  py::enum_<Mode>(m, "Mode")
        .value("Deferred", Mode::Deferred)
        .value("Synchronous", Mode::Synchronous);
        
  py::class_<CoupledField>(m, "CoupledField")
        .def(py::init<const std::string&, pcms::PythonFieldAdapter&, MPI_Comm, redev::Redev&, redev::Channel&, bool>())
        .def("Send", &CoupledField::Send, py::arg("mode") = Mode::Synchronous)
        .def("Receive", &CoupledField::Receive);

  py::class_<CouplerClient>(m, "CouplerClient")
        .def(py::init<std::string, MPI_Comm, redev::TransportType, adios2::Params, std::string>())
        .def("GetPartition", &CouplerClient::GetPartition, py::return_value_policy::reference) //redev::Partition&
        .def("AddField", &CouplerClient::AddField<pcms::PythonFieldAdapter&>, 
         py::arg("name"), py::arg("field_adapter"), py::arg("participates") = true, 
         py::return_value_policy::reference) //CoupledField*
        .def("SendField", &CouplerClient::SendField, py::arg("name"), py::arg("mode") = Mode::Synchronous) //void
        .def("ReceiveField", &CouplerClient::ReceiveField, py::arg("name")) //void
        .def("InSendPhase", &CouplerClient::InSendPhase) //bool
        .def("InReceivePhase", &CouplerClient::InReceivePhase) //bool
        .def("BeginSendPhase", &CouplerClient::BeginSendPhase) //void
        .def("EndSendPhase", &CouplerClient::EndSendPhase) //void
        .def("BeginReceivePhase", &CouplerClient::BeginReceivePhase) //void
        .def("EndReceivePhase", &CouplerClient::EndReceivePhase); //void
}


