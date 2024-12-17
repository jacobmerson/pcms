#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pcms/capi/kokkos.h>
#include <mpi4py/mpi4py.h>
#include "../client.h"
#include "../xgc_field_adapter.h"
#include <type_traits>



namespace py = pybind11;
using pcms::Mode;
using pcms::CoupledField;
using pcms::CouplerClient;


namespace pcms{
using value_type = double;

class PythonFieldAdapterConcept {
      public: 

      using memory_space = HostMemorySpace;
      using value_type = double;
      using coordinate_element_type = double;
      
      virtual int Serialize(py::array_t<value_type>& buffer,
      py::array_t<const pcms::LO>& permutation) const = 0; 

      virtual void Deserialize(py::array_t<value_type>& buffer,
      py::array_t<const pcms::LO>& permutation) const = 0;

    [[nodiscard]] virtual py::array_t<GO> GetGids() const = 0;

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
    
      py::buffer_info info_buff (
        buffer.data_handle(),
        sizeof(value_type),
        py::format_descriptor<value_type>::format(),
        1,
        { buffer.size() },
        { sizeof(value_type) }
      );

      py::buffer_info info_perm (
        const_cast<pcms::LO*>(permutation.data_handle()),
        sizeof(pcms::LO),
        py::format_descriptor<pcms::LO>::format(),
        1,
        { permutation.size() },
        { sizeof(pcms::LO) }
      );
    
      py::array_t<value_type> buff(info_buff);
      py::array_t<const pcms::LO> perm(info_perm);

      return adapter_.Serialize(buff,perm);
    }

      void Deserialize(
    ScalarArrayView<const value_type, memory_space> buffer,
    ScalarArrayView<const pcms::LO, memory_space> permutation) const {
    
      py::buffer_info info_buff (
        const_cast<value_type*>(buffer.data_handle()),
        sizeof(value_type),
        py::format_descriptor<value_type>::format(),
        1,
        { buffer.size() },
        { sizeof(value_type) }
      );
      
      py::buffer_info info_perm (
        const_cast<pcms::LO*>(permutation.data_handle()),
        sizeof(pcms::LO),
        py::format_descriptor<pcms::LO>::format(),
        1,
        { permutation.size() },
        { sizeof(pcms::LO) }
      );
    
      py::array_t<value_type> buff(info_buff);
      py::array_t<const pcms::LO> perm(info_perm);

      return adapter_.Deserialize(buff,perm);
    }

    [[nodiscard]] std::vector<GO> GetGids() const {
        py::array_t<GO> a = adapter_.GetGids();
        py::buffer_info info = a.request();

        int size = info.size;
        const GO* data = static_cast<const GO*>(info.ptr);
        return std::vector<GO>(data, data+size);
    }

    [[nodiscard]] ReversePartitionMap GetReversePartitionMap(
    const redev::Partition& partition) const {
      return adapter_.GetReversePartitionMap(partition);
    }

    [[nodiscard]] bool RankParticipatesCouplingCommunication() const noexcept {
      return adapter_.RankParticipatesCouplingCommunication();
    }
};

//Trampoline class
class PythonFieldAdapterTramp : public PythonFieldAdapterConcept {
  public:
    using PythonFieldAdapterConcept::PythonFieldAdapterConcept;

    int Serialize(py::array_t<value_type>& buffer,
      py::array_t<const pcms::LO>& permutation) const override { PYBIND11_OVERRIDE_PURE(int, PythonFieldAdapterConcept, Serialize, buffer, permutation); }

    void Deserialize(py::array_t<value_type>& buffer,
      py::array_t<const pcms::LO>& permutation) const override { PYBIND11_OVERRIDE_PURE(void, PythonFieldAdapterConcept, Deserialize, buffer, permutation); }

    [[nodiscard]] py::array_t<GO> GetGids() const override { PYBIND11_OVERRIDE_PURE(py::array_t<GO>, PythonFieldAdapterConcept, GetGids); }

    [[nodiscard]] ReversePartitionMap GetReversePartitionMap(const redev::Partition& partition) const override { PYBIND11_OVERRIDE_PURE(ReversePartitionMap, PythonFieldAdapterConcept, GetReversePartitionMap, partition); }

    [[nodiscard]] bool RankParticipatesCouplingCommunication() const noexcept override { PYBIND11_OVERRIDE_PURE(bool, PythonFieldAdapterConcept, RankParticipatesCouplingCommunication ); }
};

template <typename T>
class PythonArrayMask {
    public:

    using memory_space = HostMemorySpace;
    
    PythonArrayMask(py::array_t<const int8_t>& mask) : mask_(Convert(mask)) {}
    
    auto Apply(py::array_t<const T>& data, py::array_t<T>& filtered_data, py::array_t<const pcms::LO>& permutation) {
      mask_.Apply(Convert(data),Convert(filtered_data),Convert(permutation));
    }

    auto ToFullArray(py::array_t<const T>& filtered_data, py::array_t<T>& output_array, py::array_t<const pcms::LO> permutation = {}) {
     mask_.ToFullArray(Convert(filtered_data),Convert(output_array),Convert(permutation));
    }

    int Size() {
      return mask_.Size();
    }

    auto GetMap() const {
      return mask_.GetMap();
    }


    private:
      ArrayMask<memory_space> mask_;

      template <typename U>
      ScalarArrayView<U,memory_space> Convert(const py::array_t<U>& arr) {
        py::buffer_info arr_buf = arr.request();
        //PCMS_ASSERT
        return ScalarArrayView<U, memory_space>{
          static_cast<U*>(arr_buf.ptr), arr_buf.size
        };
      }
}; 
//use ReadReverseClassificationVertex string version
class PyReverseClassificationVertex { 
public:
    PyReverseClassificationVertex(const ReverseClassificationVertex& rcv) : rcv_(rcv) {}
    // Python-friendly method to get data
    py::dict get_data() const {
        py::dict result;
        for (auto it = rcv_.begin(); it != rcv_.end(); ++it) {
            const DimID& dimID = it->first;
            const std::set<LO>& verts = it->second;
            result[py::make_tuple(dimID.dim, dimID.id)] = py::cast(verts);
        }
        return result;
    }

private:
    const ReverseClassificationVertex& rcv_;
};
} // namespace pcms

static MPI_Comm* GetMPIComm(py::object py_comm) {
    auto comm_ptr = PyMPIComm_Get(py_comm.ptr());

    if(!comm_ptr)
        throw py::error_already_set();

    return comm_ptr;
}

PYBIND11_MODULE(pcms_python, m) {
  m.doc() = "PCMS Coupling Client C++ Pybind11 Bindings";

  m.def("pcms_kokkos_initialize_without_args", &pcms_kokkos_initialize_without_args);
  m.def("pcms_kokkos_finalize", &pcms_kokkos_finalize);

  m.def("RankPartition", [](const pcms::LO dim, const pcms::LO id, const redev::Partition& partition) {
      return std::visit(pcms::detail::GetRank{pcms::DimID{dim,id}}, partition);
  });

  m.def("Bcast", [](py::array_t<pcms::value_type> data, int plane_root, py::object plane_comm) {
      auto buf_info = data.request();
      auto* comm_ptr = GetMPIComm(plane_comm);
      MPI_Bcast(buf_info.ptr, buf_info.size, redev::getMpiType(pcms::value_type{}), plane_root, *comm_ptr);
  });
  
  py::class_<pcms::PythonFieldAdapterConcept, pcms::PythonFieldAdapterTramp>(m, "PythonFieldAdapterConcept")
            .def(py::init<>())
            .def("Serialize", &pcms::PythonFieldAdapterConcept::Serialize)
            .def("Deserialize", &pcms::PythonFieldAdapterConcept::Deserialize)
            .def("GetGids", &pcms::PythonFieldAdapterConcept::GetGids)
            .def("GetReversePartitionMap", &pcms::PythonFieldAdapterConcept::GetReversePartitionMap)
            .def("RankParticipatesCouplingCommunication", &pcms::PythonFieldAdapterConcept::RankParticipatesCouplingCommunication);

  py::class_<pcms::PythonFieldAdapter>(m, "PythonFieldAdapter")
            .def(py::init<pcms::PythonFieldAdapterConcept&>())
            .def("Serialize", &pcms::PythonFieldAdapter::Serialize)
            .def("Deserialize", &pcms::PythonFieldAdapter::Deserialize)
            .def("GetGids", &pcms::PythonFieldAdapter::GetGids)
            .def("GetReversePartitionMap", &pcms::PythonFieldAdapter::GetReversePartitionMap)
            .def("RankParticipatesCouplingCommunication", &pcms::PythonFieldAdapter::RankParticipatesCouplingCommunication);


  py::class_<pcms::PythonArrayMask<double>>(m, "PythonArrayMask")
        .def(py::init<py::array_t<const int8_t>&>())
        .def("Apply", &pcms::PythonArrayMask<double>::Apply)
        .def("ToFullArray", &pcms::PythonArrayMask<double>::ToFullArray)
        .def("Size", &pcms::PythonArrayMask<double>::Size)
        .def("GetMap", &pcms::PythonArrayMask<double>::GetMap);

  py::class_<pcms::PyReverseClassificationVertex>(m, "PyReverseClassificationVertex")
        .def(py::init<const pcms::ReverseClassificationVertex&>())
        .def("get_data", &pcms::PyReverseClassificationVertex::get_data);

  py::enum_<Mode>(m, "Mode")
        .value("Deferred", Mode::Deferred)
        .value("Synchronous", Mode::Synchronous);
  
  py::class_<pcms::CoupledField>(m, "CoupledField")
        .def(py::init<const std::string&, pcms::PythonFieldAdapter&, MPI_Comm, redev::Redev&, redev::Channel&, bool>())
        .def("Send", &CoupledField::Send, py::arg("mode") = Mode::Synchronous)
        .def("Receive", &CoupledField::Receive);

  py::class_<pcms::CouplerClient>(m, "CouplerClient")
        .def(py::init<std::string, MPI_Comm, redev::TransportType, adios2::Params, std::string>())
        .def("GetPartition", &CouplerClient::GetPartition, py::return_value_policy::reference) //redev::Partition&
        .def("AddField",[](CouplerClient& self, const std::string& name, pcms::PythonFieldAdapter& fa, bool participates = true){
          self.AddField(name,fa,participates);
        })
        .def("SendField", &CouplerClient::SendField, py::arg("name"), py::arg("mode") = Mode::Synchronous) //void
        .def("ReceiveField", &CouplerClient::ReceiveField, py::arg("name")) //void
        .def("InSendPhase", &CouplerClient::InSendPhase) //bool
        .def("InReceivePhase", &CouplerClient::InReceivePhase) //bool
        .def("BeginSendPhase", &CouplerClient::BeginSendPhase) //void
        .def("EndSendPhase", &CouplerClient::EndSendPhase) //void
        .def("BeginReceivePhase", &CouplerClient::BeginReceivePhase) //void
        .def("EndReceivePhase", &CouplerClient::EndReceivePhase); //void
}


