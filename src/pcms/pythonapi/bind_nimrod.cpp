#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pcms/field/function_space.h"
#include "pcms/field/function_space/nimrod.h"

namespace py = pybind11;

namespace pcms
{

// Bind the NIMROD evaluation backend. Evaluation itself is driven through the
// already-bound FunctionSpace / EvaluationRequest / PointEvaluator base-class
// API (see bind_field_base.cpp and bind_transfer_field.cpp), so this only needs
// to expose the NIMROD-specific construction surface:
//   traits = pcms.NimrodInputTraits(); traits.nisp = ...
//   space  = pcms.NimrodFunctionSpace.from_dump("dump.h5", traits, "bvn", eq=2)
//   field  = space.make_field("b")
//   req    = pcms.EvaluationRequest.from_coordinates(rzp, pcms.CoordinateSystem.Cylindrical)
//   vals   = space.create_point_evaluator(req).evaluate(field, out)
void bind_nimrod_module(py::module& m)
{
  py::class_<NimrodInputTraits>(m, "NimrodInputTraits")
    .def(py::init<>(), "Default-constructed NIMROD input traits.")
    .def_readwrite("nisp", &NimrodInputTraits::nisp,
                   "number of ion species; density 'n' has nqty = nisp + 1")
    .def_readwrite("nnsp", &NimrodInputTraits::nnsp,
                   "number of neutral species")
    .def_readwrite("ds_nqty", &NimrodInputTraits::ds_nqty,
                   "number of diffusivity-shape components (field 'd')")
    .def_readwrite("zimp", &NimrodInputTraits::zimp, "impurity charge")
    .def_readwrite("nmodes", &NimrodInputTraits::nmodes,
                   "number of Fourier modes")
    .def("nqty_for_field", &NimrodInputTraits::NqtyForField, py::arg("field_id"),
         "Number of components a single-character NIMROD field id carries.");

  py::class_<NimrodFunctionSpace, FunctionSpace>(m, "NimrodFunctionSpace")
    .def_static(
      "from_dump",
      [](const std::string& dumpfile, NimrodInputTraits traits,
         const std::string& fieldlist, int eq) {
        return NimrodFunctionSpace::FromDump(dumpfile, traits, fieldlist, eq);
      },
      py::arg("dumpfile"), py::arg("traits"), py::arg("fieldlist") = "bvnpf",
      py::arg("eq") = 2,
      "Open a NIMROD dump for evaluation (calls c_nimfield_init; requires "
      "nimrod.in in the working directory). Becomes the process-active dump.")
    .def(
      "make_field",
      [](const NimrodFunctionSpace& self, const std::string& field_id) {
        return self.MakeField(field_id);
      },
      py::arg("field_id"),
      "Build a Field naming a NIMROD field to evaluate (e.g. 'b', 'n', 'p').")
    .def(
      "make_field",
      [](const NimrodFunctionSpace& self, const std::string& field_id, int eq) {
        return self.MakeField(field_id, eq);
      },
      py::arg("field_id"), py::arg("eq"),
      "Build a Field naming a NIMROD field with an explicit eq selector.")
    .def("nqty_for_field",
         [](const NimrodFunctionSpace& self, const std::string& field_id) {
           return self.GetTraits().NqtyForField(field_id.at(0));
         },
         py::arg("field_id"),
         "Number of components the named NIMROD field produces at dmode=0.");
}

} // namespace pcms
