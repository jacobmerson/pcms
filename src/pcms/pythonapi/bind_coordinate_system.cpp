#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "pcms/field/coordinate_system.hpp"
#include "pcms/field/coordinate_systems/cartesian.hpp"
#include "pcms/field/coordinate_systems/cylindrical.hpp"
#include "pcms/field/coordinate_view.hpp"
#include "numpy_view.h"
#include "py_value_types.h"

namespace py = pybind11;

namespace pcms
{

namespace
{

using SystemHolder = std::shared_ptr<CoordinateSystem>;

std::string Repr(const CoordinateSystem& system)
{
  return "<pcms.CoordinateSystem '" + std::string(system.Kind()) + "' dim " +
         std::to_string(system.Dimension()) + ">";
}

} // namespace

void bind_coordinate_system_module(py::module& m)
{
  // Concrete systems are registered as subclasses so Python sees the dynamic
  // type of any system pcms returns (isinstance(cs, pcms.csys.Cartesian)).
  py::class_<CoordinateSystem, SystemHolder>(
    m, "CoordinateSystem",
    "What coordinate numbers mean. Create one through the classes in "
    "pcms.csys.")
    .def(
      "kind",
      [](const CoordinateSystem& self) { return std::string(self.Kind()); },
      "Family label, e.g. 'cartesian'")
    .def(
      "dimension",
      [](const CoordinateSystem& self) { return self.Dimension(); },
      "Number of coordinate columns (0 for the deferred Cartesian system)")
    .def("has_orthogonal_basis",
         [](const CoordinateSystem& self) { return self.HasOrthogonalBasis(); })
    .def(
      "has_unit_scale_factors",
      [](const CoordinateSystem& self) { return self.HasUnitScaleFactors(); })
    .def(
      "__eq__",
      [](const SystemHolder& a, const SystemHolder& b) {
        return SameCoordinateSystem(a, b);
      },
      py::is_operator())
    .def("__hash__",
         [](const CoordinateSystem& self) {
           return std::hash<std::string>{}(std::string(self.Kind()) + "/" +
                                           std::to_string(self.Dimension()));
         })
    .def("__repr__", [](const CoordinateSystem& self) { return Repr(self); });

  auto csys = m.def_submodule("csys", "Concrete coordinate systems");
  py::class_<csys::Cartesian, CoordinateSystem,
             std::shared_ptr<csys::Cartesian>>(csys, "Cartesian")
    .def_static(
      "create",
      [](int dim) { return ToPyHolder(csys::Cartesian::Create(dim)); },
      py::arg("dim"), "The Cartesian coordinate system of dimension 1..3")
    .def_static(
      "deferred", []() { return ToPyHolder(csys::Cartesian::Deferred()); },
      "Cartesian with the dimension taken from the coordinates it is used "
      "with");
  py::class_<csys::CylindricalRThetaZ, CoordinateSystem,
             std::shared_ptr<csys::CylindricalRThetaZ>>(csys,
                                                        "CylindricalRThetaZ")
    .def_static(
      "create", []() { return ToPyHolder(csys::CylindricalRThetaZ::Create()); },
      "Cylindrical (r, theta, z) with theta = atan2(y, x); dimension 3");
  py::class_<csys::CylindricalRZ, CoordinateSystem,
             std::shared_ptr<csys::CylindricalRZ>>(csys, "CylindricalRZ")
    .def_static(
      "create", []() { return ToPyHolder(csys::CylindricalRZ::Create()); },
      "(R, Z) poloidal section; dimension 2");

  m.def(
    "same_coordinate_system",
    [](const SystemHolder& a, const SystemHolder& b) {
      return SameCoordinateSystem(a, b);
    },
    py::arg("a"), py::arg("b"));
  m.def(
    "has_identity_metric",
    [](const CoordinateSystem& system) { return HasIdentityMetric(system); },
    py::arg("system"));
  m.def(
    "resolve_coordinate_system",
    [](const SystemHolder& system, int dim) {
      return ToPyHolder(ResolveCoordinateSystem(system, dim));
    },
    py::arg("system"), py::arg("dim"),
    "Resolves a deferred Cartesian system to dimension `dim`; checks any "
    "other system has that dimension");

  py::class_<PyCoordinateView>(m, "CoordinateView",
                               "Coordinates tagged with their coordinate "
                               "system")
    .def(py::init([](const SystemHolder& system,
                     const NumpyInput<Real>& coordinates) {
           const auto extents = detail::NumpyExtents<2>(coordinates);
           // pcms may keep the coordinates, so the view owns a copy
           py::array_t<Real> values(std::vector<py::ssize_t>{
             coordinates.shape(0), coordinates.shape(1)});
           if (values.size() > 0) {
             std::memcpy(values.mutable_data(), coordinates.data(),
                         sizeof(Real) * values.size());
           }
           detail::SetReadOnly(values);
           return PyCoordinateView{ResolveCoordinateSystem(system, extents[1]),
                                   values};
         }),
         py::arg("coordinate_system"), py::arg("coordinates"),
         "Coordinates from a 2D array (num_points × dim), which is copied. A "
         "deferred Cartesian system is resolved to the array's dimension.")
    .def("get_coordinate_system",
         [](const PyCoordinateView& self) { return ToPyHolder(self.system); })
    .def(
      "get_values", [](const PyCoordinateView& self) { return self.values; },
      "The coordinates as a read-only 2D array (num_points × dim)")
    .def("__repr__", [](const PyCoordinateView& self) {
      return "<pcms.CoordinateView " + std::to_string(self.values.shape(0)) +
             " points in " + Repr(*self.system) + ">";
    });
}

} // namespace pcms
