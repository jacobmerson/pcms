#include <optional>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pcms/discretization/discretization/omega_h.hpp"
#include "pcms/field/coordinate_map.hpp"
#include "pcms/field/coordinate_system.hpp"
#include "pcms/field/coordinate_systems/cartesian.hpp"
#include "pcms/field/coordinate_view.hpp"
#include "pcms/geometry/bound_source.hpp"
#include "pcms/field/differential_operator.hpp"
#include "pcms/field/evaluator/mls_options.h"
#include "pcms/field/evaluator/omega_h_geometry.hpp"
#include "pcms/field/field.h"
#include "pcms/field/field_data.h"
#include "pcms/field/field_layout.h"
#include "pcms/field/function_space.h"
#include "pcms/field/function_space/lagrange.h"
#include "pcms/field/function_space/polynomial_reconstruction.hpp"
#include "pcms/field/gradient_source.hpp"
#include "pcms/field/point_evaluator_factory.hpp"
#include "pcms/field/uniform_grid_binary_field.h"
#include "pcms/field/value_view.hpp"
#include "pcms/geometry/geometry.hpp"
#include "pcms/utility/arrays.h"
#include "pcms/utility/uniform_grid.h"
#include "numpy_array_transform.h"
#include "py_value_types.h"

namespace py = pybind11;

namespace pcms
{

namespace
{

using SystemHolder = std::shared_ptr<CoordinateSystem>;

void SetDOFHolderData(Field<Real>& self, const PyValueView& v)
{
  auto array = NumpyInput<Real>::ensure(v.values);
  if (!array) {
    throw py::type_error("set_dof_holder_data: values must be float64");
  }
  self.SetDOFHolderDataHost(ValueView<const Real, HostMemorySpace>(
    v.basis, NumpyToHostView<2, Real>(array)));
}

/// The Python object of `geometry`. An Omega_h geometry refers to its mesh, so
/// the object keeps the mesh alive, as create_omega_h_geometry does; it does
/// not keep the caller alive, which would form a cycle with the spaces that
/// keep their geometry alive.
py::object GeometryToPython(const std::shared_ptr<const Geometry>& geometry)
{
  py::object result = py::cast(ToPyHolder(geometry));
  if (const auto* omega_h =
        dynamic_cast<const OmegaHGeometry*>(geometry.get())) {
    py::object mesh =
      py::cast(&omega_h->GetMesh(), py::return_value_policy::reference);
    py::detail::keep_alive_impl(result, mesh);
  }
  return result;
}

} // namespace

void bind_create_field_module(py::module& m)
{
  py::class_<FieldLayout, std::shared_ptr<FieldLayout>>(
    m, "FieldLayout", "How a field's DOF holders are laid out")
    .def("get_num_components", &FieldLayout::GetNumComponents)
    .def("get_num_dof_holder", &FieldLayout::GetNumDofHolder,
         "Number of DOF holders this rank owns")
    .def("get_num_global_dof_holder", &FieldLayout::GetNumGlobalDofHolder)
    .def("get_dimension", &FieldLayout::GetDimension)
    .def("is_distributed", &FieldLayout::IsDistributed)
    .def(
      "get_coordinate_system",
      [](const FieldLayout& self) {
        return ToPyHolder(self.GetCoordinateSystem());
      },
      "What this layout's DOF-holder coordinates mean")
    .def(
      "get_dof_holder_coordinates",
      [](py::object self) {
        return ToPyCoordinateView(
          self.cast<const FieldLayout&>().GetDOFHolderCoordinates(), self);
      },
      "The owned DOF-holder coordinates (num_dof_holders × dim), read-only");

  py::class_<FieldData<Real>>(m, "FieldData", "A field's stored values")
    .def("get_value_type", &FieldData<Real>::GetValueType)
    .def("get_value_basis", &FieldData<Real>::GetValueBasis,
         "The basis the stored components are in");

  py::class_<Field<Real>>(
    m, "Field", "DOF-holder values together with the layout that orders them")
    .def("get_name", &Field<Real>::GetName)
    .def("get_layout", &Field<Real>::GetLayout,
         py::return_value_policy::reference_internal)
    .def(
      "get_data",
      [](Field<Real>& self) -> FieldData<Real>& { return self.GetData(); },
      py::return_value_policy::reference_internal)
    .def(
      "get_dof_holder_data",
      [](py::object self) {
        return ToPyValueView(self.cast<const Field<Real>&>().GetDOFHolderData(),
                             self);
      },
      "The DOF-holder values (num_dof_holders × num_components) as a "
      "ValueView whose array is read-only. When the field's data is in host "
      "memory the array shares it, so it reflects later writes; otherwise it "
      "is a copy.")
    .def("set_dof_holder_data", &SetDOFHolderData, py::arg("data"),
         "Set the DOF-holder values from a ValueView (num_dof_holders × "
         "num_components) whose basis must match the field's")
    .def(
      "set_dof_holder_data",
      [](Field<Real>& self, const NumpyInput<Real>& data) {
        RequireScalarForPlainArray("set_dof_holder_data",
                                   self.GetData().GetValueBasis());
        self.SetDOFHolderDataHost(ValueView<const Real, HostMemorySpace>(
          ValueBasis{},
          NumpyToHostComponentView<Real>(data, PerEntityComponents(data))));
      },
      py::arg("data"),
      "Set a scalar field's DOF-holder values from a plain 2D array "
      "(num_dof_holders × num_components). A 1D array (num_dof_holders,) is "
      "accepted when the field has one component.")
    .def(
      "set_dof_holder_data_unchecked",
      [](Field<Real>& self, const NumpyInput<Real>& data) {
        const auto& layout = self.GetLayout();
        const LO num_dof_holders = layout.GetNumDofHolder();
        auto view =
          NumpyToHostComponentView<Real>(data, layout.GetNumComponents());
        if (view.extent(0) != static_cast<size_t>(num_dof_holders)) {
          throw std::invalid_argument(
            "set_dof_holder_data_unchecked: data must have " +
            std::to_string(num_dof_holders) + " rows, got " +
            std::to_string(view.extent(0)));
        }
        self.SetDOFHolderDataUncheckedHost(view);
      },
      py::arg("data"),
      "Set the DOF-holder values without checking their basis (the shape is "
      "still checked). A 1D array (num_dof_holders,) is accepted when the "
      "field has one component.")
    .def(
      "synchronize", [](Field<Real>& self) { self.Synchronize(); },
      "Bring the copies of DOF holders other ranks own (e.g. Omega_h ghosts) "
      "up to date with their owners' values");

  py::class_<Function<Real>, Field<Real>>(
    m, "Function", "A Field that knows its FunctionSpace")
    .def("get_space", [](const Function<Real>& self) {
      return ToPyHolder(self.GetSpacePtr());
    });

  py::class_<Geometry, std::shared_ptr<Geometry>>(
    m, "Geometry",
    "A discretization's coordinates and point search; function spaces are "
    "created on it")
    .def("get_discretization",
         [](const Geometry& g) { return ToPyHolder(g.GetDiscretization()); })
    .def("get_coordinate_system",
         [](const Geometry& g) { return ToPyHolder(g.GetCoordinateSystem()); })
    .def("supports_nearest_boundary", &Geometry::SupportsNearestBoundary)
    .def(
      "bind",
      [](const Geometry& g, const PyCoordinateView& coords,
         OutOfBoundsPolicy policy, std::shared_ptr<CoordinateMap> map) {
        // the bound source copies the points
        auto device = CopyCoordinatesToDevice(coords);
        return std::const_pointer_cast<BoundSource>(
          g.Bind(device.view, policy, std::move(map)));
      },
      py::arg("coordinates"), py::arg("policy") = OutOfBoundsPolicy{},
      py::arg("map") = py::none(), py::keep_alive<0, 1>(),
      "Localize the points of a CoordinateView (in get_coordinate_system(), "
      "or in the source system of `map`, a CoordinateMap to "
      "get_coordinate_system(): the points are then located at their images "
      "and evaluated values re-expressed through it); the search is done "
      "once here")
    .def(
      "bind",
      [](const Geometry& g, const FunctionSpace& space,
         OutOfBoundsPolicy policy, std::shared_ptr<CoordinateMap> map) {
        auto layout = space.GetLayout();
        return std::const_pointer_cast<BoundSource>(
          g.Bind(layout->GetDOFHolderCoordinates(),
                 DOFHolderOwners(*layout, g.shared_from_this()), policy,
                 std::move(map)));
      },
      py::arg("space"), py::arg("policy") = OutOfBoundsPolicy{},
      py::arg("map") = py::none(), py::keep_alive<0, 1>(),
      "Bind the DOF-holder coordinates of `space`, through `map` if given; "
      "without a map, a space whose DOF holders are entities of this geometry "
      "binds them as known owners (with a map that hint is ignored)");

  py::class_<OmegaHGeometry, Geometry, std::shared_ptr<OmegaHGeometry>>(
    m, "OmegaHGeometry", "The geometry of an Omega_h simplex mesh");

  m.def(
    "create_omega_h_geometry",
    [](Omega_h::Mesh& mesh, const SystemHolder& system, int divisions) {
      SearchOptions options;
      options.divisions = divisions;
      return OmegaHGeometry::Create(mesh, CartesianIfNone(system), options);
    },
    py::arg("mesh"), py::arg("system") = py::none(), py::arg("divisions") = 0,
    py::keep_alive<0, 1>(),
    "Create the geometry of an Omega_h mesh: its coordinate system (default "
    "Cartesian) and point search (`divisions` grid divisions per axis, 0 to "
    "derive them from the element count)");

  py::class_<BoundSource, std::shared_ptr<BoundSource>>(
    m, "BoundSource",
    "A set of query points localized on a geometry; evaluators created from "
    "it share its search")
    .def(
      "get_geometry",
      [](const BoundSource& self) {
        return GeometryToPython(self.GetGeometry());
      },
      "The geometry the points were localized on")
    .def("num_points", &BoundSource::NumPoints)
    .def_property_readonly(
      "policy", [](const BoundSource& self) { return self.Policy(); })
    .def(
      "found_points",
      [](py::object self) {
        return IndicesToNumpy(self.cast<const BoundSource&>().FoundPoints(),
                              self);
      },
      "Indices of the query points the source evaluates, read-only")
    .def(
      "missing_points",
      [](py::object self) {
        return IndicesToNumpy(self.cast<const BoundSource&>().MissingPoints(),
                              self);
      },
      "Indices of the query points the source does not evaluate, read-only");

  py::enum_<DifferentialOperator>(m, "DifferentialOperator",
                                  "The quantity a point evaluator computes")
    .value("Value", DifferentialOperator::Value)
    .value("Gradient", DifferentialOperator::Gradient)
    .value("Divergence", DifferentialOperator::Divergence)
    .value("Curl", DifferentialOperator::Curl)
    .value("Hessian", DifferentialOperator::Hessian);

  py::class_<PointEvaluatorFactory, std::shared_ptr<PointEvaluatorFactory>>(
    m, "PointEvaluatorFactory",
    "Creates evaluators at the points of bound sources of its geometry")
    .def(
      "get_geometry",
      [](const PointEvaluatorFactory& self) {
        return GeometryToPython(self.GetGeometry());
      },
      "The geometry whose bound sources this source accepts; bind query points "
      "with its bind()")
    .def(
      "create_point_evaluator",
      [](const PointEvaluatorFactory& self,
         const std::shared_ptr<BoundSource>& bound, DifferentialOperator op) {
        return self.CreatePointEvaluator<Real>(bound, op);
      },
      py::arg("bound"), py::arg("op") = DifferentialOperator::Value,
      py::keep_alive<0, 1>(),
      "Create an evaluator at the points of `bound` (a result of bind() on "
      "this source's geometry) that applies `op`: the values for "
      "DifferentialOperator.Value, the gradient for "
      "DifferentialOperator.Gradient")
    .def("supports", &PointEvaluatorFactory::Supports, py::arg("op"),
         "Whether create_point_evaluator accepts `op`");

  m.def(
    "gradient",
    [](std::shared_ptr<PointEvaluatorFactory> source) {
      return std::const_pointer_cast<PointEvaluatorFactory>(
        Gradient(std::move(source)));
    },
    py::arg("source"),
    "A source whose evaluators write the gradient of the field they are given; "
    "it accepts the bound sources of `source`'s geometry");

  m.def(
    "divergence",
    [](std::shared_ptr<PointEvaluatorFactory> source) {
      return std::const_pointer_cast<PointEvaluatorFactory>(
        Divergence(std::move(source)));
    },
    py::arg("source"),
    "A source whose evaluators write the divergence of the field they are "
    "given; it accepts the bound sources of `source`'s geometry");

  m.def(
    "curl",
    [](std::shared_ptr<PointEvaluatorFactory> source) {
      return std::const_pointer_cast<PointEvaluatorFactory>(
        Curl(std::move(source)));
    },
    py::arg("source"),
    "A source whose evaluators write the curl of the field they are given; it "
    "accepts the bound sources of `source`'s geometry");

  py::class_<FunctionSpace, PointEvaluatorFactory,
             std::shared_ptr<FunctionSpace>>(m, "FunctionSpace")
    .def(
      "create_function",
      [](const FunctionSpace& self, std::string name,
         VarianceSignature variance, const SystemHolder& system,
         std::optional<ComponentScaling> scaling) {
        return self.CreateFunction<Real>(std::move(name), variance, system,
                                         scaling);
      },
      py::arg("name") = "", py::arg("variance") = values::Scalar,
      py::arg("system") = py::none(), py::arg("scaling") = py::none(),
      // The function refers to the space's mesh (an Omega_h function keeps a
      // tag on it), so the space, which keeps the mesh alive, outlives it.
      py::keep_alive<0, 1>(),
      "Create a Function in this space. Vector and tensor values are in "
      "`system`'s basis (default: this space's coordinate system), with "
      "`scaling` required when that system admits both scalings.")
    .def("get_layout",
         [](const FunctionSpace& self) { return ToPyHolder(self.GetLayout()); })
    .def(
      "get_coordinate_system",
      [](const FunctionSpace& self) {
        return ToPyHolder(self.GetCoordinateSystem());
      },
      "The coordinate system of the space's geometry")
    .def(
      "mesh",
      [](const FunctionSpace& self) -> Omega_h::Mesh& {
        auto disc = std::dynamic_pointer_cast<const OmegaHDiscretization>(
          self.GetDiscretization());
        if (!disc) {
          throw std::runtime_error(
            "FunctionSpace::mesh: this function space is not backed by an "
            "Omega_h mesh");
        }
        return disc->GetMesh();
      },
      py::return_value_policy::reference,
      "The Omega_h mesh this function space is defined on (Omega_h backend "
      "only).");

  py::class_<LagrangeFunctionSpace, FunctionSpace,
             std::shared_ptr<LagrangeFunctionSpace>>
    lagrange_space(m, "LagrangeFunctionSpace");

  // Backend selecting the underlying field layout. The conservative-projection
  // transfer operators require the native Omega_h backend.
  py::enum_<LagrangeFunctionSpace::Backend>(lagrange_space, "Backend")
    .value("MeshFields", LagrangeFunctionSpace::Backend::MeshFields,
           "MeshFields-backed layout (default when MeshFields is enabled)")
    .value("OmegaH", LagrangeFunctionSpace::Backend::OmegaH,
           "Native Omega_h Lagrange layout (required by the conservative and "
           "Monte Carlo projection transfer operators)")
    .export_values();

  // The mesh-backed spaces refer to the mesh, so it is kept alive with them.
  lagrange_space
    .def_static(
      "from_geometry",
      [](const std::shared_ptr<OmegaHGeometry>& geometry, int order,
         int num_components, std::string global_id_name,
         LagrangeFunctionSpace::Backend backend) {
        return LagrangeFunctionSpace::FromGeometry(
          geometry, order, num_components, std::move(global_id_name), backend);
      },
      py::arg("geometry"), py::arg("order"), py::arg("num_components"),
      py::arg("global_id_name") = "global",
      py::arg("backend") = LagrangeFunctionSpace::DefaultBackend,
      py::keep_alive<0, 1>(),
      "Create a LagrangeFunctionSpace on an Omega_h geometry; spaces on one "
      "geometry accept each other's bound sources.")
    .def_static(
      "from_uniform_grid",
      [](const UniformGrid<2>& grid, int num_components,
         const SystemHolder& mesh_coordinate_system, int order) {
        return LagrangeFunctionSpace::FromUniformGrid(
          grid, num_components, CartesianIfNone(mesh_coordinate_system), order);
      },
      py::arg("grid"), py::arg("num_components"),
      py::arg("mesh_coordinate_system") = py::none(), py::arg("order") = 1,
      "Create a LagrangeFunctionSpace from a 2D uniform grid. The coordinate "
      "system defaults to Cartesian.")
    .def_static(
      "from_uniform_grid",
      [](const UniformGrid<3>& grid, int num_components,
         const SystemHolder& mesh_coordinate_system, int order) {
        return LagrangeFunctionSpace::FromUniformGrid(
          grid, num_components, CartesianIfNone(mesh_coordinate_system), order);
      },
      py::arg("grid"), py::arg("num_components"),
      py::arg("mesh_coordinate_system") = py::none(), py::arg("order") = 1,
      "Create a LagrangeFunctionSpace from a 3D uniform grid. The coordinate "
      "system defaults to Cartesian.");

  // Configuration for PolynomialReconstructionFunctionSpace MLS evaluation.
  py::class_<MLSOptions>(m, "MLSOptions")
    .def(py::init<>(), "Default MLSOptions")
    .def_readwrite("degree", &MLSOptions::degree)
    .def_readwrite("lambda_reg", &MLSOptions::lambda)
    .def_readwrite("tol", &MLSOptions::tol)
    .def_readwrite("decay_factor", &MLSOptions::decay_factor)
    .def_readwrite("basis", &MLSOptions::basis);

  py::class_<NeighbourhoodOptions>(
    m, "NeighbourhoodOptions",
    "Support search of a PolynomialReconstructionFunctionSpace's geometry")
    .def(py::init<>(), "Default NeighbourhoodOptions")
    .def_readwrite("radius", &NeighbourhoodOptions::radius)
    .def_readwrite("min_supports", &NeighbourhoodOptions::min_supports)
    .def_readwrite("adapt_radius", &NeighbourhoodOptions::adapt_radius);

  py::class_<PolynomialReconstructionFunctionSpace, FunctionSpace,
             std::shared_ptr<PolynomialReconstructionFunctionSpace>>(
    m, "PolynomialReconstructionFunctionSpace")
    .def_static(
      "create",
      [](const PyCoordinateView& coords, MLSOptions options,
         NeighbourhoodOptions neighbourhood) {
        auto values = NumpyInput<Real>::ensure(coords.values);
        return PolynomialReconstructionFunctionSpace::Create(
          CoordinateView<HostMemorySpace>(coords.system,
                                          NumpyToHostView<2, Real>(values)),
          options, std::move(neighbourhood));
      },
      py::arg("coords"), py::arg("options") = MLSOptions{},
      py::arg("neighbourhood") = NeighbourhoodOptions{},
      "Create a PolynomialReconstructionFunctionSpace over the source points "
      "of a CoordinateView")
    .def_static(
      "from_mesh",
      [](Omega_h::Mesh& mesh, int source_entity_dim,
         const SystemHolder& coordinate_system, MLSOptions options,
         NeighbourhoodOptions neighbourhood) {
        return PolynomialReconstructionFunctionSpace::FromMesh(
          mesh, source_entity_dim, CartesianIfNone(coordinate_system), options,
          std::move(neighbourhood));
      },
      py::arg("mesh"), py::arg("source_entity_dim"),
      py::arg("coordinate_system") = py::none(),
      py::arg("options") = MLSOptions{},
      py::arg("neighbourhood") = NeighbourhoodOptions{}, py::keep_alive<0, 1>(),
      "Create a PolynomialReconstructionFunctionSpace from mesh entity "
      "coordinates.");

  m.def(
    "create_uniform_grid_from_mesh",
    [](Omega_h::Mesh& mesh, const std::array<LO, 2>& divisions) {
      return CreateUniformGridFromMesh<2>(mesh, divisions);
    },
    py::arg("mesh"), py::arg("divisions"),
    "Create a 2D uniform grid from an Omega_h mesh");

  m.def(
    "create_uniform_grid_from_mesh",
    [](Omega_h::Mesh& mesh, const std::array<LO, 3>& divisions) {
      return CreateUniformGridFromMesh<3>(mesh, divisions);
    },
    py::arg("mesh"), py::arg("divisions"),
    "Create a 3D uniform grid from an Omega_h mesh");

  m.def(
    "create_uniform_grid_binary_field",
    [](Omega_h::Mesh& mesh, const std::array<LO, 2>& divisions) {
      return CreateUniformGridBinaryField<2>(mesh, divisions).second;
    },
    py::arg("mesh"), py::arg("divisions"),
    "Create a 2D vertex mask field indicating inside/outside mesh");
}

} // namespace pcms
