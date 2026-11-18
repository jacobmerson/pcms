# mesh-fields

Lagrange fields on an unstructured mesh. The geometry is a 2D simplex mesh
from `Omega_h::build_box`; the example creates P0 (one value per element) and
P1 (one per vertex) spaces on it, explains what the layout of a field says (the
DOF holders a process owns, the ghost copies of the ones other processes own,
global ids), compares how the two orders represent a linear function (P1
exactly, P0 only at the centroids), and asks the spaces which differential
operators they support. With MeshFields enabled it also creates a P2 space and
checks that it reproduces a quadratic function.

Asserted: on one process every DOF holder is owned and has a distinct global
id, with one holder per vertex (P1) and per element (P0); P1 evaluates a
linear function to rounding (1e-12); P0 is exact at its own DOF holders (the
centroids) and within `|grad f| h` elsewhere; P2 reproduces a quadratic
function to 1e-10 (with MeshFields; skipped at compile time otherwise) and P1
does not.

## Public API used

- `backend::omega_h::CreateGeometry` (Python: `backend.omega_h.create_geometry`)
- `CreateLagrangeSpace`, `LagrangeOptions`, `LagrangeBackend::{OmegaH, MeshFields}`
- `FunctionSpace::GetLayout`, `Supports`; `DifferentialOperator`
- `FieldLayout::{GetNumDofHolder, GetNumGlobalDofHolder, HasGhosts,
  IsDistributed, GetOwnedHost, GetGidsHost, GetDOFHolderCoordinates}`
  (Python: `get_num_dof_holder`, `get_num_global_dof_holder`, `is_distributed`,
  `get_dof_holder_coordinates`)
- `Evaluate`
- `PCMS_ENABLE_MESHFIELDS` of `pcms/configuration.hpp`

## Run

```console
$ ctest -R example.fields.mesh-fields
$ build/examples/fields/mesh-fields/fields_mesh_fields
$ PYTHONPATH=build/src/pcms/pythonapi python3 examples/fields/mesh-fields/main.py
```

## What to look at

The first lines describe the layouts: how many DOF holders this process owns,
how many exist over all processes, and the first global ids. With more than one
process the owned count drops below the global one and `GetOwnedHost` marks
the ghosts; the checks about ghosts are asserted for one process only. Then the
P0 error at random points, which is bounded by the gradient times the element
size and shows what "one value per element" costs.

## TODO: gradient of a P1 field

The plan for this example included the gradient of a P1 field with
`DifferentialOperator::Gradient`. The public API cannot do that yet: only the
spline space supports `Gradient` (see splines-eqdsk); the Lagrange evaluators
of both backends support `Value` only, and `FunctionSpace::Supports` says so
(the example prints it). When Lagrange spaces support the gradient (piecewise
constant for P1), add it here: evaluate `Gradient` of the P1 field with
`QueryPoints::Evaluate`, and assert the exact gradient of the linear function.
