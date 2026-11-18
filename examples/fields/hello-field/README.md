# hello-field

A field on a uniform grid, evaluated at points. This is the smallest PCMS
program that does something, and the starting point of the other examples: it
creates a geometry (a uniform grid), a function space on it (linear Lagrange),
a field with values from a function, and evaluates the field at points, first
with `QueryPoints`, which keeps the located points for several fields and for
the next call, then with the one-shot `Evaluate`. It ends with points outside
the grid and the three things that can be done about them: an error, a fill
value, or extension of the nearest cell. It needs only Kokkos and PCMS, so CI
also builds it against an installed PCMS as the smoke test of the
installation.

Asserted: the interpolation error of a linear function is at rounding level
(1e-12) with `QueryPoints` and with `Evaluate`, the points outside the grid
are reported as missing and get the fill value, and with `NEAREST_BOUNDARY` the
linear function extends exactly.

## Public API used

- `pcms::UniformGrid`, `backend::uniform_grid::CreateUniformGridGeometry`
  (Python: `UniformGrid2D`, `create_uniform_grid_geometry`)
- `csys::Cartesian` (default in Python)
- `CreateLagrangeSpace`, `FunctionSpace::CreateFunction`,
  `FieldLayout::GetDOFHolderCoordinates`, `Field::SetDOFHolderDataHost`
  (`create_lagrange_space`, `create_function`, `set_dof_holder_data`)
- `CoordinateView`, `ValueView` (Python: `CoordinateView`, NumPy arrays)
- `QueryPoints` (`Locate`, `Evaluate`, `Bound`), `Evaluate`
- `OutOfBoundsPolicy`, `OutOfBoundsMode::{ERROR, FILL, NEAREST_BOUNDARY}`,
  `BoundSource::FoundPoints`, `BoundSource::MissingPoints`
- `pcms_error` (Python: `RuntimeError`)

## Run

```console
$ ctest -R example.fields.hello-field
$ build/examples/fields/hello-field/fields_hello_field
$ PYTHONPATH=build/src/pcms/pythonapi python3 examples/fields/hello-field/main.py
```

## What to look at

The output lists the checks. Look at the end, where the same four points
(two of them outside the unit square) go through the three policies:

```
ERROR policy: Bind: 2 query point(s) are outside the source's domain
FILL policy: 2 points found, 2 missing: #1 #3
```

The error message says how many points failed; with `FILL` the bound source
tells which (`MissingPoints`), and the rows of those points in the output hold
the fill value.
