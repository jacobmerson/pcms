# coordinate-systems

Fields in cylindrical coordinates, queried in Cartesian. PCMS tags coordinates
with a coordinate system and vector values with the basis of their
components. The example has a box mesh whose coordinates are (r, theta, z) and
evaluates fields on it at Cartesian (x, y, z) points: a `CoordinateMap`
(`CartesianToCylindrical`) maps the query points into the geometry's system
and re-expresses the components of vectors in the basis of the query points.
Shown: the map on its own (mapped points, `e_theta` rotated to Cartesian),
`QueryPoints` with a map for a scalar and for a vector field, the same through
an `Interpolator` onto a Cartesian mesh with the points bound once by
`Geometry::Bind`, and a field whose points are cylindrical but whose
components are Cartesian (the map then only maps the points). It replaces the
earlier `coordinate-transformation-example`.

Asserted against the analytic transform: the mapped points and the rotated
`e_theta` (1e-12); a scalar `1 + 2r + 3z` at Cartesian points (1e-12); the
vector `r e_r + z e_z` comes out as `(x, y, z)` (1e-12), also through the
`Interpolator`; an output in the wrong basis is refused; the borrowed-basis
field is `(x, y, z)` up to the P1 interpolation error (2e-2).

## Public API used

- `csys::Cartesian`, `csys::CylindricalRThetaZ`
- `CoordinateMap`: `CartesianToCylindrical`, `Bind`, `BoundCoordinateMap::
  {MappedPoints, TransformValues}` (Python: `CartesianToCylindrical().bind`)
- `ValueBasis`, `ComponentScaling`, `values::Vector`,
  `FunctionSpace::CreateFunction` with a variance, a scaling and a system
- `QueryPoints` with a map, `Geometry::Bind` with a policy and a map,
  `Interpolator` from a bound source
- `backend::omega_h::CreateGeometry` with a coordinate system

## Run

```console
$ ctest -R example.fields.coordinate-systems
$ build/examples/fields/coordinate-systems/fields_coordinate_systems
$ PYTHONPATH=build/src/pcms/pythonapi python3 examples/fields/coordinate-systems/main.py
```

## What to look at

The first block prints `e_theta` in Cartesian components at three points: a
vector's components depend on where it is and on the basis, a scalar's do not.
Then the checks: every one of them is an exact result except the last, whose
error is the interpolation error of a field that is not linear in the mesh's
coordinates.

## Not covered (gaps of the API)

- An axisymmetric field stored on a 2D `CylindricalRZ` geometry (the usual
  form of a tokamak equilibrium) cannot be evaluated at 3D Cartesian points
  with `CartesianToCylindrical`: `Geometry::Bind` requires the map to end in
  the geometry's system, `cylindrical_rz`, and no map goes from Cartesian
  3D to (R, Z) (dropping the angle). This example therefore uses a
  3D (r, theta, z) mesh. `PoloidalPlaneEmbedding` goes the other way (a 2D
  target queries a 3D source) and is left to the transfer track; it has no
  Python binding yet.
