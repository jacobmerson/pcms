# moving-points

Field lines traced by a batch of tracers. Particle pushers, field-line
tracers and probes evaluate a field at points that change at every step, and
locating a point on a mesh is the expensive part of evaluating there. The
example integrates the field lines of a rotation field, stored as a P1 vector
field on a triangle mesh, with RK4: each step evaluates the field at four sets
of points, one per stage, and each evaluation is a `QueryPoints::Locate`
followed by an `Evaluate`, which reuses the storage of the previous call. The
tracers on small circles stay in the domain; those on a large circle leave it,
where the field is not defined, and the `FILL` policy turns that into missing
points that the example reads from the bound source and uses to stop the
tracers.

Asserted: the first stage reports as missing exactly the tracers that start
outside the unit square; after one revolution (200 steps) the tracers inside
are back at their start within 1e-6 and keep their radius within 1e-6 (the
error of RK4 is 2e-8); every tracer on the large circle is lost.

## Public API used

- `backend::omega_h::CreateGeometry`, `CreateLagrangeSpace` with two
  components, `values::Vector`, `FunctionSpace::CreateFunction`
- `QueryPoints` (`Locate`, `Evaluate`, `Bound`)
- `OutOfBoundsPolicy` with `OutOfBoundsMode::FILL`, `BoundSource::MissingPoints`
- `CoordinateView`, `ValueView` (the vector basis of the field)

## Run

```console
$ ctest -R example.fields.moving-points
$ build/examples/fields/moving-points/fields_moving_points
$ PYTHONPATH=build/src/pcms/pythonapi python3 examples/fields/moving-points/main.py
```

## What to look at

The `Tracers::Velocity` function (`velocity` in Python) is the whole use of
PCMS: `Locate`, `Evaluate`, and `MissingPoints`. Everything else is the
integrator. The output reports the 8 of 36 tracers that start outside the
square and the closure error after the revolution. To trace many more
tracers, or in 3D, change the mesh and the starting points only.

A lost tracer here stays in the batch and gets the fill value (velocity 0);
for large batches, remove lost tracers from the batch instead so that the
located points shrink and the work with them.
