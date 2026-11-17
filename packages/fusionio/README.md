# Fusion-IO 2

Fields of fusion simulation codes (M3D-C1, geqdsk, GATO, MARS, GPEC through
Fusion-IO 1.x, NIMROD through nimeval) as PCMS functions:
`include/fusionio/fusionio.hpp` is the API. This page maps Fusion-IO 1.x's
utilities to it.

## Fusion-IO 1.x utilities

The utilities of `fusion_io/isosurface.h` are free functions in
`include/fusionio/utilities.hpp`. They take a function of a source
(`Source::CreateFunction`, its space supporting `Gradient` where the search
uses derivatives) in place of a `fio_field*`, spans of `Point`s ((R, phi, Z))
in place of `double*` triples and `double**` path arrays, and return a
`SearchResult` (`Success`, `OutOfBounds`, `Diverged`) per point in place of
`FIO_SUCCESS`, `FIO_OUT_OF_BOUNDS`, `FIO_DIVERGED`. 1.x's search hints have
no counterpart: the points are located in batches. They keep 1.x's algorithm
and arguments but advance all points (or paths) of a call together, one
batched evaluation per iteration where 1.x evaluates point by point.

| 1.x | Fusion-IO 2 |
| --- | --- |
| `fio_field::get_bounding_box(min, max)` | `Source::BoundingBox()` |
| `fio_find_max(f, &val, x, tol, max_step, dim, norm)` | `FindMax(f, val, x, tol, max_step, dim, norm)`: `val` and `x` one entry per point |
