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
| `fio_find_val(f, val, x, tol, max_step, dim, norm)` | `FindVal(f, val, x, tol, max_step, dim, norm)`: `x` one entry per point |
| `f->find_val_on_line(f0, x0, x1, x, h, tol)` | `FindValOnLine(f, f0, x0, x1, x, tol)`: one segment and guess per point |
| `fio_gridify_loop(m0, path0, axis, m, path, theta, param)` | `GridifyLoop(path0, axis, path, theta, param)`: `param` a `LoopParameter`, m = path.size() |
| `fio_gridify_surface(m0, path0, axis, nphi, ntheta, path, phi, theta)` | `GridifySurface(path0, axis, nphi, ntheta, path, phi, theta)`: `Diverged` if `path0` does not hold `nphi` loops (1.x prints and succeeds) |
| `fio_isosurface_2d(f, val, guess, axis, norm, dd, dl, tol, max_step, &n, &path)` | `Isosurface2D(f, val, guess, norm, dd, dl, tol, max_step)`: one `Path` per guess; `axis` (unused by 1.x) dropped; no dump file when a contour does not close |
| `fio_isosurface(f, val, guess, enclose, dl, tol, max_step, nphi, &n, &path)` | `Isosurface(f, val, guess, enclose, dl, tol, max_step, nphi)`: one `Path` per guess |
| `fio_geom_isosurface(f, val, guess, axis, tol, max_step, nphi, ntheta, &path)` | `GeomIsosurface(f, val, guess, axis, tol, max_step, nphi, ntheta)`: the planes together, each starting from the guess's distance to the axis (1.x: the previous plane's last) |
| `fio_gridded_isosurface(f, val, guess, axis, dl_tor, dl_pol, tol, max_step, nphi, ntheta, phi, theta, path, label)` | `GriddedIsosurface(f, val, guess, axis, dl_pol, tol, max_step, nphi, ntheta)` returns a `GriddedSurface` (phi, theta, axis per plane, path): the planes together, each plane's axis and contour searched from the arguments (1.x: from the previous plane's); `dl_tor` (unused by 1.x) and the `label` files dropped |
| `fio_find_rho_hybrid(f, target, rho, ct, st, phi, axis, tol)` | `FindRhoHybrid(f, target, rho, ct, st, phi, axis, tol)`: one ray per entry of `rho`, `ct`, `st` |
| `fio_gridded_isosurface_hybrid(f, val, guess, axis, dl_tor, dl_pol, tol, max_step, nphi, ntheta, phi, theta, path, label)` | `GriddedIsosurfaceHybrid(f, val, guess, axis, tol, max_step, nphi, ntheta)` returns a `GriddedSurface`: the planes one after another as in 1.x, the angles of a plane together, each starting from the radii the previous plane found (1.x: also the one just found at the previous angle); the unused `dl_tor`, `dl_pol` and the `label` file dropped |
| `fio_q_at_plane(f, n, x, &q, bpol, bout)` | `QAtPlane(b, x, bpol, b_out)`: `nullopt` if a point is outside |
| `fio_surface_average(f, ns, nphi, ntheta, x, jac, fout)` | `SurfaceAverage(f, ns, nphi, ntheta, x, jac)` |
| `fio_surface_average(f, n, x, &a, bpol)` (defined, not declared in `isosurface.h`) | `SurfaceAverage(f, x, bpol)` |
| `fio_isosurface_jacobian(ns, nphi, ntheta, x, jac)` | `IsosurfaceJacobian(ns, nphi, ntheta, x, jac)`: `false` if the Jacobian changes sign |
| `fio_eval_on_path(f, n, x, fout)`, `fio_eval_on_surface(f, n, x, a)` | `pcms::Evaluate(function, points, out)` (`pcms/field/query_points.hpp`), or `pcms::QueryPoints` to evaluate several functions at the same points |
