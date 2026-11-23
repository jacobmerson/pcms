# Fusion-IO 2

Fields of fusion simulation codes (M3D-C1, geqdsk, GATO, MARS, GPEC through
Fusion-IO 1.x, NIMROD through nimeval) as PCMS functions:
`include/fusionio/fusionio.hpp` is the API. This page maps Fusion-IO 1.x's
utilities to it.

## Components

Each component is a CMake option; one whose prerequisite is missing is a
configuration error naming it. `find_package(fusionio2 COMPONENTS ...)` accepts
the names in the last column and fails for one that was not built.

| Option | Default | Needs | Component |
| --- | --- | --- | --- |
| `FUSIONIO_ENABLE_WRAPPED` | on | Fusion-IO 1.x (CMake package `fusionio`) | `Wrapped` |
| `FUSIONIO_ENABLE_NATIVE` | on with `WRAPPED` and Omega_h in PCMS | `WRAPPED` (1.x reads the M3D-C1 files), PCMS with `PCMS_ENABLE_OMEGA_H` | `Native` |
| `FUSIONIO_ENABLE_NIMROD` | off | NIMROD's `nimeval` library (`NIMROD_DIR`) | `Nimrod` |
| `FUSIONIO_ENABLE_GNET` | off | PCMS with `PCMS_ENABLE_POLYNOMIAL_RECONSTRUCTION`, HDF5 | `Gnet` |
| `FUSIONIO_ENABLE_BOUT` | off | netCDF (netcdf-c, `NETCDF_DIR`) | `Bout` |

Without `WRAPPED`, `Open` throws and `fio_trace` is not built; nothing else
needs Fusion-IO 1.x.

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

## Sampling markers

`include/fusionio/sampling.hpp`: `SampleMarkers(f, domain, options)` draws
candidates in a box (`Domain`, one `Range` per coordinate of the function's
coordinate system, drawn uniformly or, for a radial axis, as `sqrt(u)`) and
keeps those where the scalar function `f` is at least `options.threshold`,
until `options.samples` are kept. The result is `Markers::rows`, a device
array `samples x (dimension + 1)` of coordinates and value.

Memory is one batch of candidates and values (`options.batch`) plus the
markers. Each round fills the candidates in place (a function of the seed and
the index of the candidate alone, so any number of threads or devices gives
the same markers), locates and evaluates them with one `pcms::QueryPoints`
(`FILL`, so points outside the function's domain are rejected), and writes the
accepted ones at the slots of a deterministic exclusive scan, clipping the last
round at `samples`. The overload with a `MarkerSink` hands each round's markers
to the caller instead (`gnet2gtc --stream` writes them to the file), so memory
does not grow with `samples`.

## GNET

With `FUSIONIO_ENABLE_GNET`, `include/fusionio/gnet.hpp`:

- `OpenGnet(path, options)` reads a GNET HDF5 file (the layout of
  `distribution5D` of the GNET output: `dimensions/{mxrho, mxzeta, mxenergy,
  mxpitch}`, `range/{engmax_eV, zetamax_rad}`, `data/f5d` of (cell, energy,
  pitch) and `data/volume3d` of (cell)) and returns the number density
  `f5d / volume3d` of the bin centres as a function of (rho, theta, zeta,
  energy, pitch): polynomial reconstruction (`degree` 1, `lambda` 1e-4, no
  weights, as GNET's tool) on a point cloud in the parameter system
  `"gnet_phase"`, whose supports are GNET's analytic bin search (the bins
  below and above the point in each coordinate, `floor(basis size^(1/5)) +
  1` of each; theta and zeta bins wrap).
- `GnetOptions::periodic` (default on) declares theta (2 pi) and zeta (the
  file's `zetamax_rad`) periodic, so the supports across a seam are fit at
  their images nearest the point. Off, they are fit at their raw coordinates,
  as the original tool does; the two differ only for points whose supports
  wrap a bin: above the centre of the last poloidal bin of the radial bins
  around the point and of the last toroidal bin.
- `GTCDomain()` is GTC's box (rho [0, 1] drawn as `sqrt`, theta [0, 2 pi],
  zeta [0, 2 pi / 5], energy [2000, 200000] eV, pitch [-1, 1]) and
  `GTCMarkerWriter` / `WriteGTCMarkers` write `/gtc_n6d` (N x 6).

`gnet2gtc` (built with the component) maps a file to GTC markers:

| Option | Default | Description |
| --- | --- | --- |
| `--input <file>` | required | GNET HDF5 file |
| `--output <file>` | required | GTC output HDF5 file (overwritten) |
| `--samples <n>` | 30000000 | number of GTC markers (max 400000000) |
| `--batch <n>` | 1000000 | candidate points per sampling round |
| `--seed <n>` | 14000 | random seed |
| `--degree <n>` | 1 | polynomial degree |
| `--lambda <x>` | 1e-4 | regularization |
| `--threshold-scale <x>` | 1e10 | threshold = scale / max(`volume3d`) |
| `--no-periodic` | | fit theta and zeta at their raw coordinates |
| `--stream` | | write each round's markers as accepted |
| `--help` | | show the options |

Differences from the original tool (`tools/gnet` of PCMS's `gnet-gtc-coupling`
branch): the candidates are a function of (seed, candidate index), not of a
Kokkos random pool, so the markers differ from the original's for the same
seed but do not depend on the number of threads; the markers are in the order
of the candidates; periodic theta and zeta by default.

## BOUT++

`include/fusionio/bout.hpp` (always built; no netCDF):

- `BoutGrid` is one block of a BOUT++ grid as BOUT++ holds it: sizes with
  guard cells, `Rxy`, `Zxy`, `zShift`, `dx`, `dy` and `g^ij` as `[nx][ny]`
  host views, `ShiftAngle` `[nx]`, `dz`, `zperiod` and the separatrix
  indices. Only grids of closed field lines and one serial block (offsets 0)
  are supported so far.
- `Build(grid)` returns the geometry and the space of its fields: a uniform
  grid in the global index coordinates (x, y, z) of the parameter system
  `"bout_index"`, y identified with the twist-shift of z by `ShiftAngle / dz`
  and z periodic; the position (R, phi, Z) = (`Rxy`, z dz + `zShift`, `Zxy`)
  in `CylindricalRThetaZ(2 pi / zperiod)` and the metric `g^ij` in index
  components, multilinear; the space is multilinear on the embedded geometry.
  The y = ny and z = nz planes are DOF holders: the twist-shifted y = 0 plane
  and the z = 0 plane.
- `InteriorOffsets(space, grid)` maps the DOF holders to BOUT++'s `[nx][ny][nz]`
  field array (the y = ny plane reads the first upper y guard plane, which
  BOUT++'s communication fills with the twist-shifted values); `CopyIn` and
  `CopyOut` copy a field through it.

With `FUSIONIO_ENABLE_BOUT`, `ReadBoutGrid(path, nz, zperiod)` reads a grid
file (nz and zperiod are BOUT.inp's `MZ` and `zperiod`; `MXG` from the file,
default 2) and computes `g^ij` of BOUT++'s field-aligned tokamak coordinates
from `Rxy`, `Bpxy`, `Btxy`, `Bxy`, `hthe` and `sinty`. `OpenBout` returns it
as a `Source` with the equilibrium profiles of the file, constant in z:

| Quantity | Variable | Units |
| --- | --- | --- |
| `TotalPressure` | `pressure` | Pa |
| `Density` (main ions) | `Ni0` | the file's (1e20 m^-3 in BOUT++'s tokamak grids), not set |
| `Temperature` (main ions, electrons) | `Ti0`, `Te0` | eV |

`bout-grid <grid.nc> <nz> <zperiod>` (built with the component) prints the
volume of a grid by quadrature and as BOUT++'s sum of J dx dy dz, the integral
of the pressure and the position and pressure at a point. The tests read the
CBM18 grid of BOUT++'s elm-pb example from the PCMS test data
(`FUSIONIO_TEST_BOUT_GRID`, `bout/cbm18/cbm18_grid.nc`; LGPL-3.0).
