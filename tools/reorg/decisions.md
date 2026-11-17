# PCMS breaking reorganization: decisions and amendments

This file records the decisions made with the maintainer before and during the
reorganization, including where they deliberately differ from the original task
description (`refactor.md`). Reviewers should read this first.

## Process

| Item | Decision |
|---|---|
| Base | `develop` de26092c, with the geometry work (`Geometry`, `BoundSource`, `QueryPoints`, one concrete `FunctionSpace` with its factories) and the bind and gradient work merged. Phase 0 (the inventory) was done on c33f816c |
| PR #379, #401 | Still open; judged not to conflict materially; not blocking |
| PR #410 (CMake 3.23) | Will not be merged; the first commit of this branch bumps `cmake_minimum_required` to 3.23 (required by `FILE_SET`, already in use) |
| Version | Minor bump: 0.3.0 → 0.4.0 |
| Push / PR | Commits are local; the maintainer pushes. The branch is reviewed as six stacked PRs into the integration branch `reorg/0.4`, which is merged into `develop` once, so downstream codes and open PRs migrate once |
| Verification | The last commit of each stacked PR: Minimal-static and Full-shared build + `ctest` |
| Verification at the tip | Additionally Full-static, `cmake --install`, `examples/external-usage-example` built against the install tree, CI's clang-format check, and a re-run of the generated commits' commands. Bisect only on failure |
| Open branches (#400 dof-access, external-eval-infra, fusion-io, nimrod) | Reorg lands first; those branches migrate afterwards with `tools/migrate-pcms.sh` |
| Old CMake target names (`pcms::core`, ...) | Kept for one release as deprecated `INTERFACE` targets linking `pcms::pcms` |
| Old header paths | No forwarding stubs; `tools/reorg/header-map.txt` + `tools/migrate-pcms.sh` cover them |
| Fortran SWIG wrappers | Not regenerated: they include `"pcms/capi/*.h"`, which keeps its path, and the C API does not change |
| Commit granularity | Deviation from the plan's one-commit-per-sub-library rule, decided at review: the `.h`→`.hpp` renames, the `pcms/`-rooted includes, the moves to `include/` and the install rules are one commit each (each mechanical commit lists its commands in order; every one reproduces its diff). Fixes to the reorg scripts are folded into the commits that add the scripts, and the decision/inventory updates into the two docs commits. Backend file moves stay one commit per backend. All namespace moves (backends and sub-libraries) are one generated commit, made by `tools/reorg/refactor/` (a Clang LibTooling tool) from `namespaces.spec`. |
| Formatting | Every commit is formatted with clang-format 18.1.3, the version CI uses; mechanical commits include the `clang-format -i` call in their command. There is no whole-tree formatting commit |

### Baseline limitation

On develop the coupler is built unconditionally (`find_package(redev REQUIRED)`,
`find_package(ADIOS2 REQUIRED)` at top level, `smoke_test` links `pcms::core`).
The "Minimal: no coupler" configuration therefore cannot be built until Phase 2.
Until then "Minimal" means: Omega_h only; meshfields, PETSc, MFEM, XGC, Python,
C, Fortran and transient off; static.

## Public API (amendment: the task originally excluded API redesign)

The change is limited to *how objects are constructed*; semantics of the
existing interfaces do not change.

### Principles

- Public headers contain **interfaces** and **free-function factories** only.
  All concrete implementation classes are private (stay in `src/`, not
  installed), except the geometry kinds (`OmegaHGeometry`,
  `UniformGridGeometry<Dim>`, `PointCloudGeometry`), which are public types so
  that their factories can return them. In-tree targets and tests may still
  include private headers.
- Function spaces are created **by space type, on a `Geometry`**, which owns
  (layouts hold it, `GetGeometry()`) the coordinate system and the point search. Mesh sources
  produce geometries.
- Codes with their own representation (XGC, MFEM in place, later NIMROD,
  Fusion-IO, BOUT++) get **per-code factories** returning the same interfaces.
- Only backend headers (`include/pcms/backends/<name>/`) may include or name
  backend-library types.
- Factories return `std::shared_ptr<Interface>`; transfer factories return
  `std::unique_ptr<TransferOperator<T>>`.

Rationale for interface-only (vs. keeping concrete classes public with factories
moved out): an audit of every non-override public method on concrete classes
found no external user that needs one. The only uses outside the owning
sub-library are the C API, the Python bindings and tests (XGC
`GetXGCLayout`/`SetLayoutName`, MFEM `OverlapMaskFromAttribute`,
`OmegaHDiscretization::GetMesh`); these become factory arguments or free
functions. Making a class public later is non-breaking; hiding it later is not.

### Public interfaces and types (`namespace pcms`)

Interfaces: `Geometry`, `BoundSource` (the located-points handle), `FieldLayout`,
`FunctionSpace`, `Field<T>` / `Function<T>`, `TransferOperator<T>`,
`CoordinateSystem` / `CoordinateMap` (with `BoundCoordinateMap` and the three
maps), coupler types, `Timestepper`. 

Backend-free types users construct directly: `Interpolator<T>` (also from a
source, a target and a bound source), `Copy<T>`, `QueryPoints` (and
`QueryPoints::Bound()`), coordinate systems and maps, `FixedTimestepper`, and
the geometry kinds `backend::omega_h::OmegaHGeometry`,
`backend::uniform_grid::UniformGridGeometry<Dim>` and
`backend::point_cloud::PointCloudGeometry` (the factories return them).

The types that only code implementing a source, a geometry kind, a layout or
field data needs are the **implementer API**, and the helpers the public
templates call are in `pcms::detail`: see "The public surface (0.4 trim)".

### Factories

```cpp
// Geometries (core)
namespace pcms::backend::uniform_grid {
std::shared_ptr<const UniformGridGeometry<Dim>> CreateUniformGridGeometry(UniformGrid<Dim>, csys);
}
namespace pcms::backend::point_cloud {
std::shared_ptr<const PointCloudGeometry>       CreatePointCloudGeometry(coords | layout, NeighbourhoodOptions = {});
}
std::shared_ptr<const FieldLayout> CreateEmptyLayout(csys, ncomp);   // for out-of-tree evaluation-only spaces

// Geometries (backend)
namespace pcms::backend::omega_h {
std::shared_ptr<const OmegaHGeometry>
CreateGeometry(::Omega_h::Mesh& mesh, csys, OmegaHGeometryOptions = {});  // borrows mesh; options: search, global id tag, owned mask
std::shared_ptr<const point_cloud::PointCloudGeometry>
CreatePointCloudGeometry(std::shared_ptr<const OmegaHGeometry> geometry, int entity_dim, NeighbourhoodOptions = {});  // sites: vertices or element centroids
// the mesh: OmegaHGeometry::GetMesh()
}

// Function spaces, by type (core)
std::shared_ptr<const FunctionSpace> CreateLagrangeSpace(geometry, order, ncomp, LagrangeOptions = {});
        // LagrangeOptions: LagrangeBackend (MeshFields or Omega_h), layout_name
std::shared_ptr<const FunctionSpace> CreateSplineSpace(geometry);
std::shared_ptr<const FunctionSpace> CreatePolynomialReconstructionSpace(geometry, MLSOptions = {});

// Adopted application spaces (backend headers)
namespace pcms::backend::xgc  { CreateFunctionSpace(..., layout_name); CreateField<T>(space, name, data, plane_comm); }
namespace pcms::backend::mfem { CreateFunctionSpace(::mfem::ParFiniteElementSpace&, ..., layout_name);
                                OverlapMaskFromAttribute(::mfem::ParMesh&, int); }

// Transfers (core header include/pcms/transfer/, always declared;
// throw "not available in this build" if the implementing backend is disabled)
std::unique_ptr<TransferOperator<Real>> CreateConservativeProjection(src_space, tgt_space, opts = {});
std::unique_ptr<TransferOperator<Real>> CreateControlVariateProjection(src_space, tgt_space, opts = {});
```

The space factories dispatch on the concrete geometry kind (Omega_h, uniform
grid, point cloud); the kinds are public classes (factories return them). A layout holds its geometry (`GetGeometry()`),
so there are no discretization factories (`Discretization` is gone).
There is one concrete `FunctionSpace`.

Unsupported combinations (e.g. a Lagrange space on a point-cloud geometry)
throw `pcms_error`.

**Factory return types.** A factory returns the most specific type it knows statically; geometry
pointers are const. The kind-named factories return the kind: `backend::omega_h::CreateGeometry`
`std::shared_ptr<const OmegaHGeometry>`, `backend::uniform_grid::CreateUniformGridGeometry`
`std::shared_ptr<const UniformGridGeometry<Dim>>`, `backend::point_cloud::CreatePointCloudGeometry`
`std::shared_ptr<const PointCloudGeometry>`, the coordinate systems' `Create` a pointer to their own
type; each converts to `std::shared_ptr<const Geometry>` / `std::shared_ptr<const CoordinateSystem>`. The space
factories return `std::shared_ptr<const FunctionSpace>`, a layout's name is given at construction
(`LagrangeOptions::layout_name`, the XGC and MFEM factories' name argument; there is no `SetName`). The C API and
the Python bindings call the factories (no `make_shared`; Python holds immutable objects through `Immutable<T>`
handles, and `backend.omega_h.create_geometry` returns the handle of the Omega_h kind). The Python
`backend.omega_h.create_point_cloud_geometry(geometry, entity_dim, neighbourhood)` takes the Omega_h geometry
whose vertices (entity_dim 0) or element centroids are the sites.

### The public surface (0.4 trim)

Three tiers, each header documenting its own:

1. **Public API**, covered by the stability promise: the interfaces and types
   listed above, the factories, and what users call on them. This includes
   `Geometry::Bind`, `BoundSource` (the handle of located points),
   `Interpolator(source, target, bound)`, `QueryPoints` and
   `QueryPoints::Bound()`, `CoordinateMap`, `BoundCoordinateMap` and the three
   maps.
2. **Implementer API**: installed, named as in 0.3 where it existed, for code that
   implements a source, a geometry kind, a layout or field data. It is not
   covered by the stability promise; each declaration carries the doc line
   "Implementer API: not covered by the public-API stability promise."
3. **`pcms::detail`**: helpers that public templates and inline functions call
   and that users have no use for. Not API. They stay in their headers.

Implementer API:

| Name | Header | For |
|---|---|---|
| `PointEvaluator<T>`, `PointEvaluatorFactory`, `TypedPointEvaluatorFactory` | `field/point_evaluator.hpp`, `field/point_evaluator_factory.hpp` | sources implemented outside PCMS (`BindImpl`, `CreatePointEvaluatorImpl`) |
| `TransformedPointEvaluator<T>` | `field/detail/transformed_point_evaluator.hpp` | the templates of `point_evaluator_factory.hpp` |
| `BindRequest`, `BoundSource::RebindKey`, `Geometry::RebindKey` | `geometry/bound_source.hpp`, `geometry/geometry.hpp` | geometry kinds (`BindImpl`, `RebindImpl`) |
| `DifferentialView` | `geometry/geometry.hpp` | geometry kinds (`Differential`) |
| `SupportResults` | `geometry/supports.hpp` | geometry kinds (`Neighbourhood`) and MLS |
| the members of the geometry kinds that users do not call (`FindElementSeeds`, `Differential`, `BindImpl`, `TakesHints`), their bound sources (`OmegaHMeshBinding`, `UniformGridBinding`, `MLSBinding`, `MLSPointEvaluator`), the point searches, the MLS helpers and `PointCloudLayout` | `backends/*/geometry.hpp` and the headers they include | geometry kinds and layouts implemented outside PCMS, and PCMS's own backends |
| `OwnerHints`, `DOFHolderOwners` | `geometry/bound_source.hpp`, `field/field_layout.hpp` | geometry kinds and layouts naming the entities of a bind |
| `Apply`, `Gradient`, `Divergence`, `Curl` | `field/gradient_source.hpp` | wrappers of a `PointEvaluatorFactory` |
| `FieldData<T>`, `DOFAccess<T>`, `SimpleFieldData<T>`, `FieldFactory` | `field/field_data.hpp`, `field/simple_field_data.hpp`, `field/field_factory.hpp` | backends and sources with their own field storage |
| `OwnedLayoutData`, `CreateEmptyLayout` | `field/field_layout.hpp`, `field/function_spaces.hpp` | layouts; evaluation-only spaces outside PCMS |
| the `FunctionSpace` constructor | `field/function_space.hpp` | backends (users call the `Create*` factories) |
| `ExchangePlan`, `FieldExchangePlanner`, `GenericFieldExchangePlanner`, `FieldLayoutCommunicator`, `FieldCommunicator<T>`, `FieldSerializer<T>`, `GlobalCommunicator<T>` | `coupler/detail/*.hpp` | the coupler and coupler adapters |

Moved to `pcms::detail` (same headers; every use in the tree is qualified):

| Header | Names |
|---|---|
| `geometry/geometry.hpp` | `CheckEntityIds`, `CheckNeighbourhoodArguments` |
| `geometry/bound_source.hpp` | `ReserveRows` |
| `field/point_evaluator.hpp` | `FillRows` |
| `field/field_layout.hpp` | `BuildOwnedClassificationHost`, `OwnedEntOffsets`, `GetMeshEntityDim` |
| `field/value_view.hpp` | `CheckWrittenValueBasis`, `ValidateValueBasis`, `ValidateValueSemantics`, `DerivedComponentScaling`, `ValueTypeOfRank`, `GradientBasis`, `SameValueBasis` (`values::Of` is removed: `VarianceSignature{...}` does the same) |
| `transfer/interpolator.hpp`, `transfer/copy.hpp` | `SameDOFHolderSites`, `CheckCopyCompatible` |
| `coupler/coupler_types.hpp` | `ToRedev` |
| `utility/assert.hpp` | `Pcms_Assert_Fail` |
| `utility/coordinate_system.hpp` | `ResolveCoordinateSystem`, `SameCoordinateSystem`, `HasIdentityMetric` |
| `utility/uniform_grid.hpp` | `DivisionsPerAxisForMesh` |
| `utility/types.hpp` | `TypeEnumFromType`, `type_identity`, `dependent_always_false` |
| `utility/arrays.hpp` | everything except `View`, `Rank1View`, `Rank2View`, `GlobalIDView`, `MakeConstRank2View` and `DeepCopyMismatchLayouts`: `Rank3View`, `Rank4View`, `make_mdspan`, `make_array_view`, `make_const_array_view`, `MakeRank1View`, `MakeRank2View`, `FlattenToRank1View`, `ConvertMismatchLayoutView2D`, `iota_view`, the `Copy*` and `ScatterRows` helpers |
| `utility/memory_spaces.hpp` | `ForEach`, `IsSmallHostBatch`, `SmallHostBatchSize` |
| `backends/xgc/reverse_classification.hpp` | `DimID` and its `std::hash` |

`ForEach` and `IsSmallHostBatch` (the QueryPoints work) were public in the
rehearsal; they are implementer-facing templates, so they are `pcms::detail`
and documented as implementer API. The types the public signatures name stay:
`ComponentScaling`, `VarianceSignature`, `View`, `Rank1View`, `Rank2View`,
`GlobalIDView`.

### Coupler

`pcms::Mode` (today `using Mode = redev::Mode`) becomes a real enum
`{Deferred, Synchronous}`, converted to `redev::Mode` inside the coupler. Public
signatures use `pcms::Mode`; redev no longer appears in public headers.

### Python

The Python API is changed to match the C++ shape in this release
(`pcms.backend.omega_h.create_geometry(mesh)`,
`pcms.create_lagrange_space(geometry, ...)`); the old names are removed.

## Backends and build layout (amendment to Phase 3)

- Namespace `pcms::backend::<name>` (instead of `pcms::<name>_backend`); bridges
  `pcms::backend::omega_h_meshfields`. Inside `pcms::backend`, `mfem::` would
  find `pcms::backend::mfem`, so backend code writes `::mfem::`. Other library
  namespaces (Kokkos, Omega_h, ...) are not shadowed and are written without
  a leading `::` (amended at review: qualifying them too changed about 1000
  lines and caused most conflicts with open PRs). Nothing enforces this; an
  unqualified `mfem::` that resolves to `pcms::backend::mfem` fails to
  compile, so the mistake cannot go unnoticed.
- Backends are **directories, namespaces and headers**, not libraries:
  sources in `src/pcms/backends/<name>/`, public headers in
  `include/pcms/backends/<name>/`, compiled into the core target that uses them
  (`pcms::field`, `pcms::transfer`, `pcms::coupler`) behind
  `PCMS_ENABLE_<BE>` / `PCMS_HAS_<BE>`. A bridge is compiled when both backends
  are enabled.
- No `pcms::<be>` targets and no `pcms_backend_part()` helper. Reason: separate
  backend libraries only pay off for static-link dependency hygiene and
  `COMPONENTS` support, neither of which is in scope; they would force a
  dispatch layer above `pcms::field`.
- Downstream links **`pcms::pcms`**, the documented, stable entry point. Splitting
  backends into libraries later (e.g. when Omega_h becomes optional) is then an
  internal change.

## Deferred (follow-ups, non-breaking)

- `ReadDiscretization(comm, path, opts)` with file-type dispatch (needs a
  partitioning option; owns the mesh it reads). Further mesh sources (MFEM
  mesh → shared infrastructure, gmsh, BOUT++).
- **Multi-field sources** (EQDSK, NIMROD, Fusion-IO, VMEC): a `FieldSource`
  layer on top of spaces (`Quantities()`, `CreateFunction(quantity)`,
  `QuantityInfo`, `OpenEQDSK`, `OpenFieldSource(path)`), designed separately.
  In this reorg `MakeEQDSKField` only changes its struct member to
  `std::shared_ptr<FunctionSpace>`.
- Batch evaluation of several quantities in one call; EQDSK flux functions
  (needs a composition evaluator).
- Making Omega_h optional for core internals.

## Include guards (amendment to Phase 1.4)

`#pragma once` is not standard C++ and is **not used**. Every header keeps an
`#ifndef`/`#define`/`#endif` guard. Instead of switching to `#pragma once`,
guard names are normalized to a path-derived scheme
(`pcms/field/field.hpp` → `PCMS_FIELD_FIELD_HPP`) in one mechanical pass
**after Phase 5**, when header paths are final. Before that, only headers that
had no guard (or `#pragma once`) are fixed.

## Phase 3 notes (backends)

- Layout: `src/pcms/backends/<name>/` with a `CMakeLists.txt` that adds the
  backend's sources and headers to the core target that uses them (see
  "Backends and build layout"); namespaces `pcms::backend::<name>`. Done for
  `xgc`, `mfem`, `point_cloud`, `uniform_grid`, `omega_h` and the bridge
  `omega_h_meshfields`.
- **MeshFields is a bridge.** Every MeshFields file needs the Omega_h adapter
  (`MeshFieldsAdapterLayout` is built from an `OmegaHDiscretization`), so the
  whole MeshFields adapter is in `backends/omega_h_meshfields/`, built only
  when both Omega_h and MeshFields are enabled. There is no `backends/meshfields`.
- **`function_space/lagrange` stays in `field`.** The plan listed it under the
  Omega_h adapter, but it is the generic Lagrange space (it also serves uniform
  grids) behind `CreateLagrangeSpace`.
- **Omega_h-based transfer algorithms stay in `transfer`** (conservative and
  control-variate projection, the integrators, `mesh_intersection`); the plan
  only listed field/localization parts for the adapter. They are private.
- `point_cloud` and `uniform_grid` have no external dependency and are used by
  other backends (the MFEM layout builds a `PointCloudDiscretization`); this is
  not treated as a bridge.
- `SupportResults`, the MLS supports `Geometry::Neighbourhood` returns, is
  declared in `geometry/supports.hpp` (it was in `adj_search.hpp`). It held
  Omega_h arrays; Phase 4 makes its members Kokkos views in
  `DeviceMemorySpace` (`supports_ptr`, `supports_idx`:
  `Kokkos::View<const LO*>`; `radii2`: `Kokkos::View<Real*>`), converted at the
  Omega_h backend boundary, so no core public header includes Omega_h.
- **Dependency hygiene (0.4 additions).** A consumer of PCMS core needs only
  Kokkos and PCMS at compile and link time: `utility/assert.hpp` no longer
  includes `<mpi.h>` (the one-argument `PCMS_ALWAYS_ASSERT` reports the rank
  from `assert.cpp`; the forms that take a communicator expand to
  `MPI_Comm_rank` at the call site); Omega_h is a private dependency of the
  core libraries (users of `pcms/backends/omega_h/omega_h.hpp` link
  `Omega_h::omega_h`, which the package config finds); the package config
  finds the private dependencies (KokkosKernels, MeshFields, PETSc, MPI) only
  for a static PCMS build, MPI, redev and ADIOS2 when the coupler or the
  C/Fortran APIs are built, and Omega_h when its backend is installed;
  installed libraries carry an RPATH. Per-backend libraries and `COMPONENTS`
  stay deferred.
- **`Discretization` is gone (semantic PR `layouts-geometry`, before 0.4).**
  Layouts hold a `std::shared_ptr<const Geometry>` with a public
  `GetGeometry()`, `DOFHolderOwners(layout)` reads it, and the XGC
  classification lives in the XGC layout; `src/pcms/discretization/` and the
  `pcms::discretization` target do not exist.
- The geometry kinds are public (stage 7): the Omega_h kind is in
  `include/pcms/backends/omega_h/geometry.hpp` (with `point_search.hpp`), the
  uniform-grid and point-cloud kinds in `include/pcms/backends/uniform_grid/geometry.hpp` and
  `include/pcms/backends/point_cloud/geometry.hpp` (with the headers they include: the point
  layout, the MLS evaluator, `field/evaluator/mls_interpolation.hpp`, `localization/mls_support_helpers.hpp`
  and `localization/uniform_grid_search.hpp`). Each factory is defined next to its class; the one for
  Omega_h is declared in `backends/omega_h/omega_h.hpp`.
- `transfer/linear_interpolant.hpp` and `transfer/multidimarray.hpp` moved to
  `backends/uniform_grid/` (only the uniform-grid evaluator uses them; field
  must not include transfer headers).
- In a backend with its own nested `detail` namespace, references to
  `pcms::detail` names are written `pcms::detail::X`.

## Phase 0 checkpoint (approved)

The classification in `inventory.md` is approved with these decisions:

1. Coupler communicator templates (`field_communicator`, `field_layout_communicator`,
   `field_exchange_planner`, `global_communicator`, `field_serializer`) become
   installed-detail (`include/pcms/coupler/detail/`); redev remains a public
   dependency of `pcms::coupler` only.
2. XGC and MFEM factories are `pcms::backend::xgc::CreateFieldFactory` and
   `pcms::backend::mfem::CreateFieldFactory`, returning
   `std::shared_ptr<FieldFactory>` (they are not evaluable function spaces).
3. `SimpleFieldData` stays a public class.
4. `spline_interpolator.hpp` becomes private.
5. `CreateUniformGridBinaryField` is a public free function in the Omega_h
   backend header.
6. `utility/print.hpp` is public (`setStdout`/`setStderr` let applications
   redirect PCMS output); `bounding_box`, `array_mask`, `mpi_type` are private.
7. Localization stays private (its public API is deferred).

Further decisions: the external usage example is rewritten in Phase 9 to use
only the public API; Python tests/examples are updated to the new Python names;
`UniformGrid::ClosestCellID` takes any indexable point type (template) instead of
`Omega_h::Vector<dim>` (`std::array` is not usable in device code without
relaxed constexpr); `FindPETSc.cmake` stays installed because the package
config's `find_dependency(PETSc)` needs it.

## Phases 5-9 notes

- **Factories (Phase 5).** The geometry base already has the space factories
  (on a `Geometry`); `pcms/field/function_spaces.hpp` declares them and replaces
  the three per-space headers, and the geometry factories are declared in
  `geometry/geometry.hpp` and `backends/omega_h/omega_h.hpp`. The Omega_h
  geometry carries the global-id tag name and the optional owned mask
  (`OmegaHGeometryOptions`), which were `LagrangeFunctionSpace::FromMesh`
  arguments. There are no discretization factories (`Discretization` is gone). `backend::omega_h::CreatePointCloudGeometry(geometry, entity_dim, ...)`
  builds the point-cloud geometry of mesh entities (reconstruction spaces over
  element centroids). `CreateUniformGridBinaryField` returns only the `Field`
  (its layout type is private).
- **Reclassified while splitting:** `utility/bounding_box.hpp` is public
  (`UniformGrid::GetCellBBOX` returns `AABBox`; `UniformGrid` is public API).
- **Install rules:** every library's installed FILE_SET contains only its
  headers under `include/pcms`; private headers are private sources.
  `pcms::localization` switched from `PUBLIC_HEADER` (installed flat) to a
  FILE_SET. The C API install include directory is `include`.
- **PUBLIC dependencies that remain:** Kokkos; redev (coupler and its
  installed-detail templates); perfstubs (`utility/detail/profile.hpp`); MFEM
  (MFEM backend header); Omega_h (Omega_h and XGC backend headers, and core
  internals). MeshFields is PRIVATE.
- **Omega_h out of core public signatures (Phase 6):** no new vector type;
  `UniformGrid` point queries are templates on the point type.
  The Python bindings follow: `create_uniform_grid_from_mesh`,
  `compute_entity_centroids` and `map_entity_field_to_vertices_{average,mls}`
  take an Omega_h mesh and moved to `pcms.backend.omega_h`; the
  `compute_entity_centroids_few` alias is removed; `patch_size_for_degree`
  (no mesh) stays top-level (maintainer decision during review).
- **Sub-library namespaces (Phase 7):** private code of utility,
  discretization, localization, field, transfer and the coupler's internal
  classes (including its installed-detail templates) moved to `pcms::<lib>`.
  Transient has no internal code. `field/evaluator/interpolator_aliases.hpp`
  keeps its generic aliases (`member_type`, `ScratchMatView`, ...) in `pcms::`
  because other files define unrelated local aliases with those names.
- **Packaging (Phase 8):** package files in `lib/cmake/pcms`
  (`PCMS_INSTALL_CMAKEDIR`); `SameMinorVersion`; version 0.4.0, SOVERSION 0.4;
  SWIG `.i` files no longer installed; `FindPETSc.cmake` kept (needed by the
  config's `find_dependency(PETSc)`); baked `HINTS` removed; baked
  `PETSC_DIR`/`PETSC_ARCH` defaults kept (used by the installed FindPETSc).
- **Guards and migration (Phase 9):** see `tools/check-public-includes.py`,
  `test/public_headers/`, `tools/migrate-pcms.sh`, `MIGRATION.md`.
