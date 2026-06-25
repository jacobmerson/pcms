#!/usr/bin/env python3
"""
Conservative field transfer between two Omega_h meshes.

This example:
  1. Builds a source Omega_h 2D simplex mesh, writes it to disk, and reads it
     back (demonstrating the mesh read path).
  2. Evaluates an analytic function onto an order-1 (linear) Lagrange field on
     the source mesh.
  3. Builds, writes, and reads back a *different* target Omega_h mesh that
     carries its own order-1 Lagrange field.
  4. Transfers the source field onto the target field with both the Monte
     Carlo (control-variate) and the mesh-intersection conservative projection
     methods, and reports the L2 / max errors and the conservation error
     (preservation of the integral) for each.

Both conservative-projection operators require order-1, scalar Lagrange spaces
on 2D simplex meshes.
"""

import os
import gc
import tempfile

import numpy as np
import pcms


# Smooth analytic field sampled onto the source mesh.
def source_function(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y) + 0.5 * x + 0.25 * y


def build_and_roundtrip_mesh(world, lib, nx, ny, path):
    """Build a 2D box mesh, write it to disk, and read it back."""
    mesh = pcms.build_box(
        world,
        pcms.Family.SIMPLEX,
        1.0, 1.0, 0.0,   # unit square (z = 0 -> 2D)
        nx, ny, 0,
        False,
    )
    pcms.write_mesh_binary(path, mesh)
    del mesh
    gc.collect()
    return pcms.read_mesh_binary(path, lib)


def integral_of_field(mesh, field):
    """Integral of an order-1 nodal field via element-averaged quadrature."""
    values = field.get_dof_holder_data()
    elem_verts = mesh.ask_elem_verts().reshape(-1, 3)
    areas = mesh.ask_sizes()  # element areas for a 2D simplex mesh
    elem_avg = values[elem_verts].mean(axis=1)
    return float(np.sum(areas * elem_avg))


def evaluate_function_onto_field(space, func):
    """Create a field on `space` and fill it from `func` at the DOF holders."""
    field = space.create_field()
    coords = field.get_dof_holder_coordinates()
    field.set_dof_holder_data(func(coords[:, 0], coords[:, 1]))
    return field


def report(name, target_field, target_mesh, reference_coords,
           reference_values, source_integral):
    values = target_field.get_dof_holder_data()
    err = values - reference_values
    l2 = np.sqrt(np.mean(err ** 2))
    linf = np.max(np.abs(err))
    target_integral = integral_of_field(target_mesh, target_field)
    cons = abs(target_integral - source_integral)
    print(f"  {name}")
    print(f"    L2 error vs analytic        : {l2:.3e}")
    print(f"    max error vs analytic       : {linf:.3e}")
    print(f"    target integral             : {target_integral:.6f}")
    print(f"    conservation error |dI|     : {cons:.3e}")


def main():
    lib = pcms.OmegaHLibrary()
    world = lib.world()

    work_dir = tempfile.mkdtemp(prefix="pcms_conservative_transfer_")
    try:
        print("=" * 64)
        print("Conservative field transfer: MonteCarlo and mesh intersection")
        print("=" * 64)

        # 1. Source mesh (read from disk) + linear Lagrange field.
        source_path = os.path.join(work_dir, "source.osh")
        source_mesh = build_and_roundtrip_mesh(world, lib, 12, 12, source_path)
        print(f"\nSource mesh : {source_mesh.nverts()} verts, "
              f"{source_mesh.nelems()} elems (read from {source_path})")

        # The conservative-projection operators require the native Omega_h
        # Lagrange backend (not the default MeshFields layout).
        source_space = pcms.LagrangeFunctionSpace.from_mesh(
            source_mesh, 1, 1, pcms.CoordinateSystem.Cartesian,
            backend=pcms.LagrangeFunctionSpace.Backend.OmegaH,
        )
        source_field = evaluate_function_onto_field(source_space,
                                                    source_function)
        source_integral = integral_of_field(source_mesh, source_field)
        print(f"Source field: order-1 Lagrange, "
              f"integral = {source_integral:.6f}")

        # 2. Target mesh (read from disk) with its own linear Lagrange field.
        target_path = os.path.join(work_dir, "target.osh")
        target_mesh = build_and_roundtrip_mesh(world, lib, 17, 17, target_path)
        print(f"\nTarget mesh : {target_mesh.nverts()} verts, "
              f"{target_mesh.nelems()} elems (read from {target_path})")

        target_space = pcms.LagrangeFunctionSpace.from_mesh(
            target_mesh, 1, 1, pcms.CoordinateSystem.Cartesian,
            backend=pcms.LagrangeFunctionSpace.Backend.OmegaH,
        )

        # Analytic reference values at the target DOF holders.
        target_template = target_space.create_field()
        target_coords = target_template.get_dof_holder_coordinates()
        reference_values = source_function(target_coords[:, 0],
                                           target_coords[:, 1])

        print("\nTransferring source field to target field:")

        # 3a. Mesh-intersection conservative projection.
        mi_target = target_space.create_field()
        mesh_intersection = pcms.OmegaHConservativeProjection(
            source_space, target_space
        )
        mesh_intersection.apply(source_field, mi_target)
        report("mesh intersection (exact quadrature)", mi_target, target_mesh,
               target_coords, reference_values, source_integral)

        # 3b. Monte Carlo (control-variate) conservative projection.
        mc_target = target_space.create_field()
        monte_carlo = pcms.OmegaHControlVariateProjection(
            source_space,
            target_space,
            samples_per_element=64,
            sampling=pcms.MonteCarloSampling.UniformRandom,
            seed=8675309,
        )
        monte_carlo.apply(source_field, mc_target)
        report("Monte Carlo (control variate, UniformRandom)", mc_target, target_mesh,
               target_coords, reference_values, source_integral)

        print("\n✓ Field transfer completed with both methods.")

    finally:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)
        #del world
        #del lib


if __name__ == "__main__":
    main()
