#!/usr/bin/env python3
"""
Field-centric Python examples for Omega_h-backed function spaces.
"""

import numpy as np
import pcms
import PyOmega_h as omega_h


def test_field_methods(world, dim, order, num_components):
    """Create an Omega_h-backed Field and exercise the public Field API."""
    nx = 10
    ny = 10 if dim > 1 else 0
    nz = 10 if dim > 2 else 0

    mesh = omega_h.build_box(
        world,
        omega_h.Family.SIMPLEX,
        1.0, 1.0, 1.0,
        nx, ny, nz,
        False
    )

    factory = pcms.LagrangeFunctionSpace.from_mesh(
        mesh, order, num_components, pcms.CoordinateSystem.Cartesian()
    )
    field = factory.create_field()

    assert field.get_num_components() == num_components
    assert field.get_num_dof_holders() > 0

    coords = field.get_dof_holder_coordinates()
    assert coords.shape[0] == field.get_num_dof_holders()
    assert coords.shape[1] == dim

    ndata = field.get_num_dof_holders() * num_components
    test_data = np.arange(ndata, dtype=np.float64)
    field.set_dof_holder_data(test_data)
    np.testing.assert_allclose(field.get_dof_holder_data(), test_data)


def test_field_transfer(world, dim, order, num_components):
    """Transfer data between identical Omega_h-backed function spaces."""
    if num_components != 1:
        return

    nx = 10
    ny = 10 if dim > 1 else 0
    nz = 10 if dim > 2 else 0

    mesh = omega_h.build_box(
        world,
        omega_h.Family.SIMPLEX,
        1.0, 1.0, 1.0,
        nx, ny, nz,
        False
    )

    source_space = pcms.LagrangeFunctionSpace.from_mesh(
        mesh, order, num_components, pcms.CoordinateSystem.Cartesian()
    )
    target_space = pcms.LagrangeFunctionSpace.from_mesh(
        mesh, order, num_components, pcms.CoordinateSystem.Cartesian()
    )

    source = source_space.create_field()
    target = target_space.create_field()

    coords = source.get_dof_holder_coordinates()
    source_data = np.zeros(source.get_num_dof_holders(), dtype=np.float64)
    for i in range(source.get_num_dof_holders()):
        source_data[i] = np.sum(coords[i, :dim])
    source.set_dof_holder_data(source_data)

    interp = pcms.Interpolator(source_space, target_space)
    interp.apply(source, target)

    np.testing.assert_allclose(target.get_dof_holder_data(), source_data, atol=1e-14)


def test_field_evaluation(world, dim, order, num_components):
    """Evaluate an Omega_h-backed field at explicit query points."""
    nx = 10
    ny = 10 if dim > 1 else 0
    nz = 10 if dim > 2 else 0

    print(
        f"\nTesting field evaluation: dim={dim}, order={order}, "
        f"num_components={num_components}"
    )

    mesh = omega_h.build_box(
        world,
        omega_h.Family.SIMPLEX,
        1.0, 1.0, 1.0,
        nx, ny, nz,
        False
    )

    factory = pcms.LagrangeFunctionSpace.from_mesh(
        mesh, order, num_components, pcms.CoordinateSystem.Cartesian()
    )
    field = factory.create_field()

    print("  Setting up field data...")
    mesh_coords = mesh.coords()
    num_verts = mesh.nverts()
    print(f"  Mesh has {num_verts} vertices")

    num_owned = field.get_num_dof_holders()
    ndata = num_owned * num_components
    print(f"  Setting up field data for {ndata} DOFs")

    def test_func(x, y, z, component):
        return np.sin(5.0 * x * y) + float(component)

    test_data = np.zeros(ndata, dtype=np.float64)
    for i in range(min(num_verts, num_owned)):
        x = mesh_coords[i * dim + 0] if dim >= 1 else 0.0
        y = mesh_coords[i * dim + 1] if dim >= 2 else 0.0
        z = mesh_coords[i * dim + 2] if dim >= 3 else 0.0
        for c in range(num_components):
            idx = i * num_components + c
            test_data[idx] = test_func(x, y, z, c)

    if order == 2 and num_owned > num_verts:
        for i in range(num_verts, num_owned):
            for c in range(num_components):
                idx = i * num_components + c
                test_data[idx] = float(c) + 0.5

    field.set_dof_holder_data(test_data)
    print("  Set field data with test function")

    if dim == 2:
        eval_coords = np.array([
            [0.5, 0.5],
            [0.25, 0.25],
            [0.75, 0.75],
            [0.1, 0.9],
            [0.9, 0.1],
        ], dtype=np.float64)
    else:
        eval_coords = np.array([
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
            [0.1, 0.9, 0.1],
            [0.9, 0.1, 0.9],
        ], dtype=np.float64)

    print(f"  Evaluating at {eval_coords.shape[0]} points")
    request = pcms.EvaluationRequest.from_coordinates(eval_coords)
    evaluator = factory.create_point_evaluator(request)
    print("  Created point evaluator")

    eval_values = np.zeros((eval_coords.shape[0], num_components),
                           dtype=np.float64)
    evaluator.evaluate(field, eval_values)
    print("  Field evaluated successfully")

    for i in range(eval_coords.shape[0]):
        coords_str = ", ".join([f"{eval_coords[i, j]:.3f}" for j in range(dim)])
        values_str = ", ".join(
            [f"{eval_values[i, c]:.4f}" for c in range(num_components)]
        )
        print(f"  Point {i} ({coords_str}): values = [{values_str}]")
        for c in range(num_components):
            val = eval_values[i, c]
            assert not np.isnan(val)
            assert not np.isinf(val)


def test_tag_operations(world, dim):
    """Exercise Omega_h mesh tag creation, mutation, and query helpers."""
    nx = 5
    ny = 5 if dim > 1 else 0
    nz = 5 if dim > 2 else 0

    mesh = omega_h.build_box(
        world,
        omega_h.Family.SIMPLEX,
        1.0, 1.0, 1.0,
        nx, ny, nz,
        False
    )
    rng = np.random.default_rng(0)

    mesh.add_tag(0, "test_float", 1, dtype="float64")
    mesh.add_tag(0, "test_int32", 1, dtype="int32")
    mesh.add_tag(0, "test_int64", 1, dtype="int64")

    assert mesh.has_tag(0, "test_float")
    assert mesh.has_tag(0, "test_int32")
    assert mesh.has_tag(0, "test_int64")

    nverts = mesh.nverts()
    temp_data = np.linspace(0.0, 100.0, nverts, dtype=np.float64)
    ids_data = np.arange(nverts, dtype=np.int32)
    velocity_data = rng.standard_normal(nverts * dim).astype(np.float64)

    mesh.add_tag(0, "temperature", 1, temp_data)
    mesh.add_tag(0, "vertex_ids", 1, ids_data)
    mesh.add_tag(0, "velocity", dim, velocity_data)

    np.testing.assert_allclose(mesh.get_tag(0, "temperature"), temp_data)
    np.testing.assert_array_equal(mesh.get_tag(0, "vertex_ids"), ids_data)
    np.testing.assert_allclose(mesh.get_tag(0, "velocity"), velocity_data)

    new_temp = np.ones(nverts, dtype=np.float64) * 50.0
    mesh.set_tag(0, "temperature", new_temp)
    np.testing.assert_allclose(mesh.get_tag(0, "temperature"), new_temp)

    if dim == 3:
      ncomps = 6
      stress_data = rng.standard_normal(mesh.nelems() * ncomps).astype(
        np.float64)
      mesh.add_tag(dim, "stress", ncomps, stress_data,
                   internal=False,
                   array_type=omega_h.ArrayType.SymmetricSquareMatrix)
      assert mesh.has_tag(dim, "stress")

    internal_data = np.zeros(nverts, dtype=np.float64)
    mesh.add_tag(0, "internal_temp", 1, internal_data, internal=True)
    assert mesh.has_tag(0, "internal_temp")

    mesh.remove_tag(0, "test_float")
    assert not mesh.has_tag(0, "test_float")
    assert mesh.has_tag(0, "temperature")
    assert mesh.ntags(0) > 0

    nelems = mesh.nelems()
    elem_quality = rng.random(nelems).astype(np.float64)
    mesh.add_tag(dim, "quality", 1, elem_quality)
    np.testing.assert_allclose(mesh.get_tag(dim, "quality"), elem_quality)

    if dim >= 2:
      nedges = mesh.nedges()
      edge_length = np.ones(nedges, dtype=np.float64)
      mesh.add_tag(1, "edge_marker", 1, edge_length)
      np.testing.assert_allclose(mesh.get_tag(1, "edge_marker"), edge_length)

    elem_verts = mesh.ask_elem_verts()
    global_ids = mesh.globals(0)
    verts_of_elems = mesh.ask_verts_of(dim)
    owned_verts = mesh.owned(0)

    assert len(elem_verts) > 0
    assert len(global_ids) == nverts
    assert len(verts_of_elems) > 0
    assert len(owned_verts) == nverts
    assert np.sum(owned_verts) > 0
    assert isinstance(mesh.has_adj(0, dim), (bool, np.bool_))


def test_entity_coordinates(world, dim):
    """Exercise entity-coordinate and averaging helpers on Omega_h meshes."""
    nx = 4
    ny = 4 if dim > 1 else 0
    nz = 4 if dim > 2 else 0

    mesh = omega_h.build_box(
        world,
        omega_h.Family.SIMPLEX,
        1.0, 1.0, 1.0,
        nx, ny, nz,
        False
    )

    vertex_coords = mesh.coords()
    assert len(vertex_coords) == mesh.nverts() * dim

    if dim >= 2:
      edge_coords = omega_h.average_field(mesh, 1, dim, vertex_coords)
      edge_coords_np = np.array(edge_coords).reshape(-1, dim)
      assert edge_coords_np.shape[0] == mesh.nedges()

    if dim == 2:
      elem_coords = omega_h.average_field(mesh, 2, dim, vertex_coords)
      elem_coords_np = np.array(elem_coords).reshape(-1, dim)
      assert elem_coords_np.shape[0] == mesh.nelems()

    if dim == 3:
      face_coords = omega_h.average_field(mesh, 2, dim, vertex_coords)
      face_coords_np = np.array(face_coords).reshape(-1, dim)
      assert face_coords_np.shape[0] == mesh.nfaces()

      region_coords = omega_h.average_field(mesh, 3, dim, vertex_coords)
      region_coords_np = np.array(region_coords).reshape(-1, dim)
      assert region_coords_np.shape[0] == mesh.nregions()

    vertex_field = np.arange(mesh.nverts(), dtype=np.float64)
    mesh.add_tag(0, "test_vertex_field", 1, vertex_field)
    averaged_field = omega_h.average_field(mesh, dim, 1, vertex_field)
    assert averaged_field.shape[0] == mesh.nents(dim)


def main():
    """Run all test cases"""
    print("Testing Omega_h Field Python bindings...")
    lib = omega_h.OmegaHLibrary()
    world = lib.world()
    print("Initialized Omega_h library and world")

    test_field_methods(world, 2, 1, 1)
    test_field_methods(world, 2, 2, 1)

    print("\n" + "=" * 60)
    print("Testing field evaluation...")
    print("=" * 60)
    test_field_evaluation(world, 2, 1, 1)
    test_field_evaluation(world, 2, 2, 1)

    print("\n" + "=" * 60)
    print("Testing field transfer...")
    print("=" * 60)
    test_field_transfer(world, 2, 1, 1)
    test_field_transfer(world, 2, 2, 1)

    print("\n" + "=" * 60)
    print("Testing tag operations...")
    print("=" * 60)
    test_tag_operations(world, 2)
    test_tag_operations(world, 3)

    print("\n" + "=" * 60)
    print("Testing entity coordinate computation...")
    print("=" * 60)
    test_entity_coordinates(world, 2)
    test_entity_coordinates(world, 3)

    print("\n✓ All tests passed!")
    del world
    del lib


if __name__ == "__main__":
    main()