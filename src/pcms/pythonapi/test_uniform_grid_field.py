"""
Field-centric Python examples for uniform-grid-backed function spaces.
"""
import numpy as np
import pcms
import PyOmega_h as omega_h


def test_uniform_grid_field_creation():
    """Create a 2D field from a uniform-grid function space."""
    # Create a simple 2D structured grid.
    grid = pcms.UniformGrid2D()
    grid.bot_left = [0.0, 0.0]
    grid.edge_length = [10.0, 10.0]
    grid.divisions = [4, 4]

    # Build the function space from the grid, then create a Field from it.
    factory = pcms.LagrangeFunctionSpace.from_uniform_grid(
        grid, 1, pcms.CoordinateSystem.Cartesian()
    )
    field = factory.create_field()

    # Order-1 2D grids store data at vertices, so a 4x4 cell grid has 5x5
    # DOF holders.
    expected_vertices = (grid.divisions[0] + 1) * (grid.divisions[1] + 1)
    print(f"Grid cells: {grid.get_num_cells()}")
    print(f"Num components: {field.get_num_components()}")
    print(f"Num owned DOF holders: {field.get_num_dof_holders()}")
    assert field.get_num_components() == 1
    assert field.get_num_dof_holders() == expected_vertices
    print(f"Field created: {field is not None}")


def test_uniform_grid_field_data_operations():
    """Set and get flat DOF data through Field."""
    # A 2x2 cell grid has 3x3 vertex DOF holders.
    grid = pcms.UniformGrid2D()
    grid.bot_left = [0.0, 0.0]
    grid.edge_length = [10.0, 10.0]
    grid.divisions = [2, 2]

    factory = pcms.LagrangeFunctionSpace.from_uniform_grid(
        grid, 1, pcms.CoordinateSystem.Cartesian()
    )
    field = factory.create_field()

    # Field data is set and retrieved as a flat 1D DOF array.
    data = np.arange(field.get_num_dof_holders(), dtype=np.float64)
    print(f"Number of vertices: {field.get_num_dof_holders()}")
    print(f"Setting data: {data}")
    field.set_dof_holder_data(data)
    retrieved_data = field.get_dof_holder_data()
    print(f"Retrieved data: {retrieved_data}")
    np.testing.assert_allclose(retrieved_data, data)
    print("Data successfully set and retrieved!")


def test_uniform_grid_field_coordinates_2d():
    """Expose 2D DOF-holder coordinates through Field."""
    # Coordinate queries should now flow through Field rather than layout
    # objects.
    grid = pcms.UniformGrid2D()
    grid.bot_left = [0.0, 0.0]
    grid.edge_length = [10.0, 10.0]
    grid.divisions = [2, 3]

    factory = pcms.LagrangeFunctionSpace.from_uniform_grid(
        grid, 1, pcms.CoordinateSystem.Cartesian()
    )
    field = factory.create_field()

    coords = field.get_dof_holder_coordinates()
    expected_vertices = (grid.divisions[0] + 1) * (grid.divisions[1] + 1)

    assert isinstance(coords, np.ndarray)
    assert coords.shape == (expected_vertices, 2)
    np.testing.assert_allclose(coords[0], [0.0, 0.0])
    np.testing.assert_allclose(coords[-1], [10.0, 10.0])
    print("2D field coordinates verified!")


def test_uniform_grid_closest_cell():
    """UniformGrid helpers remain available for grid setup."""
    # Grid setup helpers remain public even though layout classes do not.
    grid = pcms.UniformGrid2D()
    grid.bot_left = [0.0, 0.0]
    grid.edge_length = [10.0, 10.0]
    grid.divisions = [4, 4]

    cell_id = grid.closest_cell_id(np.array([5.0, 5.0]))
    cell_id_outside = grid.closest_cell_id(np.array([-1.0, -1.0]))
    print(f"Point {[5.0, 5.0]} is in cell {cell_id}")
    print(f"Point {[-1.0, -1.0]} (outside) maps to cell {cell_id_outside}")
    assert cell_id >= 0
    assert cell_id_outside >= 0


def test_uniform_grid_field_coordinates_3d():
    """Expose 3D DOF-holder coordinates and data through Field."""
    # 3D coordinate access and data operations follow the same pattern as 2D.
    grid = pcms.UniformGrid3D()
    grid.bot_left = [0.0, 0.0, 0.0]
    grid.edge_length = [10.0, 10.0, 10.0]
    grid.divisions = [2, 1, 3]

    factory = pcms.LagrangeFunctionSpace.from_uniform_grid(
        grid, 1, pcms.CoordinateSystem.Cartesian()
    )
    field = factory.create_field()

    coords = field.get_dof_holder_coordinates()
    expected_vertices = ((grid.divisions[0] + 1) *
                         (grid.divisions[1] + 1) *
                         (grid.divisions[2] + 1))

    assert coords.shape == (expected_vertices, 3)
    np.testing.assert_allclose(coords[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(coords[-1], [10.0, 10.0, 10.0])
    print("3D field coordinates verified!")

    # Data set/get on a 3D grid field.
    data = np.ones(expected_vertices, dtype=np.float64) * 42.0
    field.set_dof_holder_data(data)
    np.testing.assert_allclose(field.get_dof_holder_data(), data)
    print("3D field data set/get verified!")


def test_uniform_grid_field_evaluation():
    """Evaluate a uniform-grid field at explicit query points."""
    grid = pcms.UniformGrid2D()
    grid.bot_left = [0.0, 0.0]
    grid.edge_length = [10.0, 10.0]
    grid.divisions = [2, 2]

    factory = pcms.LagrangeFunctionSpace.from_uniform_grid(
        grid, 1, pcms.CoordinateSystem.Cartesian()
    )
    field = factory.create_field()

    # Set a linear field f(x,y) = x + y so the expected values are exact.
    coords = field.get_dof_holder_coordinates()
    data = np.array([coords[i, 0] + coords[i, 1]
                     for i in range(field.get_num_dof_holders())],
                    dtype=np.float64)
    field.set_dof_holder_data(data)

    eval_coords = np.array([[2.5, 2.5], [7.5, 7.5]], dtype=np.float64)
    request = pcms.EvaluationRequest.from_coordinates(eval_coords)
    evaluator = factory.create_point_evaluator(request)
    results = np.zeros((len(eval_coords), 1), dtype=np.float64)
    evaluator.evaluate(field, results)

    print(f"Evaluation results: {results.flatten()}")
    print(f"Expected (approximately): [5.0, 15.0]")
    np.testing.assert_allclose(results.flatten(), [5.0, 15.0], atol=1e-10)
    print("Uniform grid field evaluation verified!")


def test_uniform_grid_workflow(world):
    """
    Test complete UniformGrid workflow with Omega_h mesh and field interpolation.
    
    This test:
    1. Creates an Omega_h mesh
    2. Creates a uniform grid from the mesh
    3. Creates a binary mask field
    4. Creates and initializes an Omega_h field with f(x,y) = x + 2*y
    5. Transfers the field to uniform grid via interpolation
    6. Verifies the transferred values
    """
    # Create a simple 2D box mesh: 1.0 x 1.0 domain with 4x4 elements
    mesh = omega_h.build_box(
        world,
        omega_h.Family.SIMPLEX,
        1.0, 1.0, 0.0,
        4, 4, 0,
        False
    )

    # Create a uniform grid that covers the mesh bounding box, then construct a
    # binary mask field directly as a Field object.
    grid = pcms.create_uniform_grid_from_mesh(mesh, [4, 4])
    print(f"Created uniform grid with {grid.get_num_cells()} cells")
    mask_field = pcms.create_uniform_grid_binary_field(mesh, [4, 4])
    mask_data = mask_field.get_dof_holder_data()
    print(f"Created mask field with {len(mask_data)} vertices")

    # Build a mesh-backed source field and initialize it with f(x,y)=x+2y.
    omega_h_factory = pcms.LagrangeFunctionSpace.from_mesh(mesh, 1)
    omega_h_field = omega_h_factory.create_field()

    coords = omega_h_field.get_dof_holder_coordinates()
    omega_h_data = np.zeros(omega_h_field.get_num_dof_holders(), dtype=np.float64)
    for i in range(omega_h_field.get_num_dof_holders()):
        omega_h_data[i] = coords[i, 0] + 2.0 * coords[i, 1]
    omega_h_field.set_dof_holder_data(omega_h_data)
    print(f"Initialized Omega_h field with {omega_h_field.get_num_dof_holders()} nodes")

    ug_factory = pcms.LagrangeFunctionSpace.from_uniform_grid(
        grid, 1, pcms.CoordinateSystem.Cartesian()
    )
    ug_field = ug_factory.create_field()

    # Transfer from the unstructured mesh field to the uniform-grid field.
    interp = pcms.Interpolator(omega_h_factory, ug_factory)
    interp.apply(omega_h_field, ug_field)
    print("Field interpolation completed")

    ug_field_data = ug_field.get_dof_holder_data()
    ug_coords = ug_field.get_dof_holder_coordinates()

    # The 4x4 cell grid should produce 5x5 vertex DOFs.
    assert len(mask_data) == 25
    assert len(ug_field_data) == 25

    # Verify interpolation at every target DOF holder.
    num_errors = 0
    for vertex_id in range(len(ug_field_data)):
        x = ug_coords[vertex_id, 0]
        y = ug_coords[vertex_id, 1]
        expected = x + 2.0 * y
        actual = ug_field_data[vertex_id]
        error = abs(actual - expected)
        if error > 1e-10:
            num_errors += 1
            print(
                f"Vertex {vertex_id} at ({x:.2f}, {y:.2f}): "
                f"expected {expected:.4f}, got {actual:.4f}, error {error:.2e}"
            )
        assert error < 1e-10
    if num_errors == 0:
        print("All uniform grid field values verified successfully!")


def test_omega_h_to_omega_h_transfer_workflow(world):
    """
    Transfer a field from one Omega_h mesh to another Omega_h mesh.

    This preserves coverage for the workflow where both source and target use
    mesh-backed function spaces rather than a uniform grid target.
    """
    # Source mesh (coarser).
    src_mesh = omega_h.build_box(
        world,
        omega_h.Family.SIMPLEX,
        1.0, 1.0, 0.0,
        4, 4, 0,
        False
    )

    # Target mesh (finer).
    tgt_mesh = omega_h.build_box(
        world,
        omega_h.Family.SIMPLEX,
        1.0, 1.0, 0.0,
        8, 8, 0,
        False
    )

    # Create function spaces and fields on both meshes.
    src_factory = pcms.LagrangeFunctionSpace.from_mesh(src_mesh, 1)
    tgt_factory = pcms.LagrangeFunctionSpace.from_mesh(tgt_mesh, 1)
    src_field = src_factory.create_field()
    tgt_field = tgt_factory.create_field()
    print("Created source and target Omega_h fields")

    # Initialize the source field with f(x,y)=x+2y.
    src_coords = src_field.get_dof_holder_coordinates()
    src_data = np.zeros(src_field.get_num_dof_holders())
    for i in range(len(src_data)):
        src_data[i] = src_coords[i, 0] + 2.0 * src_coords[i, 1]
    src_field.set_dof_holder_data(src_data)
    print(f"Initialized source field with {len(src_data)} DOF values")

    # Transfer onto the target mesh and verify the target DOF values.
    interp = pcms.Interpolator(src_factory, tgt_factory)
    interp.apply(src_field, tgt_field)
    print("Omega_h to Omega_h field interpolation completed")

    tgt_coords = tgt_field.get_dof_holder_coordinates()
    tgt_data = tgt_field.get_dof_holder_data()
    errors = 0
    for i in range(tgt_field.get_num_dof_holders()):
        expected = tgt_coords[i, 0] + 2.0 * tgt_coords[i, 1]
        error = abs(expected - tgt_data[i])
        if error > 1e-10:
            errors += 1
        assert error < 1e-10
    print("Omega_h to Omega_h transfer verification completed")


def main():
    lib = omega_h.OmegaHLibrary()
    world = lib.world()

    print("=" * 60)
    print("Testing UniformGrid Field Creation")
    print("=" * 60)
    test_uniform_grid_field_creation()

    print("\n" + "=" * 60)
    print("Testing UniformGrid Field Data Operations")
    print("=" * 60)
    test_uniform_grid_field_data_operations()

    print("\n" + "=" * 60)
    print("Testing UniformGrid Field Coordinates (2D)")
    print("=" * 60)
    test_uniform_grid_field_coordinates_2d()

    print("\n" + "=" * 60)
    print("Testing UniformGrid Field Coordinates (3D)")
    print("=" * 60)
    test_uniform_grid_field_coordinates_3d()

    print("\n" + "=" * 60)
    print("Testing Closest Cell ID")
    print("=" * 60)
    test_uniform_grid_closest_cell()

    print("\n" + "=" * 60)
    print("Testing UniformGrid Field Evaluation")
    print("=" * 60)
    test_uniform_grid_field_evaluation()

    print("\n" + "=" * 60)
    print("Testing UniformGrid Workflow")
    print("=" * 60)
    test_uniform_grid_workflow(world)

    print("\n" + "=" * 60)
    print("Testing Omega_h to Omega_h Transfer Workflow")
    print("=" * 60)
    test_omega_h_to_omega_h_transfer_workflow(world)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

    del world


if __name__ == "__main__":
    main()