import pcms
import PyOmega_h as omega_h
import numpy as np

def test_copy(world, dim, order, num_components):
    """Test copying omega_h field data."""
    nx = 100
    ny = 100 if dim > 1 else 0
    nz = 100 if dim > 2 else 0
    print(f"\nStarting test: dim={dim}, order={order}, num_components={num_components}")

    # Build mesh
    mesh = omega_h.build_box(
        world, 
        omega_h.Family.SIMPLEX, 
        1.0, 1.0, 1.0, 
        nx, ny, nz, 
        False
    )
    print(f"  Built {dim}D mesh with {nx}x{ny}x{nz} elements")
    print(f"  Mesh type: {type(mesh)}")
    print(f"  Mesh object: {mesh}")

    # Create factory and layout
    print("  About to create factory...")
    factory = pcms.LagrangeFunctionSpace.from_mesh(
        mesh, order, num_components, pcms.CoordinateSystem.Cartesian()
    )
    print(f"Testing dim={dim}, order={order}, num_components={num_components}...")

    # Create original field and set data
    original = factory.create_field()
    print("  Created original field")
    ndata = original.get_num_dof_holders() * original.get_num_components()
    print(f"  Number of data points: {ndata}")

    # Create sequential array of IDs
    ids = np.arange(ndata, dtype=np.float64)
    print(f"  Created array of IDs from 0 to {ndata-1}")
    original.set_dof_holder_data(ids)
    print("  Set data in original field")

    # Create copied field and copy data
    copied = factory.create_field()
    copier = pcms.Copy(factory, factory)
    copier.apply(original, copied)
    print("  Copied data to new field")

    # Get copied data
    copied_array = copied.get_dof_holder_data()

    # Verify the copy
    assert len(copied_array) == ndata, f"Expected {ndata} elements, got {len(copied_array)}"

    # Check that all values match
    differences = np.abs(ids - copied_array)
    num_matches = np.sum(differences < 1e-12)

    assert num_matches == ndata, f"Only {num_matches}/{ndata} elements matched"

    print(f"✓ Test passed: dim={dim}, order={order}, num_components={num_components}")

def main():
    """Run all test cases"""
    print("Testing copy omega_h field data...")

    # Initialize Omega_h library
    lib = omega_h.OmegaHLibrary()
    world = lib.world()
    print("Initialized Omega_h library and world")

    # Run test cases
    test_copy(world, 2, 1, 1)
    print("first passed")
    test_copy(world, 2, 2, 1)

    print("\nAll tests passed!")

    del world
    del lib

if __name__ == "__main__":
    main()