# point-clouds

Reconstructing a field from scattered samples. Measurements, particles and the
output of other codes are values at scattered points; PCMS reconstructs a
field from them with moving least squares (MLS): the value at a point is a
local polynomial fit to the sites around it. The example samples a known
smooth function at 400 random sites, creates a point-cloud geometry and a
polynomial reconstruction space on it, and evaluates the reconstruction at new
points and onto a uniform grid with an `Interpolator`. The support radius and
its adaptation are shown with a radius that is too small: without adaptation
the points with no site in range are outside the domain of the reconstruction
(and the out-of-bounds policy applies), with adaptation the radius grows until
every point has enough sites. Last, the sites are the vertices or the element
centroids of an Omega_h mesh, where the supports of a point are found by the
mesh's adjacency instead of a distance search (this part needs Omega_h).

Asserted: the quadratic reconstruction of the smooth function is within 2e-3
of it at 200 new points and on the 21 x 21 grid (measured 7e-4); with a radius
of 0.03 and no adaptation the evaluation is refused, with adaptation it is as
accurate as with the right radius; a linear fit reproduces a linear function
from cell centroids at the vertices and from vertices at the centroids
(1e-5).

## Public API used

- `backend::point_cloud::CreatePointCloudGeometry` (from coordinates, and
  through `backend::omega_h::CreatePointCloudGeometry` from the vertices or
  centroids of a mesh)
- `NeighbourhoodOptions` (`radius`, `min_supports`, `adapt_radius`), `MLSOptions`
  (`degree`)
- `CreatePolynomialReconstructionSpace`
- `QueryPoints`, `Interpolator`, `CreateLagrangeSpace` on a uniform grid
- Python: `create_point_cloud_geometry`, `backend.omega_h.create_point_cloud_geometry`,
  `NeighbourhoodOptions`, `MLSOptions`, `create_polynomial_reconstruction_space`

## Run

```console
$ ctest -R example.fields.point-clouds
$ build/examples/fields/point-clouds/fields_point_clouds
$ PYTHONPATH=build/src/pcms/pythonapi python3 examples/fields/point-clouds/main.py
```

## What to look at

The library prints the progress of the support search (`BuildPointCloudSupports:
adjusting radius iter ...`) while it grows the radius: with `radius = 0.12`
it needs three iterations, with `0.03` twelve, ending with the same minimum of
15 supports. The measured errors of both are the same, which is the point of
the adaptation. The mesh part prints no errors but checks that the values
arrive at the right places.
