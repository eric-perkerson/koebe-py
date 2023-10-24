# Combinatorial Topology
TODO: fix point in hole being passed incorrectly that causes outer region to be eaten away

Computational topology and geometry algorithms for finding singular level curves of harmonic functions.

Draw polygonal regions in the plane with polygonal holes using draw_region.py.

Triangulate the region with acute triangles using the aCute package by Alper Ungor, found here: https://www.cise.ufl.edu/~ungor/aCute/algorithm.html

Included is a Julia script for uniformly refining a triangulation and its associated topology (for building the Voronoi dual) in O(n) time.

Solve the Laplacian partial differential equation (PDE) with Dirichlet boundary condition on the triangulated region.

Partition the region along the singular level curves of the solution to the PDE.

Build a harmonic conjugate for the solution to the PDE on each annular component.

# How to Triangulate Example
Choose a name for the example, e.g. my_example (make sure the name has no space characters)
`python draw_region.py my_example`  # Will generate a my_example.poly file for the drawn region

Now the parts of the tri_playground file starting with
`subprocess.run([
        'julia',
        'triangulate_via_julia.jl',
        file_stem,
        file_stem,
        "500"
    ])
`
can be run.
`julia triangulate_via_julia.jl my_example my_example MINIMUM_NUMBER_OF_TRIANGLES`  # MINIMUM_NUMBER_OF_TRIANGLES should be a number, e.g. 1500


# TODO
Add examples of:
round annulus
africa with round holes
south america with asymmetric, non-convex holes
3-fold symmetry
3-fold asymmetry
