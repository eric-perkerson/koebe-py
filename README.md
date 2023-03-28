# Combinatorial Topology
TODO: fix point in hole being passed incorrectly that causes outer region to be eaten away

Computational topology and geometry algorithms for finding singular level curves of harmonic functions.

Draw polygonal regions in the plane with polygonal holes using draw_region.py.

Triangulate the region with acute triangles using the aCute package by Alper Ungor, found here: https://www.cise.ufl.edu/~ungor/aCute/algorithm.html

Included is a Julia script for uniformly refining a triangulation and its associated topology (for building the Voronoi dual) in O(n) time.

Solve the Laplacian partial differential equation (PDE) with  Dirichlet boundary condition on the triangulated region.

Partition the region along the singular level curves of the solution to the PDE.

Build a harmonic conjugate for the solution to the PDE on each annular component.

# How to Triangulate Example
python draw_region.py  # Will generate a .poly file for the drawn region
julia triangulate_via_julia.jl relative_file_path_in_regions MINIMUM_NUMBER_OF_TRIANGLES
python chain.py # Will run the chain program
