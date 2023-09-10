"""
An example: one singular vertex of max index, genus 3 surface

"""

import pyvista

import dolfinx

from dolfinx.fem import (Constant, dirichletbc, Function , FunctionSpace, 
                         locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile

from dolfinx import mesh

from dolfinx import plot 

from dolfinx.plot import create_vtk_mesh

from ufl import (SpatialCoordinate, TestFunction, TrialFunction,
                 dx, grad, inner)

from petsc4py.PETSc import ScalarType

from mpi4py import MPI # new

import meshio

import numpy as np

# Definition to create mesh using meshio

def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, 
                           cell_data={"name_to_read":[cell_data]})
    return out_mesh

# Determine the number of parallel processes
proc = MPI.COMM_WORLD.rank 

# read the msh file from path and use create mesh and write to xdmf files:
# mesh.xdmf and mt.xdmf

if proc == 0:
    # Read in mesh
    msh = meshio.read("regions/3_fold_sym/3_fold_sym.output.msh")
    # Create and save one file for the mesh, and one file for the facets 
    triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
    line_mesh = create_mesh(msh, "line", prune_z=True)
    meshio.write("regions/3_fold_sym/mesh.xdmf", triangle_mesh)
    meshio.write("regions/3_fold_sym/mt.xdmf", line_mesh)

# This creates the meshtags and topology of the mesh from the xdmf files above

with XDMFFile(MPI.COMM_WORLD, "regions/3_fold_sym/mesh.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(mesh, name="Grid")
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim-1)
with XDMFFile(MPI.COMM_WORLD, "regions/3_fold_sym/mt.xdmf", "r") as xdmf:
    ft = xdmf.read_meshtags(mesh, name="Grid")


################################################################
##
##  Generating a list of edges and triangles.
##
#################################################################
# mesh = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.triangle)

# fdim = mesh.topology.dim - 1
# mesh.topology.create_connectivity(fdim, 0)
# num_facets_owned_by_proc = mesh.topology.index_map(fdim).size_local
# geometry_entitites =  dolfinx.cpp.mesh.entities_to_geometry(mesh, fdim, 
#                     np.arange(num_facets_owned_by_proc, dtype=np.int32), False)

# fdim_2 = mesh.topology.dim 
# mesh.topology.create_connectivity(fdim_2, 0)
# num_traingles_owned_by_proc = mesh.topology.index_map(fdim_2).size_local
# geometry_entitites_triangles =  dolfinx.cpp.mesh.entities_to_geometry(mesh, fdim_2, 
#                     np.arange(num_traingles_owned_by_proc, dtype=np.int32), False)
####################################################################
####################################################################
# points = mesh.geometry.x
# for e, entity in enumerate(geometry_entitites):
#     print(e, points[entity])


# mesh = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.triangle)
# fdim = mesh.topology.dim - 1
# mesh.topology.create_connectivity(fdim, 0)

# num_facets_owned_by_proc = mesh.topology.index_map(fdim).size_local
# geometry_entitites = dolfinx.cpp.mesh.entities_to_geometry(mesh, 
#             fdim, np.arange(num_facets_owned_by_proc, dtype=np.int32), False)


# #triangles=dolfinx.cpp.mesh.get_entity_vertices(dolfinx.cpp.mesh.point,0)
# #geo_ent_1 = dolfinx.cpp.mesh.entities_to_geometry(mesh, 
#             2, np.arange(num_facets_owned_by_proc, dtype=np.int32), False)


# points = mesh.geometry.x

# for e, entity in enumerate(geometry_entities):
#     print(e, points[entity])


#################################################################


### The function space

V = FunctionSpace(mesh, ("CG", 1))

# The following will print the coordinates of vertices with degree of freedom
# print(V.tabulate_dof_coordinates())
#

### Define the boundaries and boundary conditions.

left_facets_1 = ft.find(1)
left_dofs_1 = locate_dofs_topological(V, mesh.topology.dim-1, left_facets_1)
bcs_1 = dirichletbc(ScalarType(1), left_dofs_1, V)

left_facets_2 = ft.find(2)
left_dofs_2 = locate_dofs_topological(V, mesh.topology.dim-1, left_facets_2)
bcs_2 = dirichletbc(ScalarType(0), left_dofs_2, V)

left_facets_3 = ft.find(3)
left_dofs_3 = locate_dofs_topological(V, mesh.topology.dim-1, left_facets_3)
bcs_3 = dirichletbc(ScalarType(0), left_dofs_3, V)

left_facets_4 = ft.find(4)
left_dofs_4 = locate_dofs_topological(V, mesh.topology.dim-1, left_facets_4)
bcs_4 = dirichletbc(ScalarType(0), left_dofs_4, V)


### Set the trial functions, the bi-linear and linear forms, problem to solve
### and the solution.

u, v = TrialFunction(V), TestFunction(V)
a = inner(grad(u), grad(v)) * dx
x = SpatialCoordinate(mesh)
L = Constant(mesh, ScalarType(0)) * v * dx

problem = LinearProblem(a, L, bcs=[bcs_1,bcs_2,bcs_3,bcs_4], 
                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

##### All of this  is for depict the level curves, the singular one and the 
##### surface

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

x_coord = mesh.geometry.x[:,0]
y_coord = mesh.geometry.x[:,1]

fdim = mesh.topology.dim - 1
mesh.topology.create_connectivity(fdim, 0)
num_facets_owned_by_proc = mesh.topology.index_map(fdim).size_local
geometry_entitites =  dolfinx.cpp.mesh.entities_to_geometry(mesh, fdim, 
                    np.arange(num_facets_owned_by_proc, dtype=np.int32), False)

fdim_2 = mesh.topology.dim 
mesh.topology.create_connectivity(fdim_2, 0)
num_triangles_owned_by_proc = mesh.topology.index_map(fdim_2).size_local
geometry_entitites_triangles =  dolfinx.cpp.mesh.entities_to_geometry(mesh, fdim_2, 
                    np.arange(num_triangles_owned_by_proc, dtype=np.int32), False)

triangles =  geometry_entitites_triangles
z_coord= uh.x.array

triangulation = mpl.tri.Triangulation(x_coord, y_coord,
                                      triangles)

fig = plt.figure(figsize=(12,12))
gs = mpl.gridspec.GridSpec(2, 2)

ax1= fig.add_subplot(gs[0,0])
ax1.set_aspect('equal')
ax1.triplot(triangulation, 'go-', lw=1.0)
ax1.set_title("tri plot of the triangulation") 

ax2 =fig.add_subplot(gs[0,1:])
levels =[0.33051698]
ax2.tricontour(triangulation, z_coord, levels=levels)
ax2.tricontour(triangulation, z_coord, 100)
ax2.set_title("contours of the triangulation with respect to the solution") 

ax3 = fig.add_subplot(gs[1,0],projection='3d')
ax3.plot_trisurf(triangulation, z_coord)
ax3.set_title('3D Surface Plot from Triangulation')

fig.tight_layout()
plt.show()




### Reading and Printing the solution to a file
# Write the solution, uh, to a file, entry per line

with open('regions/3_fold_sym/solution_fenicsx.pde', 'w') as output_file:
     num_vertices = len(uh.x.array)
     output_file.write(f'{num_vertices}\n')
     for i, entry in enumerate(uh.x.array, 1):
#         output_file.write(str(entry) + '\n')
          output_file.write(f"{i} {entry}\n")


# with open('regions/3_fold_sym/solution_fenicsx.txt','w') as output_file:
#      for entry in uh.x.array:
# #         output_file.write(str(entry) + '\n')
#           output_file.write(f"{entry}\n")


# open file in read mode and copy solution values at node to 
# values[], a list
try:
    file_of_solution = open('regions/3_fold_sym/solution_fenicsx.txt','r')
except:
    print("I can't find this file, search the correct folder.\n")
else:     
    values=[]
    for line in file_of_solution.readlines():
        #print(line, end='')
        values.append(float(line[0:-1]))
        file_of_solution.close()

print(values[200:205], type(values[1]))

# Some estimates on the size of you domain
# mesh.geometry.x --- for the whole coordinates
# 
# x_coord = mesh.geometry.x[:,0] for only the x - ccordinates
# y_coord = x_coord = mesh.geometry.x[:,1] the y - coordinates

# print(min(mesh.geometry.x[:,0]), max(mesh.geometry.x[:,0]))

# print(min(mesh.geometry.x[:,1]),max(mesh.geometry.x[:,1]))

## We can now use the pyvista.Plotter to visualize the mesh. We visualize it 
## by showing it in 2D and warped in 3D.
## need to use the magic %matplotlib if using Jupyter and/or Ipython!


topology, cell_types, x = create_vtk_mesh(mesh, mesh.topology.dim)
grid = pyvista.UnstructuredGrid(topology, cell_types, x)


## need to use the magic %matplotlib if using Jupyter and/or Ipython!
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("fundamentals_mesh.png")


### Plotting a function using pyvista

## We want to plot the solution uh. As the function space used to 
# defined the mesh is disconnected from the function space defining the mesh,
#  we create a mesh based on the dof coordinates for the function space V. 
# We use dolfinx.plot.create_vtk_mesh with the function space as input to 
# create a  mesh with mesh geometry based on the dof coordinates.

u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)

grid_uh = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
grid_uh.point_data["u"] = uh.x.array.real
grid_uh.set_active_scalars("u")

u_plotter = pyvista.Plotter()
u_plotter.add_mesh(grid_uh, show_edges=True)
u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()
else:
    figure = plotter.screenshot("fundamentals_mesh.png")

#print(uh.x.array.real)--for the solution values.


### We can also warp the mesh by scalar to make use of the 3D plotting.
u_plotter = pyvista.Plotter()
warped = grid_uh.warp_by_scalar()
plotter2 = pyvista.Plotter()
plotter2.add_mesh(warped, show_edges=True, show_scalar_bar=True)
if not pyvista.OFF_SCREEN:
    plotter2.show()
else:
    figure = plotter.screenshot("fundamentals_mesh.png")
