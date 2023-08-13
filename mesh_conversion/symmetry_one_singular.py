"""
An example to create one singular vertex of max index
"""

import pyvista

from dolfinx.fem import (Constant, dirichletbc, Function , FunctionSpace, 
                         locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile

from dolfinx import plot
from dolfinx.plot import create_vtk_mesh

from ufl import (SpatialCoordinate, TestFunction, TrialFunction,
                 dx, grad, inner)

from petsc4py.PETSc import ScalarType

from mpi4py import MPI # new

import meshio

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
    meshio.write("mesh.xdmf", triangle_mesh)
    meshio.write("mt.xdmf", line_mesh)

# This creates the meshtags and topology of the mesh from the xdmf files above

with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(mesh, name="Grid")
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim-1)
with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "r") as xdmf:
    ft = xdmf.read_meshtags(mesh, name="Grid")

V = FunctionSpace(mesh, ("CG", 1))

#
# print(V.tabulate_dof_coordinates())
#

### Define the boundaries and boundary conditions.

left_facets_1 = ft.find(1)
left_dofs_1 = locate_dofs_topological(V, mesh.topology.dim-1, left_facets_1)
bcs_1 = dirichletbc(ScalarType(2), left_dofs_1, V)

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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

x_coord = mesh.geometry.x[:,0]
y_coord = mesh.geometry.x[:,1]


z_coord= uh.x.array

fig = plt.figure(figsize=(12,8))
gs = mpl.gridspec.GridSpec(2, 2)

ax1 = fig.add_subplot(gs[0:,0],projection="3d")
ax1.plot_trisurf(x_coord,y_coord,z_coord)
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")
ax1.set_zlabel("$z$")
ax1.set_title("Approximate surface")

ax2=fig.add_subplot(gs[0,1])
levels = np.arange(0.0,0.8, 0.0125)
ax2.tricontour(x_coord,y_coord,z_coord,levels=levels)
levels = np.arange(0.0, 2.0, 0.25)
ax2.set_xlabel("$x$")
ax2.set_ylabel("$y$")
ax2.set_title("Approximate contour in $[-0.5,0.5]^2$")
ax2.set_ybound(-0.5,0.5)
ax2.set_xbound(-0.5,0.5)

ax3=fig.add_subplot(gs[1,1:])
levels = np.arange(0.0,0.8, 0.0125)
ax3.tricontour(x_coord,y_coord,z_coord,levels=levels)
levels = np.arange(0.0, 2.0, 0.25)
ax3.set_xlabel("$x$")
ax3.set_ylabel("$y$")
ax3.set_title("Approximate contours")


fig.tight_layout()
plt.show()

#TODO: Make this into a nice array of figures!




### Reading and Printing the solution to a file

# Write the solution, uh, to a file, entry per line

with open('regions/3_fold_sym/solution_fenicsx.txt','w') as output_file:
     for entry in uh.x.array:
#         output_file.write(str(entry) + '\n')
          output_file.write(f"{entry}\n")


# open file in read mode and copy solution values at node to 
# values[], a list

file_of_solution = open('regions/3_fold_sym/solution_fenicsx.txt','r')
values=[]
for line in file_of_solution.readlines():
    #print(line, end='')
    values.append(float(line[0:-1]))
file_of_solution.close()

print(values[200:205])

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
#print(uh.x.array.real)


### We can also warp the mesh by scalar to make use of the 3D plotting.

warped = grid_uh.warp_by_scalar()
plotter2 = pyvista.Plotter()
plotter2.add_mesh(warped, show_edges=True, show_scalar_bar=True)
if not pyvista.OFF_SCREEN:
    plotter2.show()