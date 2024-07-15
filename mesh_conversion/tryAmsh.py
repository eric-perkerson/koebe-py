"""
An example with two holes--one singular vertex

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



def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, 
                           cell_data={"name_to_read":[cell_data]})
    return out_mesh


proc = MPI.COMM_WORLD.rank 

if proc == 0:
    # Read in mesh
    #msh = meshio.read("regions/test3/test3.output.msh")
    msh = meshio.read("regions/vertex18/vertex18.output.msh")
    
    
    # Create and save one file for the mesh, and one file for the facets 

    triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
    line_mesh = create_mesh(msh, "line", prune_z=True)
    meshio.write("mesh.xdmf", triangle_mesh)
    meshio.write("mt.xdmf", line_mesh)


with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(mesh, name="Grid")
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim-1)
with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "r") as xdmf:
    ft = xdmf.read_meshtags(mesh, name="Grid")


### Defining the finite element function space

V = FunctionSpace(mesh, ("CG", 1))

#u_bc = Function(V)


#
# print(V.tabulate_dof_coordinates())
#

### Define the boundaries and boundary conditions.
left_facets_1 = ft.find(1)
left_dofs_1 = locate_dofs_topological(V, mesh.topology.dim-1, left_facets_1)
bcs_1 = dirichletbc(ScalarType(100), left_dofs_1, V)

left_facets_2 = ft.find(2)
left_dofs_2 = locate_dofs_topological(V, mesh.topology.dim-1, left_facets_2)
bcs_2 = dirichletbc(ScalarType(0), left_dofs_2, V)

left_facets_3 = ft.find(3)
left_dofs_3 = locate_dofs_topological(V, mesh.topology.dim-1, left_facets_3)
bcs_3 = dirichletbc(ScalarType(0), left_dofs_3, V)



### Set the trial functions, the bi-linear and linear forms, problem to solve
### and the solution.

u, v = TrialFunction(V), TestFunction(V)
a = inner(grad(u), grad(v)) * dx
x = SpatialCoordinate(mesh)
L = Constant(mesh, ScalarType(0)) * v * dx

problem = LinearProblem(a, L, bcs=[bcs_1,bcs_2,bcs_3], 
                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()


### Reading and Printing the solution to a file

# Write the solution, uh, to a file, entry per line

with open('regions/vertex18/solution_fenicsx.txt','w') as output_file:
     for entry in uh.x.array:
#         output_file.write(str(entry) + '\n')
          output_file.write(f"{entry}\n")


# open file in read mode and copy solution values at node to 
# values[], a list


# TODO: add a try here and a message!
try:
      file_of_solution = open('regions/vertex18/solution_fenicsx.txt','r')
except:
       print("This file can't be found. Check the folder!\n")
else:
#display content
    values=[]
    for line in file_of_solution.readlines():
    #print(line, end='')
        values.append(float(line[0:-1]))
    file_of_solution.close()

#print some random values
print(values[200:205], type(values[202]))


# print(min(mesh.geometry.x[:,0]), max(mesh.geometry.x[:,0]))

# print(min(mesh.geometry.x[:,1]),max(mesh.geometry.x[:,1]))


# If you want to open with Paraview
 
# with XDMFFile(mesh.comm, "result_tryAmsh.xdmf","w") as xdmf:
#        xdmf.write_mesh(mesh)
#        xdmf.write_function(uh)

 
# As the dolfinx.MeshTag contains a value for every cell in the
# geometry, we can attach it directly to the grid

### We will visualizing the mesh using pyvista, 
# an interface to the VTK toolkit. We start by 
# converting the mesh to a format that can be used with pyvista. 
# To do this we use the function dolfinx.plot.create_vtk_mesh. 
# The first step is to create an unstructured grid that can be used by pyvista.
###



topology, cell_types, x = create_vtk_mesh(mesh, mesh.topology.dim)
grid = pyvista.UnstructuredGrid(topology, cell_types, x)

## We can now use the pyvista.Plotter to visualize the mesh. We visualize it 
## by showing it in 2D and warped in 3D.
## need to use the magic %matplotlib if using Jupyter and/or Ipython!
%matplotlib
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




####
### External post-processing
###
#For post-processing outside the python code, 
# it is suggested to save the solution to file using either 
# dolfinx.io.VTXWriter or dolfinx.io.XDMFFile and using Paraview. 
# This is especially suggested for 3D visualization.

from dolfinx import io

###TODO: Save files for postprocessing this tomorrow!

with io.VTXWriter(mesh.comm, "output.bp", [uh]) as vtx:
    vtx.write(0.0)
with XDMFFile(mesh.comm, "output.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(uh)



#print(min(mesh.geometry.x[:,0]), max(mesh.geometry.x[:,0]))

# print(min(mesh.geometry.x[:,1]),max(mesh.geometry.x[:,1]))
#print(uh.x.array.real)
