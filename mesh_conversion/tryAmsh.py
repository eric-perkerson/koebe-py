#import numpy as np
#import faulthandler; faulthandler.enable()
#from mesh_conversion import dolfinx_read_xdmf

#import faulthandler; faulthandler.enable()

#from mesh_conversion import dolfinx_read_xdmf
#gmsh.initialize()

import pyvista
from dolfinx.fem import (Constant, dirichletbc, Function , FunctionSpace, 
                         locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile

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
    msh = meshio.read("regions/test3/test3.1.msh")
   
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



V = FunctionSpace(mesh, ("CG", 1))
u_bc = Function(V)


#
# print(V.tabulate_dof_coordinates())
#

### Define the boundaries and boundary conditions.
left_facets_1 = ft.find(1)
left_dofs_1 = locate_dofs_topological(V, mesh.topology.dim-1, left_facets_1)
bcs_1 = dirichletbc(ScalarType(10), left_dofs_1, V)

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

# print(min(mesh.geometry.x[:,0]), max(mesh.geometry.x[:,0]))

# print(min(mesh.geometry.x[:,1]),max(mesh.geometry.x[:,1]))


# If you want to open with Paraview
 
# with XDMFFile(mesh.comm, "result_tryamsh.xdmf","w") as xdmf:
#        xdmf.write_mesh(mesh)
#        xdmf.write_function(uh)

 
# As the dolfinx.MeshTag contains a value for every cell in the
# geometry, we can attach it directly to the grid


topology, cell_types, x = create_vtk_mesh(mesh, mesh.topology.dim)
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
grid.cell_data["Marker"] = ct.values[ct.indices<num_local_cells]
grid.set_active_scalars("Marker")

# p = pyvista.Plotter(window_size=[800,800])
# p.add_mesh(grid, show_edges=True)
# if not pyvista.OFF_SCREEN:
#     p.show()
# else:
#     figure = p.screenshot("subdomains_unstructured.png")


grid_uh = pyvista.UnstructuredGrid(*create_vtk_mesh(V))
grid_uh.point_data["u"] = uh.x.array.real
grid_uh.set_active_scalars("u")

#print(uh.x.array.real)

p2 = pyvista.Plotter(window_size=[800, 800])
p2.add_mesh(grid_uh, show_edges=True)
if not pyvista.OFF_SCREEN:
    p2.show()
else:
    p2.screenshot("unstructured_u.png")
