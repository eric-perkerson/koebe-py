import numpy as np
import pyvista
from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.plot import create_vtk_mesh

from ufl import (SpatialCoordinate, TestFunction, TrialFunction,
                 dx, grad, inner)

from petsc4py.PETSc import ScalarType
import faulthandler; faulthandler.enable()

from mesh_conversion import dolfinx_read_xdmf
#####################

from mpi4py import MPI # new
import meshio

#import gmsh
######################


#from dolfinx.io import gmshio
#mesh, cell_markers, facet_markers =gmshio.read_from_msh("regions/test2/A.msh", MPI.COMM_WORLD, gdim=2)


#mesh, cell_markers, facet_markers = gmshio.read_from_msh("regions/test2/A.msh", MPI.COMM_WORLD, gdim=2)

import meshio
def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, 
                           cell_data={"name_to_read":[cell_data]})
    return out_mesh



#gmsh.initialize()
proc = MPI.COMM_WORLD.rank 

if proc == 0:
    # Read in mesh
    msh = meshio.read("regions/test2/two_holes.msh")
   
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

left_facets_1 = ft.find(1)
left_dofs_1 = locate_dofs_topological(V, mesh.topology.dim-1, left_facets_1)
bcs_1 = dirichletbc(ScalarType(0), left_dofs_1, V)

left_facets_2 = ft.find(2)
left_dofs_2 = locate_dofs_topological(V, mesh.topology.dim-1, left_facets_2)
bcs_2 = dirichletbc(ScalarType(1), left_dofs_2, V)

#left_facets_3 = ft.find(3)
#left_dofs_3 = locate_dofs_topological(V, mesh.topology.dim-1, left_facets_3)
#bcs_3 = dirichletbc(ScalarType(0), left_dofs_3, V)


u, v = TrialFunction(V), TestFunction(V)
a = inner(grad(u), grad(v)) * dx
x = SpatialCoordinate(mesh)
L = Constant(mesh, ScalarType(1)) * v * dx

problem = LinearProblem(a, L, bcs=[bcs_1,bcs_2], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# As the dolfinx.MeshTag contains a value for every cell in the
# geometry, we can attach it directly to the grid

topology, cell_types, x = create_vtk_mesh(mesh, mesh.topology.dim)
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
grid.cell_data["Marker"] = ct.values[ct.indices<num_local_cells]
grid.set_active_scalars("Marker")

p = pyvista.Plotter(window_size=[800, 800])
p.add_mesh(grid, show_edges=True)
if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure = p.screenshot("subdomains_unstructured.png")