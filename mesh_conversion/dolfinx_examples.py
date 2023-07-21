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


def boundary_example():
    mesh, ct, ft = dolfinx_read_xdmf("regions/test/test.1.xdmf", "regions/test/test.1.facet.xdmf") # REPLACE WITH CORRECT FILES

    #Q = FunctionSpace(mesh, ("DG", 0))

    # kappa = Function(Q)
    # bottom_cells = ct.find(0)
    # kappa.x.array[bottom_cells] = np.full_like(bottom_cells, 1, dtype=ScalarType)
    # top_cells = ct.find(0)
    # kappa.x.array[top_cells]  = np.full_like(top_cells, 35, dtype=ScalarType)

    V = FunctionSpace(mesh, ("CG", 1))
    u_bc = Function(V)
    left_facets = ft.find(1) # BOUNDARY CHOSEN
    left_dofs = locate_dofs_topological(V, mesh.topology.dim-1, left_facets)
    bcs = [dirichletbc(ScalarType(1), left_dofs, V)]

    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    x = SpatialCoordinate(mesh)
    L = Constant(mesh, ScalarType(1)) * v * dx

    problem = LinearProblem(a, L, bcs=bcs, 
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    topology, cell_types, x = create_vtk_mesh(mesh, mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    grid.cell_data["Marker"] = ct.values[ct.indices<num_local_cells]
    grid.set_active_scalars("Marker")

    grid_uh = pyvista.UnstructuredGrid(*create_vtk_mesh(V))
    grid_uh.point_data["u"] = uh.x.array.real
    grid_uh.set_active_scalars("u")
    p2 = pyvista.Plotter(window_size=[800, 800])
    p2.add_mesh(grid_uh, show_edges=True)
    if not pyvista.OFF_SCREEN:
        p2.show()
    else:
        p2.screenshot("unstructured_u.png")



if __name__ == "__main__":
    boundary_example()