import meshio
import gmsh
import numpy as np
import collections
import argparse
from dolfinx.io import XDMFFile
from mpi4py import MPI


def read_triangle(poly_file, ele_file, node_file=None):
    """
    poly-file reader, that creates a python dictionary
    with information about vertices, edges and holes.
    it assumes that vertices have no attributes
    no regional attributes or area constraints are parsed.

    if verticies are not given in the poly-file, they must be given in a node-file.
    if node-file is not given, and vertices are not given in the poly-file, the function will search for a node-file with the same name as the poly-file, but with the extension .node
    """

    output = {'vertices': None, 'holes': None, 'segments': None, 'triangles': None}

    # open file and store lines in a list
    file = open(poly_file, 'r')
    lines = file.readlines()
    file.close()
    lines = [x.strip('\n').split() for x in lines]

    # clean up lines (remove empty lines and comments)
    lines = [x for x in lines if x != []]
    lines = [x for x in lines if x[0] != '#']

    # Divide lines into vertices, segments and holes
    vertex_lines, segment_lines, hole_lines = [], [], []

    # handle vertices
    n_vertices, dimension, attr, bdry_markers = [int(x) for x in lines[0]]
    if n_vertices == 0:
        if node_file is None:
            # no node-file given, so search for a node-file with the same name as the poly-file, but with the extension .node
            node_file = poly_file[:-4] + 'node'

        # open file and store lines in a list
        try:
            file = open(node_file, 'r')
            vertex_lines = file.readlines()
            file.close()
            vertex_lines = [x.strip('\n').split() for x in vertex_lines]
            vertex_lines = [x for x in vertex_lines if x != []]
            vertex_lines = [x for x in vertex_lines if x[0] != '#']
        except:
            raise Exception(f"no vertices given in poly-file and no node-file found at {node_file}")

        # append vertex lines to lines, so that the rest of the code can be used as is.
        lines = vertex_lines + lines[1:]

    # vertex stats
    n_vertices, dimension, attr, bdry_markers = [int(x) for x in lines[0]]
    vertex_lines = lines[1:n_vertices+1]

    # segment stats
    n_segments, bdry_markers = [int(x) for x in lines[n_vertices+1]]
    segment_lines = lines[n_vertices+2:n_vertices+n_segments+2]

    # hole stats
    n_holes = int(lines[n_segments+n_vertices+2][0])
    hole_lines = lines[n_segments + n_vertices + 3:n_segments + n_vertices + 3 + n_holes]

    # store vertices
    vertices = []
    for line in vertex_lines:
        values = [float(x) for x in line] # read as tuple in case of boundary markers
        vertices.append(values[1:]) # ignore label
    output['vertices'] = np.array(vertices)

    # store segments
    segments = []
    for line in segment_lines:
        values = [int(x) for x in line] # read as tuple in case of boundary markers
        values[1] -= 1 # subtract 1 to get 0-based indexing
        values[2] -= 1 # subtract 1 to get 0-based indexing
        segments.append(values[1:]) # ignore label
    output['segments'] = np.array(segments)

    # store holes
    holes = []
    for line in hole_lines:
        values = [float(x) for x in line]
        holes.append(values[1:])  # ignore label
    output['holes'] = np.array(holes)

    # Read triangles
    try:
        # open file and store lines in a list
        file = open(ele_file, 'r')
        triangle_lines = file.readlines()
        file.close()
        triangle_lines = [x.strip('\n').split() for x in triangle_lines]
        triangle_lines = [x for x in triangle_lines if x != []]
        triangle_lines = [x for x in triangle_lines if x[0] != '#']

        # store triangles
        triangles = []
        for line in triangle_lines[1:]:
            values = [int(x) for x in line]  # read as tuple in case of boundary markers
            values[1] -= 1  # subtract 1 to get 0-based indexing
            values[2] -= 1 # subtract 1 to get 0-based indexing
            values[3] -= 1  # subtract 1 to get 0-based indexing
            triangles.append(values[1:]) # ignore label
        output['triangles'] = np.array(triangles)
    except Exception:
        raise Exception("no triangles given in ele-file")

    return output


def create_xdmf(cell_type, mesh_name):
    """
    creates an xdmf file from a mesh file with the given cell type (triangle or line)
    """
    # read mesh
    mesh = meshio.read(mesh_name)

    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data('gmsh:physical', cell_type=cell_type)
    points = mesh.points[:, :2]
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data]})
    return out_mesh


def create_mesh(output_name, poly_dict):
    """
    creates a .msh file using gmsh and meshio from the verticies, segments and holes in the poly_dict.
    Also creates XDMF files for the mesh and the line facets
    """
    # initialize gmsh
    gmsh.initialize()

    # create new model
    gmsh.model.add("Grid")

    # add vertices
    points = []
    for vertex in poly_dict['vertices']:
        x, y, z = vertex[0], vertex[1], 0
        points.append(gmsh.model.geo.addPoint(x, y, z))

    # add boundary segments
    lines = collections.defaultdict(list)
    for segment in poly_dict['segments']:
        start = points[segment[0]]
        end = points[segment[1]]

        boundary = None if len(segment) == 2 else segment[2]
        line = gmsh.model.geo.addLine(start, end)

        if boundary:
            lines[boundary].append(line)

    # add triangles
    triangles = []
    for triangle in poly_dict['triangles']:
        l1 = gmsh.model.geo.addLine(points[triangle[0]], points[triangle[1]])
        l2 = gmsh.model.geo.addLine(points[triangle[1]], points[triangle[2]])
        l3 = gmsh.model.geo.addLine(points[triangle[2]], points[triangle[0]])

        loop = gmsh.model.geo.addCurveLoop([l1, l2, l3])
        triangle = gmsh.model.geo.addPlaneSurface([loop])
        triangles.append(triangle)

    # synchronize geometry before adding physical groups
    gmsh.option.set_number("Geometry.Tolerance", 1e-3)
    gmsh.model.geo.remove_all_duplicates()
    gmsh.model.geo.synchronize()

    # add physical groups
    for boundary, line_list in lines.items():
        gmsh.model.addPhysicalGroup(dim=1, tags=line_list, tag=boundary)
    gmsh.model.addPhysicalGroup(dim=2, tags=triangles, tag=0)

    # generate mesh
    gmsh.model.mesh.generate(2)

    # names of physical groups
    gmsh.model.setPhysicalName(dim=2, tag=0, name="compound_surface")
    for boundary, line_list in lines.items():
        gmsh.model.setPhysicalName(dim=1, tag=boundary, name=f"boundary_{boundary}")

    # write mesh to file
    gmsh.write(f"{output_name}.msh")

    # clear gmsh
    gmsh.clear()

    # create xdmf file
    meshio.write(f"{output_name}.xdmf", create_xdmf("triangle", f"{output_name}.msh"))
    meshio.write(f"{output_name}.facet.xdmf", create_xdmf("line", f"{output_name}.msh"))


# def plot_mesh(mesh):
#     """
#     plots a mesh using pyvista
#     """
#     pyvista.set_plot_theme("document")

#     p = pyvista.Plotter(window_size=(800, 800))
#     p.add_mesh(
#         mesh=pyvista.from_meshio(mesh),
#         scalar_bar_args={"title": "Materials"},
#         show_scalar_bar=False,
#         show_edges=True,
#     )
#     p.view_xy()
#     p.show()


def dolfinx_read_xdmf(mesh_file, facet_file):
    """
    Takes a mesh and facet file and returns a dolfinx mesh, cell tags and facet tags
    """
    with XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        cell_tags = xdmf.read_meshtags(mesh, name="Grid")
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    with XDMFFile(MPI.COMM_WORLD, facet_file, "r") as xdmf:
        facet_tags = xdmf.read_meshtags(mesh, name="Grid")

    return mesh, cell_tags, facet_tags


if __name__ == '__main__':
    # print versions of imports
    print("-------------------------------------")
    print('meshio version: ', meshio.__version__)
    print('gmsh version: ', gmsh.__version__)
    print("-------------------------------------\n")

    # create command line parser
    parser = argparse.ArgumentParser(
        description=(
            'Converts a Triangle mesh file to xdmf. If no node-file is given, the function will search for a'
            'node-file with the same name as the poly-file, but with the extension .node'
        )
    )
    parser.add_argument('-p', '--poly', help='Name of the .poly file to convert', required=True, dest='poly')
    parser.add_argument('-e', '--ele', help='Name of the .ele file to convert', required=True, dest='ele')
    parser.add_argument('-n', '--node', help='Name of the .node file to convert', required=False, dest='node')

    # parse command line arguments
    args = parser.parse_args()

    # read poly-file
    poly_file = args.poly
    node_file = args.node
    ele_file = args.ele

    mesh = read_triangle(poly_file, ele_file, node_file)

    print()
    print(f"vertices: {mesh['vertices'].shape}")
    print(f"segments: {mesh['segments'].shape}")
    print(f"holes: {mesh['holes'].shape}")
    print(f"triangles: {mesh['triangles'].shape}")
    print()

    # create mesh
    output_name = poly_file[:-5]
    create_mesh(output_name, mesh)

    # plot mesh
    # plot_mesh(meshio.read(output_name + '.msh'))
