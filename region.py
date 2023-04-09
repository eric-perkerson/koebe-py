"""Defines the region object, which contains information about the boundary components and holes.
This object corresponds to a planar straight line graph (PSLG), which is represented in a .poly
file."""
import numpy as np


def edges_by_component(components):
    """Get the edges by component"""
    edges = [[] for _ in range(len(components))]
    counter = 1
    for i, component in enumerate(components):
        for _ in range(len(component) - 1):
            edges[i].append([counter, counter + 1])
            counter += 1
        edges[i].append([counter, edges[i][0][0]])
        counter += 1
    return edges


class Region:
    """Defines a region object which contains the information about the boundary components and one
    point in each hole."""
    def __init__(
        self,
        coordinates,
        vertex_boundary_markers,
        edges,
        edge_boundary_markers,
        points_in_holes,
        components=None
    ):
        self.coordinates = coordinates
        self.vertex_boundary_markers = vertex_boundary_markers
        self.edges = edges
        self.edge_boundary_markers = edge_boundary_markers
        self.points_in_holes = points_in_holes
        if components is None:
            self.components = self.build_components()
        else:
            self.components = components

    @staticmethod
    def region_from_components(components, points_in_holes=None):
        coordinates = np.concatenate(
            [np.array(item).reshape((1, -1)) for sublist in components for item in sublist],
            axis=0
        )
        vertex_boundary_markers = np.concatenate(
            [np.full(len(component), i) for i, component in enumerate(components)]
        )
        edges = [item for sublist in edges_by_component(components) for item in sublist]
        edge_boundary_markers = vertex_boundary_markers.copy()
        if points_in_holes is None:
            all_means = [list(np.mean(np.array(component), axis=0)) for component in components]
            points_in_holes = all_means[1:]
        return Region(coordinates, vertex_boundary_markers, edges, edge_boundary_markers, points_in_holes, components)

    def write(self, stream):
        component_lengths = list(map(len, self.components))
        num_vert = sum(component_lengths)
        num_edge = num_vert
        num_holes = len(self.points_in_holes)
        edges_by_components = edges_by_component(self.components)

        dimension = 2
        num_attributes = 0
        num_bdry_markers = 1

        counter = 0  # Counts vertices first
        stream.write(
            str(num_vert) + ' ' + str(dimension) + ' ' + str(num_attributes) + ' ' + str(num_bdry_markers) + '\n'
        )
        for i in range(len(self.components)):
            bdry_marker = i + 1  # Boundary marker 1 for exterior boundary, 2, 3, etc. for interior boundary components
            for v in range(component_lengths[i]):
                counter = counter + 1
                stream.write(
                    str(counter) + ' ' + str(float(self.components[i][v][0])) + ' '
                    + str(float(self.components[i][v][1])) + ' ' + str(bdry_marker) + '\n'
                )
        counter = 0  # Counts edges now
        stream.write(str(num_edge) + ' ' + str(num_bdry_markers) + '\n')
        for i in range(len(self.components)):
            bdry_marker = i + 1
            for e in edges_by_components[i]:
                counter = counter + 1
                stream.write(str(counter) + ' ' + str(e[0]) + ' ' + str(e[1]) + ' ' + str(bdry_marker) + '\n')
        stream.write(str(num_holes) + '\n')
        for i in range(num_holes):
            stream.write(str(i) + ' ' + str(self.points_in_holes[i][0]) + ' ' + str(self.points_in_holes[i][1]) + '\n')

    def build_components(self):
        """Build the boundary components of the region"""
        components_raw = [[]]
        current_component = 0
        for edge in self.edges:
            components_raw[current_component].append(edge[0])
            if edge[0] > edge[1]:
                components_raw.append([])
                current_component += 1
        components = components_raw[:-1]

        coordinate_components = []
        for i, component in enumerate(components):
            coordinate_components.append([])
            coordinate_components[i] = np.vstack(
                list(map(lambda x: self.coordinates[x][np.newaxis, :], component))
            )

        return coordinate_components

    @staticmethod
    def read_poly(file_name):
        """Reads a poly file

        Parameters
        ----------
        file_name : str
            The poly file name to read

        Returns
        -------
        tuple
            vertices, vertex_boundary_markers, edges, edge_boundary_markers, points_in_holes

        Raises
        ------
        ValueError
            If the poly file does not have the expected parameters
        """
        (
            vertices,
            vertex_boundary_markers,
            edges_zero_indexed,
            edge_boundary_markers,
            points_in_holes
        ) = read_poly(file_name)
        region = Region(
            vertices,
            vertex_boundary_markers,
            edges_zero_indexed,
            edge_boundary_markers,
            points_in_holes
        )
        return region


def read_node(file_name):
    """Reads a .node file and returns the vertices and boundary markers

    Parameters
    ----------
    file_name : str
        The file name to read

    Returns
    -------
    vertices : np.array
        2D array of x and y coordinates of the vertices in the .node file
    bdry_marker_array : np.array
        1D array of boundary markers of the vertices in the .node file
    """
    with open(file_name, 'r', encoding='utf-8') as file:
        for line_index, line in enumerate(file):
            if line.strip()[0] == '#':  # Skip comment lines
                continue
            if line_index == 0:  # First real line in file contains header information
                (
                    num_vertices,
                    dimension,
                    num_attributes,
                    num_bdry_markers
                ) = list(map(int, line.split()))
                vertices = np.zeros((num_vertices, 2))
                bdry_marker_array = np.zeros(num_vertices, dtype=int)
                if dimension != 2:
                    raise ValueError('Dimension must be 2')
                if num_attributes != 0:
                    raise ValueError('The number of boundary markers must be either 0 or 1')
                if num_bdry_markers not in [0, 1]:
                    raise ValueError('The number of boundary markers must be either 0 or 1')
            else:  # Following lines contain one vertex/node each
                (
                    one_based_index,
                    x_coordinate,
                    y_coordinate,
                    bdry_marker
                ) = list(map(float, line.split()))
                zero_based_index = int(one_based_index) - 1
                vertices[zero_based_index, 0] = x_coordinate
                vertices[zero_based_index, 1] = y_coordinate
                bdry_marker_array[zero_based_index] = int(bdry_marker)
    return vertices, bdry_marker_array


def read_ele(file_name):
    """Reads a .ele file and returns the triangle array

    Parameters
    ----------
    file_name : str
        The file name to read

    Returns
    -------
    triangles : np.array
        2D array where each row is the three indices making a triangle in the .ele file
    """
    with open(file_name, 'r', encoding='utf-8') as file:
        for line_index, line in enumerate(file):
            if line.strip()[0] == '#':  # Skip comment lines
                continue
            if line_index == 0:  # First non-comment line in file contains header information
                num_triangles, _, num_attributes = list(map(int, line.split()))
                triangles = np.zeros((num_triangles, 3), dtype=int)
                if num_attributes != 0:
                    raise ValueError('num_attributes must be 0')
            else:  # Following lines contain one vertex/node each
                one_based_index, v_1, v_2, v_3 = list(map(int, line.split()))
                zero_based_index = int(one_based_index) - 1
                triangles[zero_based_index, 0] = v_1
                triangles[zero_based_index, 1] = v_2
                triangles[zero_based_index, 2] = v_3
    return triangles - 1  # Correct for 1-based indexing


def read_poly(file_name):
    """Reads a .poly file and returns numpy arrays representing the triangulation data

    Parameters
    ----------
    file_name : str
        The file name to read
    """
    with open(file_name, 'r', encoding='utf-8') as file:
        # Read the header line
        (
            num_vertices,
            dimension,
            num_attributes,
            num_vertex_boundary_markers
        ) = list(map(int, file.readline().split()))
        if dimension != 2:
            raise Exception('Dimension must be 2')
        if num_vertex_boundary_markers != 1:
            raise Warning('Number of vertex boundary markers should be 1')
        if num_attributes != 0:
            raise Exception('Number of attributes must be 0')
        vertices = np.zeros((num_vertices, 2), dtype=float)
        vertex_boundary_markers = np.zeros(num_vertices, dtype=int)

        for _ in range(num_vertices):
            line = file.readline()
            (
                one_based_index_str,
                coordinate_1_str,
                coordinate_2_str,
                boundary_marker_str
            ) = line.split()
            zero_based_index = int(one_based_index_str) - 1
            vertices[zero_based_index, 0] = float(coordinate_1_str)
            vertices[zero_based_index, 1] = float(coordinate_2_str)
            vertex_boundary_markers[zero_based_index] = int(boundary_marker_str)

        # Read edge header line
        num_edges, num_edge_boundary_markers = list(map(int, file.readline().split()))
        if num_edge_boundary_markers != 1:
            raise Warning('Number of edge boundary markers should be 1')
        edges = np.zeros((num_edges, 2), dtype=int)
        edge_boundary_markers = np.zeros(num_edges, dtype=int)
        for _ in range(num_edges):
            one_based_index, e_1, e_2, boundary_marker = list(map(int, file.readline().split()))
            zero_based_index = one_based_index - 1
            edges[zero_based_index, 0] = e_1
            edges[zero_based_index, 1] = e_2
            edge_boundary_markers[zero_based_index] = boundary_marker

        # Read hole header line
        num_holes = int(file.readline().split())
        points_in_holes = np.zeros((num_holes, 2), dtype=float)
        for _ in range(num_holes):
            one_based_index_str, x_coordinate_str, y_coordinate_str = file.readline().split()
            one_based_index = int(one_based_index_str)
            zero_based_index = one_based_index - 1
            x_coordinate = float(x_coordinate_str)
            y_coordinate = float(y_coordinate_str)
            points_in_holes[zero_based_index, 0] = x_coordinate
            points_in_holes[zero_based_index, 1] = y_coordinate

        vertices = vertices - 1  # Correct for 1-based indexing
        edges = vertices - 1  # Correct for 1-based indexing
        return vertices, vertex_boundary_markers, edges, edge_boundary_markers, points_in_holes
