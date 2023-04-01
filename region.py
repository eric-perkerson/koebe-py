import numpy as np


class Region(object):
    """Defines a region object which contains the information about the boundary components and one point in each hole."""
    def __init__(self, components, points_in_holes=None):
        self.components = components
        self.points_in_holes = None
        if not points_in_holes:
            all_means = [list(np.mean(np.array(c), axis=0)) for c in components]
            self.points_in_holes = all_means[1:]
        else:
            self.points_in_holes = points_in_holes

    def edges_by_component(self):
        """Get the edges by component"""
        edges = [[] for i in range(len(self.components))]
        counter = 1
        for i in range(len(self.components)):
            for j in range(len(self.components[i]) - 1):
                edges[i].append([counter, counter + 1])
                counter += 1
            edges[i].append([counter, edges[i][0][0]])
            counter += 1
        return edges

    def write(self, stream):
        component_lengths = list(map(len, self.components))
        num_vert = sum(component_lengths)
        num_edge = num_vert
        num_holes = len(self.points_in_holes)
        edges_by_components = self.edges_by_component()

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
            counter = counter + 1
        stream.write(str(num_holes) + '\n')
        for i in range(num_holes):
            stream.write(str(i) + ' ' + str(self.points_in_holes[i][0]) + ' ' + str(self.points_in_holes[i][1]) + '\n')

    @staticmethod
    def read(file_name):
        x_list = []
        y_list = []
        bdry_marker_list = []
        with open(file_name, 'r') as f:
            counter = 0
            for line in f:
                if counter == 0:
                    num_vertices, dimension, num_attributes, num_bdry_markers = line.split()
                    assert dimension == 2
                    assert num_bdry_markers == 0 or num_bdry_markers == 1
                else:
                    index, x, y, bdry_marker = line.split()
                    x_list.append(x)
                    y_list.append(y)
                    bdry_marker_list.append(bdry_marker)
                    assert index == counter
                counter += 1


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
    with open(file_name, 'r') as f:
        for line_index, line in enumerate(f):
            if line.strip()[0] == '#':  # Skip comment lines
                continue
            if line_index == 0:  # First real line in file contains header information
                num_vertices, dimension, num_attributes, num_bdry_markers = list(map(int, line.split()))
                vertices = np.zeros((num_vertices, 2))
                bdry_marker_array = np.zeros(num_vertices, dtype=int)
                assert dimension == 2
                assert num_attributes == 0
                assert num_bdry_markers == 0 or num_bdry_markers == 1
            else:  # Following lines contain one vertex/node each
                one_based_index, x, y, bdry_marker = list(map(float, line.split()))
                zero_based_index = int(one_based_index) - 1
                vertices[zero_based_index, 0] = x
                vertices[zero_based_index, 1] = y
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
    with open(file_name, 'r') as f:
        for line_index, line in enumerate(f):
            if line.strip()[0] == '#':  # Skip comment lines
                continue
            if line_index == 0:  # First real line in file contains header information
                num_triangles, nodes_per_triangle, num_attributes = list(map(int, line.split()))
                triangles = np.zeros((num_triangles, 3), dtype=int)
                assert num_attributes == 0
            else:  # Following lines contain one vertex/node each
                one_based_index, v1, v2, v3 = list(map(int, line.split()))
                zero_based_index = int(one_based_index) - 1
                triangles[zero_based_index, 0] = v1
                triangles[zero_based_index, 1] = v2
                triangles[zero_based_index, 2] = v3
    return triangles - 1  # Correct for 1-based indexing


def read_poly(file_name):
    """Reads a .poly file and returns numpy arrays representing the triangulation data

    Parameters
    ----------
    file_name : str
        The file name to read
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        # Read the header line
        num_vertices, dimension, num_attributes, num_vertex_boundary_markers = list(map(int, f.readline().split()))
        if num_vertex_boundary_markers != 1:
            raise Warning('Number of vertex boundary markers should be 1')
        if dimension != 2:
            raise Exception('Dimension must be 2')
        if num_attributes != 0:
            raise Exception('Number of attributes must be 0')
        vertices = np.zeros((num_vertices, 2), dtype=float)
        vertex_boundary_markers = np.zeros(num_vertices, dtype=int)

        for i in range(num_vertices):
            line = f.readline()
            one_based_index_str, v1_str, v2_str, boundary_marker_str = line.split()
            zero_based_index = int(one_based_index_str) - 1
            vertices[zero_based_index, 0] = v1_str
            vertices[zero_based_index, 1] = v2_str
            vertex_boundary_markers[zero_based_index] = int(boundary_marker_str)

        # Read edge header line
        num_edges, num_edge_boundary_markers = list(map(int, f.readline().split()))
        if num_edge_boundary_markers != 1:
            raise Warning('Number of edge boundary markers should be 1')
        edges = np.zeros((num_edges, 2), dtype=int)
        edge_boundary_markers = np.zeros(num_edges, dtype=int)
        for i in range(num_edges):
            one_based_index, e1, e2, boundary_marker = list(map(int, f.readline().split()))
            zero_based_index = one_based_index - 1
            edges[zero_based_index, 0] = e1
            edges[zero_based_index, 1] = e2
            edge_boundary_markers[zero_based_index] = boundary_marker

        # Read hole header line
        num_holes = int(f.readline().split())
        points_in_holes = np.zeros((num_holes, 2), dtype=float)
        for i in range(num_holes):
            one_based_index_str, x_coordinate_str, y_coordinate_str = f.readline().split()
            one_based_index = int(one_based_index_str)
            zero_based_index = one_based_index - 1
            x_coordinate = float(x_coordinate_str)
            y_coordinate = float(y_coordinate_str)
            points_in_holes[zero_based_index, 0] = x_coordinate
            points_in_holes[zero_based_index, 1] = y_coordinate

        vertices = vertices - 1  # Correct for 1-based indexing
        edges = vertices - 1  # Correct for 1-based indexing
        return vertices, vertex_boundary_markers, edges, edge_boundary_markers, points_in_holes


class Region2(object):
    """Defines a region object which contains the information about the boundary components and one point in each hole."""
    def __init__(self, coordinates, vertex_boundary_markers, edges, edge_boundary_markers, points_in_holes):
        self.coordinates = coordinates
        self.vertex_boundary_markers = vertex_boundary_markers
        self.edges = edges
        self.edge_boundary_markers = edge_boundary_markers
        self.points_in_holes = points_in_holes

    def get_components(self):
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
            coordinate_components[i] = np.vstack(list(map(lambda x: self.coordinates[x][np.newaxis, :], component)))

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
        with open(file_name, 'r') as f:
            lines = f.readlines()

        # Read the vertex header line
        num_vertices, dimension, num_attributes, num_bdry_markers = list(map(int, lines[0].split()))
        vertices = np.zeros((num_vertices, 2))
        vertex_boundary_markers = np.zeros(num_vertices, dtype=int)
        if dimension != 2:
            raise ValueError
        if num_attributes != 0:
            raise ValueError
        if num_bdry_markers != 0 and num_bdry_markers != 1:
            raise ValueError

        for line_index, line in enumerate(lines[1:num_vertices + 1]):
            split_line = line.split()
            # vertex_index = int(split_line[0])
            # print(vertex_index)
            vertices[line_index, 0] = float(split_line[1])
            vertices[line_index, 1] = float(split_line[2])
            vertex_boundary_markers[line_index] = int(split_line[3])

        # Read the segment/edge header line
        num_edges, num_edge_boundary_markers = list(map(int, lines[num_vertices + 1].split()))
        edges = np.zeros((num_edges, 2), dtype=int)
        edge_boundary_markers = np.zeros(num_edges, dtype=int)
        for line_index, line in enumerate(lines[num_vertices + 2:num_vertices + num_edges + 2]):
            split_line = line.split()
            # edge_index = int(split_line[0])
            # print(edge_index)
            edges[line_index, 0] = int(split_line[1])
            edges[line_index, 1] = int(split_line[2])
            edge_boundary_markers[line_index] = int(split_line[3])

        # Read the hole header line
        num_holes = int(lines[num_vertices + num_edges + 2].split()[0])
        points_in_holes = np.zeros((num_holes, 2))
        for line_index, line in enumerate(lines[num_vertices + num_edges + 3:num_vertices + num_edges + num_holes + 3]):
            split_line = line.split()
            # hole_index = int(split_line[0])
            # print(hole_index)
            points_in_holes[line_index, 0] = float(split_line[1])
            points_in_holes[line_index, 1] = float(split_line[2])

        edges_zero_indexed = edges - 1
        region = Region2(vertices, vertex_boundary_markers, edges_zero_indexed, edge_boundary_markers, points_in_holes)
        return region
