"""This module contains the Triangulation object."""
import numpy as np
import numba
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import networkx as nx
from collections import defaultdict
from pathlib import Path
from region import read_node, read_ele, Region
import faulthandler
faulthandler.enable()


@numba.jit
def triangle_circumcenter(a_x, a_y, b_x, b_y, c_x, c_y):
    """Calculate the coordinates of the circumcenter of a triangle given the coordinates of its
    vertices"""
    multiplier = 1 / (
        2 * (
            a_x * (b_y - c_y)
            + b_x * (c_y - a_y)
            + c_x * (a_y - b_y)
        )
    )

    norm_a = a_x * a_x + a_y * a_y
    norm_b = b_x * b_x + b_y * b_y
    norm_c = c_x * c_x + c_y * c_y

    circumcenter_x = multiplier * (
        norm_a * (b_y - c_y)
        + norm_b * (c_y - a_y)
        + norm_c * (a_y - b_y)
    )
    circumcenter_y = multiplier * (
        norm_a * (c_x - b_x)
        + norm_b * (a_x - c_x)
        + norm_c * (b_x - a_x)
    )
    return circumcenter_x, circumcenter_y


def triangle_area(triangle_coordinates):
    """Returns the area of a triangle given the triangle coordinates of a triangle using Heron's
    formula.

    Parameters
    ----------
    triangle_coordinates : np.array
        A size (3, 2) matrix with the coordinates of each vertex in the triangle

    Returns
    -------
    area
        The area of the triangle
    """
    length_1 = np.linalg.norm(triangle_coordinates[0, :] - triangle_coordinates[1, :])
    length_2 = np.linalg.norm(triangle_coordinates[1, :] - triangle_coordinates[2, :])
    length_3 = np.linalg.norm(triangle_coordinates[2, :] - triangle_coordinates[0, :])
    semi_perimeter = (length_1 + length_2 + length_3) / 2
    return np.sqrt(
        semi_perimeter
        * (semi_perimeter - length_1)
        * (semi_perimeter - length_2)
        * (semi_perimeter - length_3)
    )


def pad_polygons_to_matrix(polygons):
    """Pad a list of polygons with zeros to make a matrix of the same size

    Parameters
    ----------
    polygons : list of lists

    Returns
    -------
    padded_polygons : np.array of shape (n_polygons, max_polygon_length)
    """

    max_polygon_length = max([len(polygon) for polygon in polygons])
    n_polygons = len(polygons)
    padded_polygons = np.full((n_polygons, max_polygon_length), fill_value=-1, dtype=np.int32)

    for i, polygon in enumerate(polygons):
        padded_polygons[i, :len(polygon)] = polygon

    return padded_polygons


@numba.jit
def find_point_in_polygon_compiled(x, y, padded_polygonization, verticies):
    """Finds the index of the face in the given polygonization that contains the given point

    Parameters
    ----------
    x : float, the x-coordinate of the point
    y : float, the y-coordinate of the point
    padded_polygonization : np.array of shape (n_polygons, max_polygon_length)
    verticies : np.array of shape (n_verticies, 2)

    Returns
    -------
    index : int, the index of the face that contains the point
    """
    n_polygons = len(padded_polygonization)

    for i in range(n_polygons):
        if point_inside_convex_padded_polygon_compiled(x, y, padded_polygonization[i], verticies):
            return i

    return -1


@numba.jit
def point_inside_convex_padded_polygon_compiled(x, y, padded_polygon, coordinates):
    """Check if a point lies inside a convex polygon using NumPy and Numba.
    The polygon must be ordered in a counterclockwise direction.

    Args:
    - x: The x coordinate of the point
    - y: The y coordinate of the point
    - padded_polygon: An array containing the indices of the vertices of the polygon
    - coordinates: A 2D NumPy array containing the coordinates of the vertices of the polygon

    Returns:
    - A boolean indicating whether the point lies inside the polygon
    """
    true_polygon_length = 0
    for i in range(len(padded_polygon) - 1, -1, -1):
        if padded_polygon[i] != -1:
            true_polygon_length = i + 1
            break

    # polygon_wrap_padded = np.insert(padded_polygon, 0, padded_polygon[true_polygon_length], axis=0, dtype=np.int32)
    # numba cannot handle np.insert, so:
    polygon_wrap_padded = np.zeros(true_polygon_length + 1, dtype=np.int32)
    polygon_wrap_padded[0] = padded_polygon[true_polygon_length - 1]
    polygon_wrap_padded[1:] = padded_polygon[:true_polygon_length]

    for i in range(true_polygon_length):
        if point_to_right_of_line_compiled(
            coordinates[polygon_wrap_padded[i], 0],
            coordinates[polygon_wrap_padded[i], 1],
            coordinates[polygon_wrap_padded[i + 1], 0],
            coordinates[polygon_wrap_padded[i + 1], 1],
            x,
            y,
        ):
            return False
    return True


@numba.jit
def point_to_right_of_line_compiled(tail_x, tail_y, head_x, head_y, point_x, point_y):
    """Check if a point lies to the right of a line oriented from tail to head using."""
    # compute the cross product of the vectors (tail -> head) and (tail -> point)
    return (head_y - tail_y) * (point_x - tail_x) - (head_x - tail_x) * (point_y - tail_y) > 0


@numba.jit
def build_poly_topo_bdryEdges_intEdges_compiled(padded_polygonization):
    """returns a polgonization topology giving adjacent polygons across each edge of the given padded polygonization

    0 means that edge is a boundary edge
    -1 means that edge does not exist
    """
    n_polygons = len(padded_polygonization)
    max_polygon_length = len(padded_polygonization[0])
    edges_wrapped_ordered = np.zeros((n_polygons, max_polygon_length, 2), dtype=np.int64)  # (0, 0) for non-existent edges
    polygonization_topology = np.full((n_polygons, max_polygon_length), -1, dtype=np.int64)  # -1 for non-existent edges
    boundary_edges = np.zeros((max_polygon_length * n_polygons, max_polygon_length), dtype=np.int64)
    internal_edges = np.zeros((max_polygon_length * n_polygons, max_polygon_length), dtype=np.int64)
    n_boundary_edges = 0
    n_internal_edges = 0
    found_flag = False

    # build edges_wrapped_ordered
    for i in range(n_polygons):
        for j in range(max_polygon_length):
            if j == max_polygon_length - 1:
                # j has reached end of polygon
                if padded_polygonization[i, j] != -1:
                    # CASE : edge wraps back to beginning of polygon
                    edges_wrapped_ordered[i, j, 0] = padded_polygonization[i, j]
                    edges_wrapped_ordered[i, j, 1] = padded_polygonization[i, 0]
                else:
                    # CASE : no more edges that exist
                    break
            else:
                # j has not reached end of polygon
                if padded_polygonization[i, j + 1] == -1:
                    if padded_polygonization[i, j] != 0:
                        # CASE : edge wraps back to first vertex of polygon
                        edges_wrapped_ordered[i, j, 0] = padded_polygonization[i, j]
                        edges_wrapped_ordered[i, j, 1] = padded_polygonization[i, 0]
                    else:
                        # CASE : no more edges that exist
                        break
                else:
                    # CASE : edge exists
                    edges_wrapped_ordered[i, j, 0] = padded_polygonization[i, j]
                    edges_wrapped_ordered[i, j, 1] = padded_polygonization[i, j + 1]

    # build polygonization_topology
    for i in range(n_polygons):
        for k in range(max_polygon_length):
            if polygonization_topology[i, k] != -1:
                # Continue if adjacent poly already computed
                continue
            if edges_wrapped_ordered[i, k, 0] == -1:
                # bread if end of polygon reached
                break

            found_flag = False
            current_edge = edges_wrapped_ordered[i, k, ::-1]  # reverse edge direction
            for j in range(n_polygons):
                if found_flag:
                    break
                for m in range(max_polygon_length):
                    if found_flag:
                        break
                    if current_edge[0] == edges_wrapped_ordered[j, m, 0] and current_edge[1] == edges_wrapped_ordered[j, m, 1]:
                        # CASE : edge found
                        polygonization_topology[i, k] = j
                        polygonization_topology[j, m] = i
                        n_internal_edges += 1

                        if i <= j:
                            internal_edges[n_internal_edges - 1, 0] = i
                            internal_edges[n_internal_edges - 1, 1] = j
                        else:
                            internal_edges[n_internal_edges - 1, 0] = j
                            internal_edges[n_internal_edges - 1, 1] = i

                        found_flag = True

            if not found_flag:
                polygonization_topology[i, k] = 0
                n_boundary_edges += 1
                boundary_edges[n_boundary_edges - 1, 0] = current_edge[1]  # remember edge was reversed
                boundary_edges[n_boundary_edges - 1, 1] = current_edge[0]  # remember edge was reversed

    # take only first two columns of boundary_edges and internal_edges and rows up to n_boundary_edges and n_internal_edges
    boundary_edges = boundary_edges[:n_boundary_edges, :2]
    internal_edges = internal_edges[:n_internal_edges, :2]

    return n_boundary_edges, n_internal_edges, polygonization_topology, boundary_edges, internal_edges


@numba.jit
def boundary_connected_classes_unseeded_compiled(boundary_edges):
    """
    Returns a list of boundary connected classes of boundary edges
    """
    n_boundary_edges = len(boundary_edges)
    class_list = np.zeros((1000, n_boundary_edges), dtype=np.int64)
    edges_used = np.zeros(n_boundary_edges, dtype=np.int64)

    edges_used[0] = 1
    start = boundary_edges[0, 0]
    current_vertex = boundary_edges[0, 1]

    class_list[0, 0] = start
    class_list[0, 1] = current_vertex

    n_classes = 1
    position = 2  # this is the position of the next vertex to be added to the class

    # compute class 1
    found_flag = False
    all_found_flag = False
    while not found_flag:
        for i in range(n_boundary_edges):
            if edges_used[i] == 1:
                continue
            if boundary_edges[i, 0] == current_vertex:
                if boundary_edges[i, 1] == start:
                    found_flag = True
                class_list[0, position] = boundary_edges[i, 1]
                current_vertex = boundary_edges[i, 1]
                edges_used[i] = 1
                position += 1
    class_list[0, 0] = position - 1  # store length of class

    # determine if all edges have been found
    while not all_found_flag:
        all_found_flag = True
        for i in range(n_boundary_edges):
            if edges_used[i] == 0:
                # CASE : edge has not been used, so start class 2 with it
                all_found_flag = False
                n_classes += 1
                start = boundary_edges[i, 0]
                current_vertex = boundary_edges[i, 1]
                class_list[n_classes - 1, 0] = start
                class_list[n_classes - 1, 1] = current_vertex
                position = 2
                edges_used[i] = 1
                break
        if all_found_flag:
            break
        # compute class 2
        found_flag = False
        while not found_flag:
            for i in range(n_boundary_edges):
                if edges_used[i] == 1:
                    continue
                if boundary_edges[i, 0] == current_vertex:
                    if boundary_edges[i, 1] == start:
                        found_flag = True
                    class_list[n_classes - 1, position] = boundary_edges[i, 1]
                    current_vertex = boundary_edges[i, 1]
                    edges_used[i] = 1
                    position += 1
        class_list[n_classes - 1, 0] = position - 1  # store length of class

    # unpack class_list and remove 0s using length of class
    class_list = class_list[:n_classes, :]
    components = []
    for row in class_list:
        components.append(row[1:row[0] + 1])

    return components


@numba.jit
def winding_number_compiled(x, y, path, coordinates):
    """Find the winding number of the path (indices in the given list of coordinates) with respect to the point (x, y)"""
    n_vertices = len(path)
    winding_number = 0
    angle = 0
    sign = 0

    for i in range(n_vertices):
        head_1 = coordinates[path[i]]
        head_2 = coordinates[path[(i + 1) % n_vertices]]  # TODO: check if this is correct
        vector_1 = head_1 - np.array([x, y])
        vector_2 = head_2 - np.array([x, y])
        angle = angle_compiled(vector_1, vector_2)

        if point_to_right_of_line_compiled(x, y, head_1[0], head_1[1], head_2[0], head_2[1]):
            sign = -1
        else:
            sign = 1

        winding_number += sign * angle

    return winding_number / (2 * np.pi)


@numba.jit
def angle_compiled(vector_1, vector_2):
    """Find the angle between two vectors"""
    dot = np.dot(vector_1, vector_2)
    norm1 = np.linalg.norm(vector_1)
    norm2 = np.linalg.norm(vector_2)
    hold = dot / (norm1 * norm2)
    if hold >= 1:
        hold = 1
    elif hold <= -1:
        hold = -1
    angle = np.arccos(hold)
    return angle


@numba.jit
def polygon_path_to_vertex_path_compiled(faces_path, padded_polygonization, inner_ring_verticies, outer_ring_verticies):
    """Given a path of faces, converts them to a list of verticies from the outer to inner ring"""
    n_path_polygons = len(faces_path)
    max_polygon_length = len(padded_polygonization[0])

    current_polygon = np.zeros(max_polygon_length, dtype=np.int64)
    next_polygon = np.zeros(max_polygon_length, dtype=np.int64)
    current_vertex = 0
    next_vertex = 1
    n_vertex_path = 2
    vertex_path = np.zeros(max_polygon_length * n_path_polygons, dtype=np.int64)

    # find the first polygon in the path
    current_polygon_index = 0
    current_polygon = padded_polygonization[faces_path[current_polygon_index]]
    # find the edge where the vertex changes from the inner ring to not the inner ring
    while not (current_polygon[current_vertex] in inner_ring_verticies and current_polygon[next_vertex] not in inner_ring_verticies):
        current_vertex = next_vertex

        if current_vertex == max_polygon_length - 1 or current_polygon[current_vertex + 1] == -1:
            next_vertex = 0
        else:
            next_vertex = current_vertex + 1

    # add the first vertex to the path
    vertex_path[0] = current_polygon[current_vertex]
    vertex_path[1] = current_polygon[next_vertex]
    current_vertex = next_vertex

    # for intermediate polygons in the path
    for current_polygon_index in range(n_path_polygons - 1):
        current_polygon = padded_polygonization[faces_path[current_polygon_index]]
        next_polygon = padded_polygonization[faces_path[current_polygon_index + 1]]

        # traverse new polygon until the position lines up with the vertex path
        while current_polygon[current_vertex] != vertex_path[n_vertex_path - 1]:
            current_vertex = next_vertex
            if current_vertex == max_polygon_length - 1 or current_polygon[current_vertex + 1] == -1:
                next_vertex = 0
            else:
                next_vertex = current_vertex + 1

        # Now that alignment is done, traverse polygon while adding verticies to vertexPath until nextPolygon is reached
        while np.count_nonzero(next_polygon == current_polygon[current_vertex]) == 0:
            current_vertex = next_vertex
            if current_vertex == max_polygon_length - 1 or current_polygon[current_vertex + 1] == -1:
                next_vertex = 0
            else:
                next_vertex = current_vertex + 1
            n_vertex_path += 1
            vertex_path[n_vertex_path - 1] = current_polygon[current_vertex]

    # for the last polygon in the path
    current_polygon_index = n_path_polygons - 1
    current_polygon = padded_polygonization[faces_path[current_polygon_index]]

    # traverse new polygon until the position lines up with the vertex path
    while current_polygon[current_vertex] != vertex_path[n_vertex_path - 1]:
        current_vertex = next_vertex
        if current_vertex == max_polygon_length - 1 or current_polygon[current_vertex + 1] == -1:
            next_vertex = 0
        else:
            next_vertex = current_vertex + 1

    # Now that alignment is done, traverse polygon while adding verticies to vertexPath until outerRingverticies is reached
    while np.count_nonzero(outer_ring_verticies == current_polygon[current_vertex]) == 0:
        current_vertex = next_vertex
        if current_vertex == max_polygon_length - 1 or current_polygon[current_vertex + 1] == -1:
            next_vertex = 0
        else:
            next_vertex = current_vertex + 1
        n_vertex_path += 1
        vertex_path[n_vertex_path - 1] = current_polygon[current_vertex]

    # remove trailing zeros
    vertex_path = vertex_path[:n_vertex_path]

    return vertex_path


@numba.jit
def to_right_of_edge_lookup_polygons_compiled(padded_polygonization):
    """
    Creates a lookup table for which polygons are to the right of a given edge

    Contains rows with the following information:
    [edge[0], edge[1], polygon_index]

    """
    n_polygons = len(padded_polygonization)
    max_polygon_length = len(padded_polygonization[0])
    table = np.zeros((max_polygon_length * n_polygons + 1, 3), dtype=np.int64)

    counter = 0
    actual_polygon_length = 0
    for i in range(n_polygons):
        for j in range(max_polygon_length - 1, -1, -1):
            if padded_polygonization[i][j] != -1:
                actual_polygon_length = j + 1  # +1 because j is zero indexed
                break
        for j in range(actual_polygon_length - 1):
            counter += 1
            table[counter][0] = padded_polygonization[i][j+1]
            table[counter][1] = padded_polygonization[i][j]
            table[counter][2] = i
        counter += 1
        table[counter][0] = padded_polygonization[i][0]
        table[counter][1] = padded_polygonization[i][actual_polygon_length - 1]
        table[counter][2] = i

    table[0][0] = counter - 1
    return table[1:]


class Triangulation(object):
    """Triangulation/Voronoi dual object"""
    def __init__(self, region, vertices, vertex_boundary_markers, triangles, topology):
        self.region = region
        self.vertices = vertices
        self.triangles = triangles
        self.vertex_boundary_markers = vertex_boundary_markers
        self.topology = topology

        self._triangulation_edges, self._edge_boundary_markers = self.make_triangulation_edges()
        self.triangulation_edges_unique, self.edge_boundary_markers_unique = self.make_unique_triangulation_edges()

        self.num_vertices = len(vertices)
        self.num_edges = len(self.triangulation_edges_unique)
        self.num_triangles = len(triangles)

        self.triangle_coordinates = self.make_triangle_coordinates()
        self.barycenters = self.make_barycenters()
        self.circumcenters = self.make_circumcenters()  # coordinates of voin diagram. lambda[0]

        self.vertex_topology = self.build_vertex_topology()

        self.vertex_index_to_triangle = self.make_vertex_index_to_triangle()
        if topology is not None:
            self.voronoi_tesselation = self.make_voronoi_tesselation()  # lambda[2]
            self.contained_polygons = self.make_contained_polygons()  # lambda[2]
            self.voronoi_edges = self.make_voronoi_edges()  # lambda[1]
            self.to_right_of_edge_poly_dict = self.build_edge_to_right_polygons_dict()

    def build_slitted_weighted_voronoi_graph(
            self,
            # lambda[0] = verticies of v which is the triangles of the triangulization (x,y) needs to be comp, lambda[1] = list of polygon edges (not important), lambda[2] = list of polygons
            # voronoi_verticies, # lambda[0] (circumcenters)
            # contained_verticies, # indecies of the contained verticies. (flattened list of contained_polygons)
            # contained_faces, # indecies of the contained polygons.
            omega0,  # point we are trying to find shortest path from
            point_in_hole,
            # to_right_of_edge_poly_dict  # dictionary. we have v cells, can take point and manuver around it. for each of those edges, if we
            ):
        """Build the slitted weighted voronoi graph"""

        proxy_infinity = 1e10
        # TODO: make sure these are correct
        contained_verticies = set()
        for polygon in self.contained_polygons:
            for vertex in polygon:
                contained_verticies.add(vertex)
        contained_verticies = np.array(list(contained_verticies))
        contained_faces = np.arange(len(self.contained_polygons))

        # find which face contains omega0
        padded_polygonization_all = pad_polygons_to_matrix(self.contained_polygons)
        omega0_face = find_point_in_polygon_compiled(omega0[0], omega0[1], padded_polygonization_all, self.circumcenters)
        if omega0_face not in contained_faces:
            raise ValueError('omega0 not in contained_faces')

        """
        # keep only the polygons that are contained in the region
        padded_polygonization = pad_polygons_to_matrix(np.take(self.contained_polygons, contained_faces, axis=0))
        n_polygons = len(padded_polygonization)
        max_polygon_length = len(padded_polygonization[0])  # all polygons have the same length
        """
        padded_polygonization = padded_polygonization_all

        # construct and parse polygonizationTopology
        n_boundary_edges, n_internal_edges, polygonization_topology, boundary_edges, internal_edges = \
            build_poly_topo_bdryEdges_intEdges_compiled(padded_polygonization)

        # construct connected components from boundary edges
        components = boundary_connected_classes_unseeded_compiled(boundary_edges)
        boundary_comp_1 = components[0]
        boundary_comp_2 = components[1]

        # find winding number of point in hole (indicies in the given list of coordinates)
        # makes use of the counter clockwise nature of verticies
        winding_1 = winding_number_compiled(point_in_hole[0], point_in_hole[1], boundary_comp_1, self.circumcenters)
        outer_ring_vertices, inner_ring_vertices = boundary_comp_1, boundary_comp_2
        if winding_1 <= 0:
            outer_ring_vertices = boundary_comp_2
            inner_ring_vertices = boundary_comp_1

        # create inner and outer ring faces
        # finding all faces that contain the inner and outer ring verticies
        inner_ring_faces = [i for i in range(len(padded_polygonization)) if any(np.intersect1d(inner_ring_vertices, padded_polygonization[i]))]
        outer_ring_faces = [i for i in range(len(padded_polygonization)) if any(np.intersect1d(outer_ring_vertices, padded_polygonization[i]))]

        # create contained faces graph (networkx) and shortest path function
        contained_faces_graph = nx.Graph()
        contained_faces_graph.add_nodes_from(range(len(contained_faces)))
        for edge in internal_edges:
            contained_faces_graph.add_edge(*edge)

        shortest_paths = dict(nx.shortest_path(contained_faces_graph, source=omega0_face))
        paths_to_inner_ring_faces = [shortest_paths[face] for face in inner_ring_faces]
        paths_to_outer_ring_faces = [shortest_paths[face] for face in outer_ring_faces]

        # construct the slit
        slit_face_start = np.argmin([len(path) for path in paths_to_inner_ring_faces])
        slit_face_end = np.argmin([len(path) for path in paths_to_outer_ring_faces])
        slit_face_path = np.array(paths_to_inner_ring_faces[slit_face_start][::-1] + paths_to_outer_ring_faces[slit_face_end][1:])

        slit_vertices = polygon_path_to_vertex_path_compiled(slit_face_path, padded_polygonization, inner_ring_vertices, outer_ring_vertices)

        # omega0n candidates (the verticies that are not part of the slit)
        omega0_candidates = np.setdiff1d(self.contained_polygons[omega0_face], slit_vertices)  # TODO: check if this is correct
        omega0n = omega0_candidates[0]

        # add weights to edges (makeing sure to use inf for edges of slit)
        # TODO: make sure that indexing of edges and graph are correct
        edges_with_weight_with_inf = []
        for i in range(len(self.voronoi_edges)):
            # only append edges that should have weight of inf
            if (self.voronoi_edges[i][0] not in contained_verticies or
                self.voronoi_edges[i][1] not in contained_verticies or
                self.voronoi_edges[i][0] in slit_vertices or
                self.voronoi_edges[i][1] in slit_vertices or
                (self.to_right_of_edge_poly_dict[tuple(self.voronoi_edges[i])] not in contained_faces and
                 self.to_right_of_edge_poly_dict[tuple(self.voronoi_edges[i][::-1])] not in contained_faces)):

                edges_with_weight_with_inf.append(i)  # add 1 to account for 1 indexing

        print('edges_with_weight_with_inf', edges_with_weight_with_inf)
        edges_plus_weights = [(u, v, proxy_infinity) if i in edges_with_weight_with_inf else (u, v, 1) for i, (u, v) in enumerate(self.voronoi_edges)]

        # construct the final graph
        Lambda_Graph = nx.Graph()
        Lambda_Graph.add_nodes_from(range(len(self.circumcenters)))
        Lambda_Graph.add_weighted_edges_from(edges_plus_weights)
        nx.set_node_attributes(Lambda_Graph, {i: str(i) for i in range(len(self.circumcenters))}, 'name')

        return Lambda_Graph, omega0n, slit_vertices

    def build_edge_to_right_polygons_dict(self):
        """Build a dictionary of the polygons to the right of each edge"""
        # TODO: why not use voronoi edges in loop/why is the lookup so much larger. Shared edges?
        padded_polygonization = pad_polygons_to_matrix(self.contained_polygons)  # TODO: contained polygons instead of voronoi tessalation?
        to_right_of_edge_lookup = to_right_of_edge_lookup_polygons_compiled(padded_polygonization)

        to_right_of_edge_poly_dict = defaultdict(lambda: -1)
        for i in range(len(to_right_of_edge_lookup)):
            to_right_of_edge_poly_dict[tuple(to_right_of_edge_lookup[i, 0:2])] = to_right_of_edge_lookup[i, 2]

        return to_right_of_edge_poly_dict

    def make_barycenters(self):
        """Build the array of barycenters from a triangulation"""
        barycenters = np.array(list(map(
            lambda x: np.mean(x, axis=0),
            self.triangle_coordinates
        )))
        return barycenters

    def make_circumcenters(self):
        """Build the array of circumcenters from a triangulation"""
        circumcenters = np.array(list(map(
            lambda x: triangle_circumcenter(
                x[0][0], x[0][1],
                x[1][0], x[1][1],
                x[2][0], x[2][1]
            ),
            self.triangle_coordinates
        )))
        return circumcenters

    def make_triangulation_edges(self):
        """Make the edges of the triangulation"""
        edges_list = []
        edge_boundary_marker_list = []
        num_triangles = self.triangles.shape[0]
        for triangle in range(num_triangles):
            edges_list.append(np.sort(np.array(
                [self.triangles[triangle, 0], self.triangles[triangle, 1]]
            )))
            edges_list.append(np.sort(np.array(
                [self.triangles[triangle, 1], self.triangles[triangle, 2]]
            )))
            edges_list.append(np.sort(np.array(
                [self.triangles[triangle, 2], self.triangles[triangle, 0]]
            )))

            edge_boundary_marker_list.append(
                0 if self.vertex_boundary_markers[self.triangles[triangle, 0]] != self.vertex_boundary_markers[self.triangles[triangle, 1]] else self.vertex_boundary_markers[self.triangles[triangle, 0]]
            )
            edge_boundary_marker_list.append(
                0 if self.vertex_boundary_markers[self.triangles[triangle, 1]] != self.vertex_boundary_markers[self.triangles[triangle, 2]] else self.vertex_boundary_markers[self.triangles[triangle, 1]]
            )
            edge_boundary_marker_list.append(
                0 if self.vertex_boundary_markers[self.triangles[triangle, 2]] != self.vertex_boundary_markers[self.triangles[triangle, 0]] else self.vertex_boundary_markers[self.triangles[triangle, 2]]
            )
        return np.vstack(edges_list), np.array(edge_boundary_marker_list)

    def make_unique_triangulation_edges(self):
        """Get the unique edges by first sorting"""
        edges_sorted = np.copy(self._triangulation_edges)
        edge_boundary_markers_sorted = np.copy(self._edge_boundary_markers)

        num_edges = len(edges_sorted)
        sort_perm = np.lexsort(np.rot90(edges_sorted))
        edges_sorted = edges_sorted[sort_perm]
        edge_boundary_markers_sorted = edge_boundary_markers_sorted[sort_perm]

        keep_indicator = np.full((num_edges,), True)
        for i in range(num_edges - 1):
            if np.all(edges_sorted[i, :] == edges_sorted[i + 1, :]):
                keep_indicator[i] = False
        edges_unique = edges_sorted[keep_indicator, :]

        edge_boundary_markers_unique = edge_boundary_markers_sorted[keep_indicator]
        return edges_unique, edge_boundary_markers_unique

    def make_triangle_coordinates(self):
        """Make a array containing each of the triangles in terms of their coordinates"""
        triangle_coordinate_list = []
        num_triangles = self.triangles.shape[0]
        for triangle in range(num_triangles):
            triangle_coordinate_list.append(np.vstack([
                self.vertices[self.triangles[triangle, 0], :],
                self.vertices[self.triangles[triangle, 1], :],
                self.vertices[self.triangles[triangle, 2], :]
            ]))
        return np.stack(triangle_coordinate_list)

    def make_vertex_index_to_triangle(self):
        """Create a mapping from each interior vertex to a triangle_index which contains it"""
        vertex_index_to_triangle = np.zeros(len(self.vertices), dtype=np.int64)
        for triangle_index, triangle in enumerate(self.triangles):
            for vertex in triangle:
                if self.vertex_boundary_markers[vertex] == 0:  # If vertex is an interior vertex
                    vertex_index_to_triangle[vertex] = triangle_index
        return vertex_index_to_triangle

    def build_vertex_topology(self):
        """Build a data structure where the ith row is all vertices connected by an edge to
        vertex i"""
        # TODO: TEST THIS
        num_edges = len(self.triangulation_edges_unique)
        num_neighbors_table = np.zeros(self.num_vertices, dtype=np.int64)
        valence_upper_bound = 50
        vertex_topology = np.full((self.num_vertices, valence_upper_bound), -1, dtype=np.int64)
        for i in range(num_edges):
            v1 = self.triangulation_edges_unique[i, 0]
            v2 = self.triangulation_edges_unique[i, 1]
            num_neighbors_table[v1] += 1
            num_neighbors_table[v2] += 1
            vertex_topology[v1, num_neighbors_table[v1] - 1] = v2
            vertex_topology[v2, num_neighbors_table[v2] - 1] = v1
        max_valence = np.max(num_neighbors_table)
        return vertex_topology[:, :max_valence]

    def neighbors_ordered_compiled(self, vertex):
        """Take an interior vertex and return all vertices in the triangulation that are neighbors of
        the given vertex in either counterclockwise or clockwise order. Will fail SILENTLY on boundary
        vertices."""
        max_valence = len(self.vertex_topology[0])
        neighbors_extended = self.vertex_topology[vertex]
        num_neighbors = max_valence
        for j in range(max_valence):
            if neighbors_extended[j] == -1:
                num_neighbors = j
                break
        neighbors = neighbors_extended[:num_neighbors]
        result = np.zeros(num_neighbors, dtype=np.int64)
        result[0] = neighbors[0]
        # There will always be two intersections between two rings of neighbors.
        # In the first iteration, pick one arbitrarily,
        # since we don't care if counterclockwise or clockwise.
        for j in range(max_valence):
            if self.vertex_topology[result[0], j] in neighbors:
                result[1] = self.vertex_topology[result[0], j]
                break
        # Now continue chaining together elements of neighbors connected to the previously added
        # element of result
        for i in range(1, num_neighbors - 1):
            for j in range(max_valence):
                if (self.vertex_topology[result[i], j] in neighbors) and (self.vertex_topology[result[i], j] != result[i - 1]):
                    result[i + 1] = self.vertex_topology[result[i], j]
                    break
            if result[i] == result[0]:
                return result[:-1]
        return result

    def make_voronoi_tesselation(self):
        """Build the Voronoi tesselation from a triangulation"""
        # Loop over the vertices and for interior vertices (bdry_mrk == 0)
        # build the corresponding Voronoi cell
        voronoi_tesselation = [[] for _ in range(len(self.vertices))]
        for vertex in range(len(self.vertices)):
            if self.vertex_boundary_markers[vertex] == 0:
                starting_triangle = self.vertex_index_to_triangle[vertex]
                active_triangle = starting_triangle
                voronoi_tesselation[vertex].append(starting_triangle)
                while True:
                    active_vertex_location = np.where(
                        self.triangles[active_triangle] == vertex
                    )[0][0]
                    # active_vertex is the head of the edge across which we add the next triangle
                    next_triangle_location = active_vertex_location - 1 % 3
                    appending_triangle = self.topology[active_triangle][next_triangle_location]
                    active_triangle = appending_triangle
                    if appending_triangle == starting_triangle:
                        break
                    voronoi_tesselation[vertex].append(appending_triangle)
        return voronoi_tesselation

    def make_contained_polygons(self):
        """Make the list of polygons from the Voronoi tesselation that are completely contained
        in the region"""
        contained_polygons = [v for v in self.voronoi_tesselation if len(v) != 0]
        return contained_polygons

    @staticmethod
    def make_polygon_edges(polygon):
        num_vertices = len(polygon)
        voronoi_edges = (
            [[polygon[i], polygon[i + 1]] for i in range(num_vertices - 1)]
            + [[polygon[-1], polygon[0]]]
        )
        return voronoi_edges

    def make_voronoi_edges(self):
        """Make the edges of the voronoi"""
        voronoi_edges_non_flat = list(map(self.make_polygon_edges, self.contained_polygons))
        voronoi_edges = np.array([val for sublist in voronoi_edges_non_flat for val in sublist])
        return voronoi_edges

    @staticmethod
    def read(file_name):
        """Read a triangulation object from files with the given path"""
        path = Path(file_name)
        region = Region.read_poly(path.with_suffix('.poly'))
        vertices, vertex_boundary_markers = read_node(path.with_suffix('.node'))
        triangles = read_ele(path.with_suffix('.ele'))
        if path.with_suffix('.topo.ele').is_file():
            topology = read_ele(path.with_suffix('.topo.ele'))
        else:
            topology = None
        return Triangulation(region, vertices, vertex_boundary_markers, triangles, topology)

    def write(self, file_name):
        """Write a triangulation object as a .poly file with node and edge information containing boundary markers"""
        dimension = 2
        num_attributes = 0
        num_boundary_markers = 1

        with open(file_name, 'w', encoding='utf-8') as stream:
            stream.write(
                f'{self.num_vertices} {dimension} {num_attributes} {num_boundary_markers}\n'
            )
            for i in range(len(self.vertices)):
                stream.write(
                    f'{i + 1} {self.vertices[i, 0]} {self.vertices[i, 1]} {self.vertex_boundary_markers[i]}\n'
                )

            stream.write(
                f'{self.num_edges} {num_boundary_markers}\n'
            )
            for i in range(len(self.triangulation_edges_unique)):
                stream.write(
                    f'{i + 1} {self.triangulation_edges_unique[i, 0] + 1} '
                    + f'{self.triangulation_edges_unique[i, 1] + 1} {self.edge_boundary_markers_unique[i]}\n'
                )

            stream.write(
                f'{len(self.region.points_in_holes)}\n'
            )
            for i in range(len(self.region.points_in_holes)):
                stream.write(
                    f'{i + 1} {self.region.points_in_holes[i, 0]} {self.region.points_in_holes[i, 1]}\n'
                )

    def show(self, file_name, show_vertex_indices=False, show_triangle_indices=False, face_color=[153/255, 204/255, 255/255]):
        """Show an image of the triangulation"""
        fig, axes = plt.subplots()
        axes.scatter(self.vertices[:, 0], self.vertices[:, 1])
        lines = [
            [
                tuple(self.vertices[edge[0]]),
                tuple(self.vertices[edge[1]])
            ] for edge in self._triangulation_edges
        ]
        line_collection = mc.LineCollection(lines, linewidths=2)
        # color_array = np.ones(self.num_triangles) * color  # np.random.random(self.num_triangles) * 500
        poly_collection = mc.PolyCollection(
            self.triangle_coordinates,
            # array=color_array, cmap=matplotlib.cm.Blues
        )
        poly_collection.set(facecolor=face_color)
        axes.add_collection(line_collection)
        axes.add_collection(poly_collection)
        if show_triangle_indices:
            barycenters = np.array(list(map(
                lambda x: np.mean(x, axis=0),
                self.triangle_coordinates
            )))
            for i in range(self.num_triangles):
                plt.text(barycenters[i, 0], barycenters[i, 1], str(i))
        if show_vertex_indices:
            for i in range(self.num_vertices):
                plt.text(self.vertices[i, 0], self.vertices[i, 1], str(i))
        axes.autoscale()
        axes.margins(0.1)
        fig.savefig(file_name)

    def show_voronoi_tesselation(self, file_name, show_vertex_indices=False, show_polygon_indices=False):
        """Show the voronoi tesselation"""
        fig, axes = plt.subplots()

        # Plot the edges of the polygons
        polygon_lines = [
            [
                tuple(self.circumcenters[edge[0]]),
                tuple(self.circumcenters[edge[1]])
            ] for edge in self.voronoi_edges
        ]
        polygon_line_collection = mc.LineCollection(polygon_lines, linewidths=2)

        # Plot the region outline
        region_lines = [
            [
                tuple(self.region.coordinates[edge[0]]),
                tuple(self.region.coordinates[edge[1]])
            ] for edge in self.region.edges
        ]
        region_line_collection = mc.LineCollection(region_lines, linewidths=2)

        # Plot the polygons
        polygon_coordinates = [
            np.array(list(map(lambda x: self.circumcenters[x], polygon)))
            for polygon in self.contained_polygons
        ]
        polygon_collection = mc.PolyCollection(
            polygon_coordinates
        )
        polygon_collection.set(facecolor=[180/255, 213/255, 246/255])
        axes.add_collection(polygon_line_collection)
        axes.add_collection(region_line_collection)
        axes.add_collection(polygon_collection)
        if show_polygon_indices:
            barycenters = np.array(list(map(
                lambda x: np.mean(x, axis=0),
                polygon_coordinates
            )))
            for i in range(len(barycenters)):
                plt.text(barycenters[i, 0], barycenters[i, 1], str(i), fontsize=6, weight='bold')
        if show_vertex_indices:
            for i in range(len(self.circumcenters)):
                plt.text(self.circumcenters[i, 0], self.circumcenters[i, 1], str(i), fontsize=6, weight='bold')

        axes.autoscale()
        axes.margins(0.1)
        axes.scatter(self.circumcenters[:, 0], self.circumcenters[:, 1])
        fig.show()
        fig.savefig(file_name)


if __name__ == '__main__':
    path = Path("regions/test/test")
    T = Triangulation.read(path)

    slitted_weighted_voronoi_graph, omega0, slit_verticies = T.build_slitted_weighted_voronoi_graph(omega0=(650, 420), point_in_hole=(900, 400))

    print("slit verticies: ", slit_verticies)
    print("omega0: ", omega0)
    print("slitted_weighted_voronoi_graph: ", slitted_weighted_voronoi_graph)

    # print all edges of networkx graph
    print(list(slitted_weighted_voronoi_graph.edges(data=True)))

    # print all nodes of networkx graph
    print(list(slitted_weighted_voronoi_graph.nodes(data=True)))

    T.show("regions/test/test.png", show_vertex_indices=True, show_triangle_indices=False)
    T.show_voronoi_tesselation("regions/test/test_voronoi_vert.png", show_vertex_indices=True, show_polygon_indices=False)
    T.show_voronoi_tesselation("regions/test/test_voronoi_poly.png", show_vertex_indices=False, show_polygon_indices=True)
