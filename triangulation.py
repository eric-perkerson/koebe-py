import math
import numpy as np
import numba
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from cmcrameri import cm
from pathlib import Path
from region import read_node, read_ele, read_pde, Region


@numba.njit
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


# # Test
# angle_compiled(
#     np.array([1.0, 0.0]),
#     np.array([0.0, 1.0])
# )
# angle_compiled(
#     np.array([1.0, 0.0]),
#     np.array([1.0, 1.0])
# )


@numba.njit
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



@numba.njit
def line_height_intersect(point_1, point_2, value_1, value_2, height):
    t = (height - value_1) / (value_2 - value_1)
    return t * (point_2 - point_1) + point_1


def tri_level_sets(triangle_coordinates, triangle_f_values, heights):
    sort_perm = np.argsort(triangle_f_values)
    triangle_f_values_sorted = triangle_f_values[sort_perm]
    low_vertex = sort_perm[0]
    mid_vertex = sort_perm[1]
    high_vertex = sort_perm[2]
    def build_line(height):
        if triangle_f_values_sorted[0] <= height and height <= triangle_f_values_sorted[1]:
            return np.array([
                line_height_intersect(
                    triangle_coordinates[low_vertex],
                    triangle_coordinates[mid_vertex],
                    triangle_f_values_sorted[0],
                    triangle_f_values_sorted[1],
                    height
                ),
                line_height_intersect(
                    triangle_coordinates[low_vertex],
                    triangle_coordinates[high_vertex],
                    triangle_f_values_sorted[0],
                    triangle_f_values_sorted[2],
                    height
                )
            ])
        if triangle_f_values_sorted[1] <= height and height <= triangle_f_values_sorted[2]:
            return np.array([
                line_height_intersect(
                    triangle_coordinates[low_vertex],
                    triangle_coordinates[high_vertex],
                    triangle_f_values_sorted[0],
                    triangle_f_values_sorted[2],
                    height
                ),
                line_height_intersect(
                    triangle_coordinates[mid_vertex],
                    triangle_coordinates[high_vertex],
                    triangle_f_values_sorted[1],
                    triangle_f_values_sorted[2],
                    height
                )
            ])
        else:
            return np.array([])
    results = list(map(build_line, heights))
    return results


def flatten_list_of_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def build_polygon_edges(polygon_vertices):
    edges = []
    for i in range(len(polygon_vertices) - 1):
        edge = [
            polygon_vertices[i],
            polygon_vertices[i + 1]
        ]
        edges.append(edge)
    edges.append([polygon_vertices[-1], polygon_vertices[0]])
    return edges


@numba.njit
def point_to_right_of_line_compiled(tail_x, tail_y, head_x, head_y, point_x, point_y):
    """Check if a point lies to the right of a line oriented from tail to head using."""
    # compute the cross product of the vectors (tail -> head) and (tail -> point)
    return (head_y - tail_y) * (point_x - tail_x) - (head_x - tail_x) * (point_y - tail_y) > 0


@numba.njit
def segment_intersects_segment(a_x, a_y, b_x, b_y, c_x, c_y, d_x, d_y):
    return (
        (
            point_to_right_of_line_compiled(
                a_x, a_y, b_x, b_y, c_x, c_y
            )
            != point_to_right_of_line_compiled(
                a_x, a_y, b_x, b_y, d_x, d_y
            )
        )
        and (
            point_to_right_of_line_compiled(
                c_x, c_y, d_x, d_y, a_x, a_y
            )
            != point_to_right_of_line_compiled(
                c_x, c_y, d_x, d_y, b_x, b_y
            )
        )
    )


def polygon_oriented_counterclockwise(polygon, coordinates):
    """For convex polygons, returns True if the polygon is oriented counterclockwise and returns False otherwise"""
    n = len(polygon)
    for i in range(n):
        if point_to_right_of_line_compiled(
            coordinates[polygon[i], 0],
            coordinates[polygon[i], 1],
            coordinates[polygon[(i + 1) % n], 0],
            coordinates[polygon[(i + 1) % n], 1],
            coordinates[polygon[(i + 2) % n], 0],
            coordinates[polygon[(i + 2) % n], 1]
        ):
            return False
    return True


@numba.njit
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


class Triangulation(object):
    """Triangulation/Voronoi dual object"""
    def __init__(self, region, vertices, vertex_boundary_markers, triangles, topology, pde_values):
        self.region = region
        self.vertices = vertices
        self.triangles = triangles
        self.vertex_boundary_markers = vertex_boundary_markers
        self.topology = topology
        self.pde_values = pde_values

        self._triangulation_edges_all, self._edge_boundary_markers = self.make_triangulation_edges()
        self.triangulation_edges, self.edge_boundary_markers_unique = self.make_unique_triangulation_edges()

        self.num_vertices = len(vertices)
        self.num_edges = len(self.triangulation_edges)
        self.num_triangles = len(triangles)

        self.triangle_coordinates = self.make_triangle_coordinates()
        self.barycenters = self.make_barycenters()
        self.circumcenters = self.make_circumcenters()  # coordinates of voin diagram. lambda[0]
        self.edge_to_right_of_triangle_dict = self.make_triangle_to_right_of_edge_dict()

        self.conductance = self.build_conductance_dict()

        self.vertex_topology = self.build_vertex_topology()

        self.vertex_index_to_triangle = self.make_vertex_index_to_triangle()
        if topology is not None:
            self.voronoi_tesselation = self.make_voronoi_tesselation()  # lambda[2]
            self.contained_polygons, self.contained_to_original_index = self.make_contained_polygons()  # lambda[2]
            self._orient_vertex_topology()
            self._rebase_vertex_topology()

            # Now make the inverse mapping array
            self.original_to_contained_index = np.full((self.num_vertices,), -1)
            for i in range(len(self.contained_to_original_index)):
                self.original_to_contained_index[self.contained_to_original_index[i]] = i

            self.voronoi_edges = self.make_voronoi_edges()  # lambda[1]
            # self.to_right_of_edge_poly_dict = self.build_edge_to_right_polygons_dict()

        if pde_values is not None:
            self.singular_vertices = self.find_singular_vertices()
            self.singular_heights = self.pde_values[self.singular_vertices]

    def make_triangle_to_right_of_edge_dict(self):
        edge_to_right_of_triangle_dict = {}
        for i, triangle in enumerate(self.triangles):
            edge_to_right_of_triangle_dict[tuple([triangle[1], triangle[0]])] = i
            edge_to_right_of_triangle_dict[tuple([triangle[2], triangle[1]])] = i
            edge_to_right_of_triangle_dict[tuple([triangle[0], triangle[2]])] = i
        return edge_to_right_of_triangle_dict

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
        for triangle_index in range(num_triangles):
            triangle = self.triangles[triangle_index]

            edges_list.append(np.sort(np.array(
                [triangle[0], triangle[1]]
            )))
            edges_list.append(np.sort(np.array(
                [triangle[1], triangle[2]]
            )))
            edges_list.append(np.sort(np.array(
                [triangle[2], triangle[0]]
            )))

            if self.vertex_boundary_markers[triangle[0]] != self.vertex_boundary_markers[triangle[1]]:
                edge_boundary_marker_list.append(0)
            else:
                edge_boundary_marker_list.append(self.vertex_boundary_markers[triangle[0]])

            if self.vertex_boundary_markers[triangle[1]] != self.vertex_boundary_markers[triangle[2]]:
                edge_boundary_marker_list.append(0)
            else:
                edge_boundary_marker_list.append(self.vertex_boundary_markers[triangle[1]])

            if self.vertex_boundary_markers[triangle[2]] != self.vertex_boundary_markers[triangle[0]]:
                edge_boundary_marker_list.append(0)
            else:
                edge_boundary_marker_list.append(self.vertex_boundary_markers[triangle[2]])

        return np.vstack(edges_list), np.array(edge_boundary_marker_list)

    def make_unique_triangulation_edges(self):
        """Get the unique edges by first sorting"""
        edges_sorted = np.copy(self._triangulation_edges_all)
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

    def _build_vertex_topology_unordered(self):
        """Build a data structure where the ith row is all vertices connected by an edge to
        vertex i"""
        num_edges = len(self.triangulation_edges)
        num_neighbors_table = np.zeros(self.num_vertices, dtype=np.int64)
        valence_upper_bound = 50
        vertex_topology = np.full((self.num_vertices, valence_upper_bound), -1, dtype=np.int64)
        for i in range(num_edges):
            v1 = self.triangulation_edges[i, 0]
            v2 = self.triangulation_edges[i, 1]
            num_neighbors_table[v1] += 1
            num_neighbors_table[v2] += 1
            vertex_topology[v1, num_neighbors_table[v1] - 1] = v2
            vertex_topology[v2, num_neighbors_table[v2] - 1] = v1
        max_valence = np.max(num_neighbors_table)
        return vertex_topology[:, :max_valence]

    def build_vertex_topology(self):
        vertex_topology_unordered = self._build_vertex_topology_unordered()
        vertex_topology = []
        for vertex_index in range(self.num_vertices):
            neighbors_ordered = self.vertex_neighbors_ordered(vertex_index, vertex_topology_unordered)
            vertex_topology.append(neighbors_ordered)

        return vertex_topology

    def _orient_vertex_topology(self):
        """Orients vertex_topology so each is oriented counterclockwise"""
        for i, topo in enumerate(self.vertex_topology):
            if topo[0] == -1:
                continue
            if not polygon_oriented_counterclockwise(topo, self.vertices):
                self.vertex_topology[i] = np.flip(topo)

    def _rebase_vertex_topology(self):
        """Set the vertex topology start point to align with the start point of each polygon"""
        for contained_poly_index in range(len(self.contained_polygons)):
            poly = self.contained_polygons[contained_poly_index]
            triangle_index = self.contained_to_original_index[contained_poly_index]

            # Find the triangle edge intersecting the first_poly_edge
            for i in range(len(self.vertex_topology[triangle_index])):
                if segment_intersects_segment(
                    self.circumcenters[poly[0]][0],
                    self.circumcenters[poly[0]][1],
                    self.circumcenters[poly[1]][0],
                    self.circumcenters[poly[1]][1],
                    self.vertices[triangle_index][0],
                    self.vertices[triangle_index][1],
                    self.vertices[self.vertex_topology[triangle_index][i]][0],
                    self.vertices[self.vertex_topology[triangle_index][i]][1]
                ):
                    # Negative to rotate clockwise instead of counterclockwise
                    self.vertex_topology[triangle_index] = np.roll(self.vertex_topology[triangle_index], -i)
                    break

    def vertex_neighbors_ordered(self, vertex, vertex_topology_unordered):
        """Take an interior vertex and return all vertices in the triangulation that are neighbors of
        the given vertex in either counterclockwise or clockwise order. Will fail SILENTLY on boundary
        vertices."""
        max_valence = len(vertex_topology_unordered[0])

        # If vertex is a boundary vertex, return vector of -1s
        if self.vertex_boundary_markers[vertex] != 0:
            return np.full(max_valence, -1)

        neighbors_extended = vertex_topology_unordered[vertex]
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
            if vertex_topology_unordered[result[0], j] in neighbors:
                result[1] = vertex_topology_unordered[result[0], j]
                break
        # Now continue chaining together elements of neighbors connected to the previously added
        # element of result
        for i in range(1, num_neighbors - 1):
            for j in range(max_valence):
                if (vertex_topology_unordered[result[i], j] in neighbors) and (vertex_topology_unordered[result[i], j] != result[i - 1]):
                    result[i + 1] = vertex_topology_unordered[result[i], j]
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
        contained_polygons_indices = [i for i in range(len(self.voronoi_tesselation)) if len(self.voronoi_tesselation[i]) != 0]
        return contained_polygons, contained_polygons_indices

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

    def find_singular_vertices(self):
        singular_vertices = []
        for vertex, neighbors in enumerate(self.vertex_topology):
            vertex_value = self.pde_values[vertex]
            sign_changes = 0
            sign_values = []  # Tracks whether neighbors are bigger or smaller than vertex_value
            if neighbors[0] != -1:  # While there is still a valid neighbor...
                for neighbor in neighbors:
                    diff = np.sign(vertex_value - self.pde_values[neighbor])
                    sign_values.append(diff)
                for j in range(0, len(sign_values) - 1):  # TODO: What if sign value is 0? This will count 2 sign changes where there should only be one
                    if sign_values[j] != sign_values[j + 1]:
                        sign_changes += 1
                if sign_values[len(sign_values) - 1] != sign_values[0]:  # Checks for sign change between start and end
                    sign_changes += 1
            if sign_changes > 2:
                singular_vertices.append(vertex)
        return singular_vertices

    def find_singular_intersecting_edges(self, singular_height_index=0):
        # Find the edges that intersect the singluar level curve, oriented so the first vertex has a lower f value
        h = self.singular_heights[singular_height_index]
        self.singular_vertices
        edge_pde_values = self.pde_values[self.triangulation_edges]
        intersecting_edges = []
        for i in range(len(edge_pde_values)):
            if edge_pde_values[i, 0] >= h and edge_pde_values[i, 1] < h:
                intersecting_edges.append(tuple(self.triangulation_edges[i]))
            elif edge_pde_values[i, 0] < h and edge_pde_values[i, 1] >= h:
                intersecting_edges.append(tuple(np.flip(self.triangulation_edges[i])))
        return intersecting_edges

    @staticmethod
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

        if path.with_suffix('.pde').is_file():
            pde_values = read_pde(path.with_suffix('.pde'))
        else:
            pde_values = None
        return Triangulation(region, vertices, vertex_boundary_markers, triangles, topology, pde_values)

    def build_conductance_dict(self):
        """Builds a dictionary mapping edges in the triangulation to their conductance, given by
        the distance between circumcenters of the adjoining triangles divided by the distance
        between the endpoints of the edge.

        Returns
        -------
        dict
            dictionary mapping edges in the triangulation to their conductance
        """
        result = {}
        for edge_index, edge in enumerate(self.triangulation_edges):
            if self.edge_boundary_markers_unique[edge_index] != 0:
                continue
            circumcenter_distance = np.linalg.norm(
                self.circumcenters[self.edge_to_right_of_triangle_dict[tuple(edge)]]
                - self.circumcenters[self.edge_to_right_of_triangle_dict[tuple(edge[::-1])]]  # Reversed edge
            )
            distance = np.linalg.norm(
                self.vertices[edge[0]] - self.vertices[edge[1]]
            )
            conductance = circumcenter_distance / distance
            result[tuple(edge)] = conductance
            result[tuple(np.flip(edge))] = conductance
        return result

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
            for i in range(len(self.triangulation_edges)):
                stream.write(
                    f'{i + 1} {self.triangulation_edges[i, 0] + 1} '
                    + f'{self.triangulation_edges[i, 1] + 1} {self.edge_boundary_markers_unique[i]}\n'
                )

            stream.write(
                f'{len(self.region.points_in_holes)}\n'
            )
            for i in range(len(self.region.points_in_holes)):
                stream.write(
                    f'{i + 1} {self.region.points_in_holes[i, 0]} {self.region.points_in_holes[i, 1]}\n'
                )

    def show(
        self,
        file_name='tmp.png',
        show_vertices=False,
        show_edges=False,
        show_triangles=True,
        show_vertex_indices=False,
        show_triangle_indices=False,
        show_level_curves=False,
        show_singular_level_curves=False,
        highlight_vertices=[],
        highlight_edges=[],
        highlight_triangles=[],
        face_color=[153/255, 204/255, 255/255],
        highlight_triangles_color=[247/255, 165/255, 131/255],
        num_level_curves=25,
        line_width=1,
        fig=None,
        axes=None,
        **kwargs
    ):
        """Show an image of the triangulation"""

        def subsample_color_map(colormap, num_samples, start_color=0, end_color=255, reverse=False):
            sample_points_float = np.linspace(start_color, end_color, num_samples)
            sample_points = np.floor(sample_points_float).astype(np.int64)
            all_colors = colormap.colors
            if reverse:
                all_colors = np.flip(all_colors, axis=0)
            return all_colors[sample_points]

        graded_level_curve_color_map = cm.lajolla
        singular_level_curve_color_map = cm.tokyo

        if fig is None and axes is None:
            fig, axes = plt.subplots()
        if show_vertices:
            axes.scatter(self.vertices[:, 0], self.vertices[:, 1])
        if show_edges:
            lines = [
                [
                    tuple(self.vertices[edge[0]]),
                    tuple(self.vertices[edge[1]])
                ] for edge in self._triangulation_edges_all
            ]
            line_collection = mc.LineCollection(lines, linewidths=line_width)
            line_collection.set_color([0/255, 255/255, 0/255])
            axes.add_collection(line_collection)
        # color_array = np.ones(self.num_triangles) * color  # np.random.random(self.num_triangles) * 500
        if highlight_edges:
            triangle_lines = [
                [
                    tuple(self.vertices[edge[0]]),
                    tuple(self.vertices[edge[1]])
                ] for edge in self.triangulation_edges
            ]
            triangle_line_collection = mc.LineCollection(triangle_lines, linewidths=2)
            triangle_line_collection.set_color(face_color)
            axes.add_collection(triangle_line_collection)
        if show_triangles:
            poly_collection = mc.PolyCollection(
                self.triangle_coordinates,
                # array=color_array, cmap=matplotlib.cm.Blues
            )
            poly_collection.set(facecolor=face_color)
            axes.add_collection(poly_collection)
        if highlight_triangles:
            triangle_coordinates = [
                np.array(
                    list(map(lambda x: self.vertices[x], self.triangles[triangle]))
                )
                for triangle in highlight_triangles
            ]
            triangle_collection = mc.PolyCollection(
                triangle_coordinates
            )
            triangle_collection.set(facecolor=highlight_triangles_color)
            axes.add_collection(triangle_collection)
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
        if show_level_curves or show_singular_level_curves:
            if len(self.singular_heights) > 0:
                singular_level_curve_colors = subsample_color_map(
                    singular_level_curve_color_map,
                    len(self.singular_heights),
                    start_color=32,
                    end_color=223,
                    reverse=True
                )
            if show_level_curves:
                heights = np.linspace(0, 1, num_level_curves + 2)[1:-1]
                colors = subsample_color_map(graded_level_curve_color_map, len(heights), end_color=240)
                if show_singular_level_curves:
                    heights = np.concatenate([heights, self.singular_heights])
                    colors = np.concatenate([colors, singular_level_curve_colors])
            else:
                if show_singular_level_curves:
                    heights = self.singular_heights
                    colors = singular_level_curve_colors

            for i, height in enumerate(heights):
                tri_level_sets_unfiltered = flatten_list_of_lists(
                    list(map(lambda i: tri_level_sets(
                        self.triangle_coordinates[i],
                        self.pde_values[self.triangles[i]],
                        [height]
                    ), range(self.num_triangles)))
                )
                tri_level_sets_filtered = [
                    line_segment for line_segment in tri_level_sets_unfiltered if len(line_segment) > 0
                ]
                lines = [
                    [
                        tuple(line_segment[0]),
                        tuple(line_segment[1])
                    ] for line_segment in tri_level_sets_filtered
                ]
                line_collection = mc.LineCollection(lines, linewidths=line_width)
                line_collection.set(color=colors[i])
                axes.add_collection(line_collection)

        axes.grid(False)
        axes.axis('off')
        axes.autoscale()
        axes.margins(0.0)
        fig.savefig(file_name, **kwargs)

    def show_voronoi_tesselation(
        self,
        file_name='tmp.png',
        show_vertex_indices=False,
        show_polygon_indices=False,
        show_vertices=False,
        show_edges=False,
        show_polygons=True,
        show_region=True,
        highlight_vertices=[],
        highlight_edges=[],
        highlight_polygons=[],
        highlight_vertices_color=[0, 1, 1],
        highlight_edges_color=[1, 1, 0],
        highlight_polygons_color=[247/255, 165/255, 131/255],
        fig=None,
        axes=None,
        **kwargs
    ):
        """Show the voronoi tesselation"""
        if fig is None and axes is None:
            fig, axes = plt.subplots()
        if show_vertices:
            axes.scatter(self.circumcenters[:, 0], self.circumcenters[:, 1])
        if highlight_vertices:
            axes.scatter(
                self.circumcenters[np.array(highlight_vertices), 0],
                self.circumcenters[np.array(highlight_vertices), 1],
                c=np.tile(highlight_vertices_color, (len(highlight_vertices), 1)),
                s=8,
                zorder=5
            )
        if show_edges:
            polygon_lines = [
                [
                    tuple(self.circumcenters[edge[0]]),
                    tuple(self.circumcenters[edge[1]])
                ] for edge in self.voronoi_edges
            ]
            polygon_line_collection = mc.LineCollection(polygon_lines, linewidths=2)
            axes.add_collection(polygon_line_collection)
        if highlight_edges:
            polygon_lines = [
                [
                    tuple(self.circumcenters[edge[0]]),
                    tuple(self.circumcenters[edge[1]])
                ] for edge in highlight_edges
            ]
            polygon_line_collection = mc.LineCollection(polygon_lines, linewidths=2)
            polygon_line_collection.set(color=highlight_edges_color)
            axes.add_collection(polygon_line_collection)
        if show_region:
            region_lines = [
                [
                    tuple(self.region.coordinates[edge[0]]),
                    tuple(self.region.coordinates[edge[1]])
                ] for edge in self.region.edges
            ]
            region_line_collection = mc.LineCollection(region_lines, linewidths=2)
            axes.add_collection(region_line_collection)
        if show_polygons:
            polygon_coordinates = [
                np.array(list(map(lambda x: self.circumcenters[x], polygon)))
                for polygon in self.contained_polygons
            ]
            polygon_collection = mc.PolyCollection(
                polygon_coordinates
            )
            polygon_collection.set(facecolor=[180/255, 213/255, 246/255])
            axes.add_collection(polygon_collection)
        if highlight_polygons:
            polygon_coordinates = [
                np.array(
                    list(map(lambda x: self.circumcenters[x], polygon))
                )
                for polygon in [self.contained_polygons[i] for i in highlight_polygons]
            ]
            polygon_collection = mc.PolyCollection(
                polygon_coordinates
            )
            polygon_collection.set(facecolor=highlight_polygons_color)
            axes.add_collection(polygon_collection)
        if show_polygon_indices:
            polygon_coordinates = [
                np.array(
                    list(map(lambda x: self.circumcenters[x], polygon))
                )
                for polygon in self.contained_polygons
            ]
            barycenters = np.array(list(map(
                lambda x: np.mean(x, axis=0),
                polygon_coordinates
            )))
            for i in range(len(barycenters)):
                plt.text(
                    barycenters[i, 0],
                    barycenters[i, 1],
                    str(i),
                    fontsize=6,
                    weight='bold',
                    zorder=7
                )
        if show_vertex_indices:
            for i in range(len(self.circumcenters)):
                plt.text(
                    self.circumcenters[i, 0],
                    self.circumcenters[i, 1],
                    str(i),
                    fontsize=6,
                    weight='bold',
                    zorder=6
                )

        axes.autoscale()
        axes.margins(0.1)
        fig.savefig(file_name)


if __name__ == '__main__':
    path = Path("regions/test/test")
    T = Triangulation.read(path)

