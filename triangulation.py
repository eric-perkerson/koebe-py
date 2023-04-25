"""This module contains the Triangulation object."""
import numpy as np
import numba
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import networkx as nx
# from matplotlib.collections import PolyCollection
from pathlib import Path

from region import read_node, read_ele, Region

COLOR_PARAMETER = 250


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
    padded_polygons = np.zeros((n_polygons, max_polygon_length), dtype=np.int32)

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
    - point: A tuple (pointX, pointY) representing the coordinates of the point
    - polygon: A 1D NumPy array containing the indices of the vertices of the polygon
    - coordinates: A 2D NumPy array containing the coordinates of the vertices of the polygon
    
    Returns:
    - A boolean indicating whether the point lies inside the polygon
    """
    true_polygon_length = np.sum(padded_polygon != 0)

    polygon_wrap_padded = np.insert(padded_polygon, 0, padded_polygon[true_polygon_length], axis=0)
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
    edges_wrapped_ordered = np.zeros((n_polygons, max_polygon_length, 2), dtype=np.int64) # (0, 0) for non-existent edges
    polygonization_topology = np.full((n_polygons, max_polygon_length), -1, dtype=np.int64) # -1 for non-existent edges
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
                if padded_polygonization[i, j] != 0:
                    # CASE : edge wraps back to beginning of polygon
                    edges_wrapped_ordered[i, j, 0] = padded_polygonization[i, j]
                    edges_wrapped_ordered[i, j, 1] = padded_polygonization[i, 0]
                else:
                    # CASE : no more edges that exist
                    break
            else:
                # j has not reached end of polygon
                if padded_polygonization[i, j + 1] == 0:
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
            if edges_wrapped_ordered[i, k, 0] == 0:
                # bread if end of polygon reached
                break

            found_flag = False
            current_edge = edges_wrapped_ordered[i, k, ::-1] # reverse edge direction
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
                boundary_edges[n_boundary_edges - 1, 0] = current_edge[1] # rememer edge was reversed 
                boundary_edges[n_boundary_edges - 1, 1] = current_edge[0] # rememer edge was reversed
    
    return n_boundary_edges, n_internal_edges, polygonization_topology, boundary_edges, internal_edges
                



class Triangulation(object):
    """Triangulation/Voronoi dual object"""
    def __init__(self, region, vertices, boundary_markers, triangles, topology):
        self.region = region
        self.vertices = vertices
        self.triangles = triangles
        self.boundary_markers = boundary_markers
        self.topology = topology

        self.triangulation_edges = self.make_triangulation_edges()
        self.triangulation_edges_unique = self.make_unique_triangulation_edges()
        self.num_vertices = len(vertices)
        self.num_triangles = len(triangles)

        self.triangle_coordinates = self.make_triangle_coordinates()
        self.barycenters = self.make_barycenters()
        self.circumcenters = self.make_circumcenters()

        self.vertex_topology = self.build_vertex_topology()

        self.vertex_index_to_triangle = self.make_vertex_index_to_triangle()
        if topology is not None:
            self.voronoi_tesselation = self.make_voronoi_tesselation()
            self.contained_polygons = self.make_contained_polygons()
            self.voronoi_edges = self.make_voronoi_edges()

    def build_slitted_weighted_voronoi_graph(
            self, 
            lambda_, # lambda[0] = verticies of nodes, lambda[1] = list of edges, lambda[2] = list of polygons
            contained_verticies, # all verticies with no boundary marker? create method to find this?
            contained_faces, # contained polygons? single indexes or list of them?
            omega0, # point we are trying to find shortest path from
            point_in_hole, #
            to_right_of_edge_poly_map # optional in mathematica code. 
            ):
        """Build the slitted weighted voronoi graph"""
        proxy_infinity = float('inf')

        # find which face contains omega0
        padded_polygonization_all = pad_polygons_to_matrix(np.array(lambda_[2]))
        omega0_face = find_point_in_polygon_compiled(omega0[0], omega0[1], padded_polygonization_all, lambda_[0])
        if omega0_face not in contained_faces:
            raise ValueError('omega0 not in contained_faces')

        # keep only the polygons that are contained in the region
        padded_polygonization = pad_polygons_to_matrix(np.take(lambda_[2], contained_faces, axis=0))
        n_polygons = len(padded_polygonization)
        max_polygon_length = len(padded_polygonization[0]) # all polygons have the same length

        # construct and parse polygonizationTopology
        n_boundary_edges, n_internal_edges, polygonization_topology, boundary_edges, internal_edges = \
            build_poly_topo_bdryEdges_intEdges_compiled(padded_polygonization)

        # construct connected components from boundary edges
        classes = boundary_connected_classes_unseeded_compiled[boundary_edges]
        class_1 = classes[0, 1 : classes[0, 0] + 1]
        class_2 = classes[1, 1 : classes[1, 0] + 1]

        # find winding number of point in hole (indicies in the given list of coordinates)
        winding_1 = winding_number_compiled(point_in_hole[0], point_in_hole[1], class_1, lambda_[0])
        outer_ring_vertices, inner_ring_vertices = class_1, class_2
        if winding_1 <= 0:
            outer_ring_vertices = class_2
            inner_ring_vertices = class_1

        # create inner and outer ring faces
        inner_ring_faces = None
        outer_ring_faces = None

        # create contained faces graph (networkx) and shortest path function
        contained_faces_graph = nx.Graph()
        shortest_paths = nx.single_source_shortest_path(contained_faces_graph, omega0_face) # TODO: check if this is correct
        paths_to_inner_ring = []
        paths_to_outer_ring = []
        for face in contained_faces_graph:
            if face in inner_ring_faces:
                paths_to_inner_ring.append(shortest_paths[face])
            elif face in outer_ring_faces:
                paths_to_outer_ring.append(shortest_paths[face])

        # construct the slit


        # omega0n candidates

        # add weights to edges (makeing sure to use inf for edges of slit)

        # construct the final graph

        

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
        return np.vstack(edges_list)

    def make_unique_triangulation_edges(self):
        """Get the unique edges by first sorting"""
        # TODO: TEST THIS
        edges_sorted = np.sort(self.triangulation_edges, axis=0)
        edges_unique = np.zeros((len(edges_sorted), 2), dtype=np.int64)
        edges_unique_length = 0
        for i in range(len(edges_sorted) - 1):
            if (edges_sorted[i, 0] != edges_sorted[i + 1, 0]) or (edges_sorted[i, 1] != edges_sorted[i + 1, 1]):
                edges_unique[edges_unique_length, 0] = edges_sorted[i, 0]
                edges_unique[edges_unique_length, 1] = edges_sorted[i, 1]
                edges_unique_length += 1
        edges_unique = edges_unique[:edges_unique_length]
        return edges_unique

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
                if self.boundary_markers[vertex] == 0:  # If vertex is an interior vertex
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
            if self.boundary_markers[vertex] == 0:
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
        vertices, boundary_markers = read_node(path.with_suffix('.node'))
        triangles = read_ele(path.with_suffix('.ele'))
        if path.with_suffix('.topo.ele').is_file():
            topology = read_ele(path.with_suffix('.topo.ele'))
        else:
            topology = None
        return Triangulation(region, vertices, boundary_markers, triangles, topology)

    def show(self, file_name, show_vertex_indices=False, show_triangle_indices=False):
        """Show an image of the triangulation"""
        fig, axes = plt.subplots()
        axes.scatter(self.vertices[:, 0], self.vertices[:, 1])
        lines = [
            [
                tuple(self.vertices[edge[0]]),
                tuple(self.vertices[edge[1]])
            ] for edge in self.triangulation_edges
        ]
        line_collection = mc.LineCollection(lines, linewidths=2)
        color_array = np.ones(self.num_triangles) * COLOR_PARAMETER  # np.random.random(self.num_triangles) * 500
        poly_collection = mc.PolyCollection(
            self.triangle_coordinates,
            array=color_array, cmap=matplotlib.cm.Blues
        )
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

    def show_voronoi_tesselation(self, show_vertex_indices=False, show_polygon_indices=False):
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
        num_polygons = len(self.contained_polygons)
        color_array = np.ones(num_polygons) * COLOR_PARAMETER  # np.random.random(num_polygons) * 500
        polygon_coordinates = [
            np.array(list(map(lambda x: self.circumcenters[x], polygon)))
            for polygon in self.contained_polygons
        ]
        polygon_collection = mc.PolyCollection(
            polygon_coordinates,
            array=color_array,
            cmap=matplotlib.cm.Blues
        )

        axes.add_collection(polygon_line_collection)
        axes.add_collection(region_line_collection)
        axes.add_collection(polygon_collection)
        if show_polygon_indices:
            barycenters = np.array(list(map(
                lambda x: np.mean(x, axis=0),
                polygon_coordinates
            )))
            for i in range(len(barycenters)):
                plt.text(barycenters[i, 0], barycenters[i, 1], str(i))
        if show_vertex_indices:
            for i in range(len(self.circumcenters)):
                plt.text(self.circumcenters[i, 0], self.circumcenters[i, 1], str(i))
        axes.autoscale()
        axes.margins(0.1)
        axes.scatter(self.circumcenters[:, 0], self.circumcenters[:, 1])
        fig.show()


if __name__ == '__main__':
    path = Path('/Users/eric/Code/combinatorial-topology/regions/example/example.1.poly')
    T = Triangulation.read(path)
    T.show_triangulation()
