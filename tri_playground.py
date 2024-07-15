from region import Region
from triangulation import (
    Triangulation,
    point_to_right_of_line_compiled,
)
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm
import subprocess
from region import Region
import pyvista
from matplotlib import collections as mc
import numba
import networkx as nx
import itertools


@numba.njit
def get_first_unused(already_used):
    for i in range(len(already_used)):
        if not already_used[i]:
            return i
    return -1


def add_edges_to_axes(edge_list, axes, color):
    lines = [
        [
            tuple(tri.vertices[edge[0]]),
            tuple(tri.vertices[edge[1]])
        ] for edge in edge_list
    ]
    colors = np.tile(color, (len(edge_list), 1))
    line_collection = mc.LineCollection(lines, linewidths=2, colors=colors)
    axes.add_collection(line_collection)


NUM_TRIANGLES = 1000
USE_WOLFRAM_SOLVER = True

file_stem = "vertex18"
# file_stem = 'No_3_fold_sym'
# file_stem = '3_fold_sym'
path = Path(f'regions/{file_stem}/{file_stem}')
# tri = Triangulation.read(path)

subprocess.run([
    'julia',
    'triangulate_via_julia.jl',
    file_stem,
    file_stem,
    str(NUM_TRIANGLES)
])

if USE_WOLFRAM_SOLVER:
    subprocess.run([
        'wolframscript',
        'solve_pde.wls'
    ])
else:
    t = Triangulation.read(f'regions/{file_stem}/{file_stem}.poly')
    t.write(f'regions/{file_stem}/{file_stem}.output.poly')

    subprocess.run([
        'python',
        'mesh_conversion/mesh_conversion.py',
        '-p',
        f'regions/{file_stem}/{file_stem}.output.poly',
        '-n',
        f'regions/{file_stem}/{file_stem}.node',
        '-e',
        f'regions/{file_stem}/{file_stem}.ele',
    ])

#     subprocess.run([
#         'python',
#         'mesh_conversion/fenicsx_solver.py',
#         file_stem,
#     ])

tri = Triangulation.read(f'regions/{file_stem}/{file_stem}.poly')

# TEMPORARY
tri.show(
    'test.png',
    show_level_curves=True,
    show_triangles=True,
    show_edges=True
)
plt.show()

from triangulation import triangle_area
area_values = [
    triangle_area(tri.triangle_coordinates[i]) for i in range(tri.num_triangles)
]
np.max(area_values)
np.min(area_values)
np.max(area_values) / np.min(area_values)

# END TEMPORARY


singular_height_index = 0
intersecting_edges = tri.find_singular_intersecting_edges(singular_height_index)

# Push outer boundary back by one
boundary_edge_dict = {
    1: [],
    2: [],
    3: [],
}
for boundary_marker in [1, 2, 3]:
    for edge in tri.triangulation_edges:
        if (
            (tri.vertex_boundary_markers[edge[0]] == boundary_marker)
            ^ (tri.vertex_boundary_markers[edge[1]] == boundary_marker)  # Use XOR here to exclude pure boundary edges
        ):
            boundary_edge_dict[boundary_marker].append(edge)

tri.show(
    str(path.with_suffix('.png')),
    show_level_curves=False,
    show_edges=True,
    show_singular_level_curves=True,
    show_vertex_indices=False,
    dpi=300,
    num_level_curves=500,
    line_width=0.75
)
axes = plt.gca()
add_edges_to_axes(intersecting_edges, axes, color=[1, 0, 1])
add_edges_to_axes(boundary_edge_dict[1], axes, color=[1, 0, 0])
add_edges_to_axes(boundary_edge_dict[2], axes, color=[0, 1, 0])
add_edges_to_axes(boundary_edge_dict[3], axes, color=[0, 0, 1])
plt.show()


def flux_on_contributing_edges(edges):
    flux = 0.0
    for edge in edges:
        flux += tri.conductance[edge] * np.abs(
            tri.pde_values[edge[0]] - tri.pde_values[edge[1]]
        )
    return flux


flux_on_contributing_edges(intersecting_edges)
flux_on_contributing_edges([tuple(edge) for edge in boundary_edge_dict[1]])
(
    flux_on_contributing_edges([tuple(edge) for edge in boundary_edge_dict[2]])
    + flux_on_contributing_edges([tuple(edge) for edge in boundary_edge_dict[3]])
)

# Find connected components using the lower pde value for each intersecting edge
lower_vertices = np.unique([edge[0] if tri.pde_values[edge[0]] < tri.singular_heights[singular_height_index] else edge[1] for edge in intersecting_edges])
if np.any(tri.vertex_boundary_markers[lower_vertices] != 0):
    raise Exception('lower_vertices intersects the boundary, vertex topology will not be fully initialized')


component_vertices_1 = [lower_vertices[0]]
already_used = np.zeros(len(lower_vertices), dtype=np.bool_)
already_used[0] = True

break_flag = False
while not break_flag:
    break_flag = True
    for component_vertex in component_vertices_1:
        for neighboring_vertex in tri.vertex_topology[component_vertex]:
            for index, test_vertex in enumerate(lower_vertices):
                if already_used[index]:
                    continue
                if test_vertex == neighboring_vertex:
                    print(test_vertex)
                    component_vertices_1.append(test_vertex)
                    already_used[index] = True
                    break_flag = False


component_vertices_2 = lower_vertices[np.where(~already_used)[0]]

tri.show(
    str(path.with_suffix('.png')),
    show_level_curves=False,
    show_singular_level_curves=True,
    show_vertex_indices=False,
    dpi=500,
    num_level_curves=500,
    line_width=0.75
)
axes = plt.gca()
plt.scatter(
    tri.vertices[lower_vertices][:, 0],
    tri.vertices[lower_vertices][:, 1],
    s=25,
    color=[1, 0, 0]
)
plt.scatter(
    tri.vertices[component_vertices_1][:, 0],
    tri.vertices[component_vertices_1][:, 1],
    s=10,
    color=[0, 1, 0]
)
plt.scatter(
    tri.vertices[component_vertices_2][:, 0],
    tri.vertices[component_vertices_2][:, 1],
    s=10,
    color=[0, 0, 1]
)
plt.show()


component_edges_1 = []
for edge in intersecting_edges:
    if (edge[0] in component_vertices_1) or (edge[1] in component_vertices_1):
        component_edges_1.append(edge)

component_edges_2 = []
for edge in intersecting_edges:
    if (edge[0] in component_vertices_2) or (edge[1] in component_vertices_2):
        component_edges_2.append(edge)


flux_on_contributing_edges([tuple(edge) for edge in boundary_edge_dict[2]])
flux_on_contributing_edges([tuple(edge) for edge in component_edges_1])
flux_on_contributing_edges([tuple(edge) for edge in boundary_edge_dict[3]])
flux_on_contributing_edges([tuple(edge) for edge in component_edges_2])

tri.show(
    str(path.with_suffix('.png')),
    show_level_curves=False,
    show_singular_level_curves=True,
    show_vertex_indices=False,
    dpi=500,
    num_level_curves=500,
    line_width=0.75
)
axes = plt.gca()
add_edges_to_axes(component_edges_1, axes, color=[1, 0, 0])
add_edges_to_axes(component_edges_2, axes, color=[0, 1, 1])
plt.scatter(
    tri.vertices[tri.singular_vertices][0][0],
    tri.vertices[tri.singular_vertices][0][1],
    s=10,
    color=[0, 0, 1]
)
plt.show()




# Uniformize each piece of the genus 2 surface by breaking into 3 triangulation objects
plt.scatter(
    tri.region.coordinates[:, 0],
    tri.region.coordinates[:, 1]
)
plt.show()

# Find connected components using the lower pde value for each intersecting edge
upper_vertices = np.unique([edge[0] if tri.pde_values[edge[0]] >= tri.singular_heights[singular_height_index] else edge[1] for edge in intersecting_edges])
if np.any(tri.vertex_boundary_markers[upper_vertices] != 0):
    raise Exception('lower_vertices intersects the boundary, vertex topology will not be fully initialized')


plt.scatter(
    tri.region.coordinates[:, 0],
    tri.region.coordinates[:, 1]
)
plt.scatter(
    tri.vertices[upper_vertices][:, 0],
    tri.vertices[upper_vertices][:, 1]
)
plt.show()

upper_region = Region(
    coordinates,
    vertex_boundary_markers,
    edges,
    edge_boundary_markers,
    points_in_holes,
    components=None
)
upper_triangulation = Triangulation(region, vertices, vertex_boundary_markers, triangles, topology, pde_values)
tri




# # from region import Region
# # domain = Region.region_from_components(
# #     [
# #         [
# #             (2.0, 0.0),
# #             (1.0000000000000002, 1.7320508075688772),
# #             (-0.9999999999999996, 1.7320508075688776),
# #             (-2.0, 2.4492935982947064e-16),
# #             (-1.0000000000000009, -1.7320508075688767),
# #             (1.0, -1.7320508075688772)
# #         ],
# #         [
# #             (0.9000000000000001, 2.4492935982947065e-17),
# #             (1.0, 0.17320508075688773),
# #             (1.2000000000000002, 0.17320508075688776),
# #             (1.3, 0.0),
# #             (1.2000000000000002, -0.1732050807568877),
# #             (1.0000000000000002, -0.1732050807568878)
# #         ],
# #         [
# #             (-0.7499999999999998, 0.9526279441628828),
# #             (-0.6499999999999999, 1.1258330249197706),
# #             (-0.44999999999999984, 1.1258330249197706),
# #             (-0.3499999999999998, 0.9526279441628828),
# #             (-0.44999999999999973, 0.7794228634059951),
# #             (-0.6499999999999997, 0.779422863405995)
# #         ],
# #         [
# #             (-0.7500000000000004, -0.9526279441628823),
# #             (-0.6500000000000006, -0.7794228634059945),
# #             (-0.4500000000000005, -0.7794228634059945),
# #             (-0.3500000000000005, -0.9526279441628823),
# #             (-0.4500000000000004, -1.12583302491977),
# #             (-0.6500000000000004, -1.12583302491977)
# #         ]
# #     ]
# # )


# Annulus
from region import Region
from triangulation import (
    Triangulation,
    point_to_right_of_line_compiled,
    polygon_oriented_counterclockwise,
    segment_intersects_segment,
    tri_level_sets
)
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm
import subprocess
from region import Region
import pyvista
from matplotlib import collections as mc
import numba
import networkx as nx
import itertools


def flux_on_contributing_edges(edges):
    flux = 0.0
    for edge in edges:
        flux += tri.conductance[edge] * np.abs(
            tri.pde_values[edge[0]] - tri.pde_values[edge[1]]
        )
    return flux


NUM_TRIANGLES = 1000
USE_WOLFRAM_SOLVER = True

base_cell = 149  # 178
file_stem = "concentric_annulus"

# subprocess.run([
#     'julia',
#     'triangulate_via_julia.jl',
#     file_stem,
#     file_stem,
#     str(NUM_TRIANGLES)
# ])

# if USE_WOLFRAM_SOLVER:
#     subprocess.run([
#         'wolframscript',
#         'solve_pde.wls'
#     ])
# else:
#     t = Triangulation.read(f'regions/{file_stem}/{file_stem}.poly')
#     t.write(f'regions/{file_stem}/{file_stem}.output.poly')

#     subprocess.run([
#         'python',
#         'mesh_conversion/mesh_conversion.py',
#         '-p',
#         f'regions/{file_stem}/{file_stem}.output.poly',
#         '-n',
#         f'regions/{file_stem}/{file_stem}.node',
#         '-e',
#         f'regions/{file_stem}/{file_stem}.ele',
#     ])

#     subprocess.run([
#         'python',
#         'mesh_conversion/fenicsx_solver.py',
#         file_stem,
#     ])

path = Path(f'regions/{file_stem}/{file_stem}')
tri = Triangulation.read(f'regions/{file_stem}/{file_stem}.poly')
hole_x, hole_y = tri.region.points_in_holes[0]

# Define base_point to use along with the point_in_hole to define the ray to determine the slit
base_point = tri.vertices[tri.contained_to_original_index[base_cell]]
for cell in tri.voronoi_tesselation:
    for vertex in cell:
        point_x, point_y = tri.circumcenters[vertex]
        to_right = point_to_right_of_line_compiled(hole_x, hole_y, base_point[0], base_point[1], point_x, point_y)


def segment_intersects_line(tail, head):
    tail_to_right = point_to_right_of_line_compiled(
        hole_x,
        hole_y,
        base_point[0],
        base_point[1],
        tail[0],
        tail[1]
    )
    head_to_right = point_to_right_of_line_compiled(
        hole_x,
        hole_y,
        base_point[0],
        base_point[1],
        head[0],
        head[1]
    )
    return head_to_right ^ tail_to_right


def segment_intersects_line_positive(tail, head):
    tail_to_right = point_to_right_of_line_compiled(hole_x, hole_y, base_point[0], base_point[1], tail[0], tail[1])
    head_to_right = point_to_right_of_line_compiled(hole_x, hole_y, base_point[0], base_point[1], head[0], head[1])
    return (head_to_right and not tail_to_right)


def segment_intersects_line_negative(tail, head):
    tail_to_right = point_to_right_of_line_compiled(hole_x, hole_y, base_point[0], base_point[1], tail[0], tail[1])
    head_to_right = point_to_right_of_line_compiled(hole_x, hole_y, base_point[0], base_point[1], head[0], head[1])
    return (not head_to_right and tail_to_right)


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


# Create the contained topology
contained_topology_all = [
    [
        tri.original_to_contained_index[vertex]
        if vertex in tri.contained_to_original_index
        else -1
        for vertex in cell
    ] for cell in tri.vertex_topology
]
contained_topology = [contained_topology_all[i] for i in tri.contained_to_original_index]

# # Show setup with line from point_in_hole to base_point
# tri.show_voronoi_tesselation(
#     'voronoi.png',
#     show_vertex_indices=False,
#     show_polygon_indices=False,
#     show_edges=True,
#     # highlight_polygons=cell_path
# )
# plt.scatter(
#     [base_point[0]],
#     [base_point[1]],
#     c=[[0, 1, 1]],
# )
# plt.scatter(
#     [hole_x],
#     [hole_y],
#     c=[[1, 0, 0]]
# )
# hole_point = np.array([hole_x, hole_y])
# line_segment_end = 2 * (base_point - hole_point) + hole_point
# lines = [
#     [
#         tuple(line_segment_end),
#         tuple(hole_point)
#     ]
# ]
# line_collection = mc.LineCollection(lines, linewidths=2)
# line_collection.set(color=[1, 0, 0])
# axes = plt.gca()
# axes.add_collection(line_collection)
# plt.show()


# Create cell path from base_cell to boundary_1
poly = base_cell
poly_path_outward = []
while poly != -1:
    cell_vertices = tri.contained_polygons[poly]
    edges = build_polygon_edges(cell_vertices)
    for i, edge in enumerate(edges):
        if segment_intersects_line_negative(
                tri.circumcenters[edge[0]],
                tri.circumcenters[edge[1]]
        ):
            poly_path_outward.append(poly)
            poly = contained_topology[poly][i]

# Create cell path from base_cell to boundary_0
poly = base_cell
poly_path_inward = []
while poly != -1:
    cell_vertices = tri.contained_polygons[poly]
    edges = build_polygon_edges(cell_vertices)
    for i, edge in enumerate(edges):
        if segment_intersects_line_positive(
                tri.circumcenters[edge[0]],
                tri.circumcenters[edge[1]]
        ):
            poly_path_inward.append(poly)
            poly = contained_topology[poly][i]

# Create slit cell path by joining
poly_path_inward = poly_path_inward[1:]
poly_path_inward.reverse()
cell_path = poly_path_inward + poly_path_outward

# Create poly edge path to left of line, starting at the outer boundary going to the inner boundary
connected_component = []
perpendicular_edges = []
for cell_path_index, cell in enumerate(reversed(cell_path)):
    flag = False
    edges = tri.make_polygon_edges(tri.contained_polygons[cell])
    num_edges = len(edges)
    edge_index = -1
    while True:
        edge_index = (edge_index + 1) % num_edges  # Next edge
        edge = edges[edge_index]
        if flag:
            if (not segment_intersects_line_positive(
                tri.circumcenters[edge[0]],
                tri.circumcenters[edge[1]]
            )):
                if (contained_topology[cell][edge_index] != -1):  # Might remove this depending on which path is needed
                    connected_component.append(edge)
                    perpendicular_edges.append((cell, contained_topology[cell][edge_index]))
            else:
                break
        if segment_intersects_line_negative(
            tri.circumcenters[edge[0]],
            tri.circumcenters[edge[1]]
        ):
            flag = True


# Edges to weight
edges_to_weight = []
for cell_path_index, cell in enumerate(reversed(cell_path)):
    edges = tri.make_polygon_edges(tri.contained_polygons[cell])
    for edge in edges:
        if segment_intersects_line(tri.circumcenters[edge[0]], tri.circumcenters[edge[1]]):
            edges_to_weight.append(edge)
edges_to_weight = list(set(map(lambda x: tuple(np.sort(x)), edges_to_weight)))


# tri.show_voronoi_tesselation(
#     'test.png',
#     highlight_polygons=cell_path,
#     show_polygon_indices=False,
#     show_vertex_indices=False,
#     show_edges=True
# )
# plt.scatter(
#     [base_point[0]],
#     [base_point[1]],
#     c=[[0, 1, 1]],
# )
# plt.scatter(
#     [hole_x],
#     [hole_y],
#     c=[[1, 0, 0]]
# )
# hole_point = np.array([hole_x, hole_y])
# line_segment_end = 2 * (base_point - hole_point) + hole_point
# lines = [
#     [
#         tuple(line_segment_end),
#         tuple(hole_point)
#     ]
# ]
# line_collection = mc.LineCollection(lines, linewidths=2)
# line_collection.set(color=[1, 0, 0])
# axes = plt.gca()
# axes.add_collection(line_collection)
# plt.show()


# Create contained_edges
triangulation_edges_reindexed = tri.original_to_contained_index[tri.triangulation_edges]
contained_edges = []
for edge in triangulation_edges_reindexed:
    if -1 not in edge:
        contained_edges.append(list(edge))

# # Create graph of cells
# lambda_graph = nx.Graph()
# lambda_graph.add_nodes_from(range(len(tri.contained_polygons)))
# lambda_graph.add_edges_from(contained_edges)
# nx.set_edge_attributes(lambda_graph, values=1, name='weight')
# for edge in perpendicular_edges:
#     lambda_graph.edges[edge[0], edge[1]]['weight'] = np.finfo(np.float32).max
# shortest_paths = nx.single_source_dijkstra(lambda_graph, base_cell, target=None, cutoff=None, weight='weight')[1]

# Choose omega_0 as the slit vertex that has the smallest angle relative to the line from the point in hole through
# the circumcenter of the base_cell

# TODO: rotate to avoid having the negative x-axis near the annulus slit
slit_path = [edge[0] for edge in connected_component]
slit_path.append(connected_component[-1][1])  # TODO: Why?
# Connected component goes from outer boundary to inner boundary. Reverse after making slit
slit_path = list(reversed(slit_path))
angles = np.array([
    np.arctan2(
        tri.circumcenters[vertex][1] - hole_x,
        tri.circumcenters[vertex][0] - hole_y
    )
    for vertex in slit_path
])
omega_0 = slit_path[np.argmin(angles)]


def add_voronoi_edges_to_axes(edge_list, axes, color):
    lines = [
        [
            tuple(tri.circumcenters[edge[0]]),
            tuple(tri.circumcenters[edge[1]])
        ] for edge in edge_list
    ]
    colors = np.tile(color, (len(edge_list), 1))
    line_collection = mc.LineCollection(lines, linewidths=2, colors=colors)
    axes.add_collection(line_collection)


# tri.show_voronoi_tesselation(
#     'test.png',
#     show_polygon_indices=False,
#     show_vertex_indices=True,
#     show_edges=True
# )
# axes = plt.gca()
# add_voronoi_edges_to_axes(connected_component, axes, [1, 1, 0])
# hole_point = np.array([hole_x, hole_y])
# line_segment_end = 2 * (base_point - hole_point) + hole_point
# lines = [
#     [
#         tuple(line_segment_end),
#         tuple(hole_point)
#     ]
# ]
# line_collection = mc.LineCollection(lines, linewidths=2)
# line_collection.set(color=[1, 0, 0])
# axes.add_collection(line_collection)
# edges_to_weight_coordinates = [
#     [
#         tuple(tri.circumcenters[edge[0]]),
#         tuple(tri.circumcenters[edge[1]])
#     ]
#     for edge in edges_to_weight
# ]
# edges_to_weight_collection = mc.LineCollection(edges_to_weight_coordinates, linewidths=2)
# edges_to_weight_collection.set(color=[247/255, 165/255, 131/255])
# axes.add_collection(edges_to_weight_collection)
# plt.scatter(
#     tri.circumcenters[omega_0][0],
#     tri.circumcenters[omega_0][1],
#     s=25,
#     color=[0, 0.5, 0.5],
#     zorder=5
# )
# plt.show()

# Create graph of circumcenters (Lambda[0])
lambda_graph = nx.Graph()
lambda_graph.add_nodes_from(range(len(tri.circumcenters)))
lambda_graph.add_edges_from(tri.voronoi_edges)
nx.set_edge_attributes(lambda_graph, values=1, name='weight')
for edge in edges_to_weight:
    lambda_graph.edges[edge[0], edge[1]]['weight'] = np.finfo(np.float32).max
shortest_paths = nx.single_source_dijkstra(lambda_graph, omega_0, target=None, cutoff=None, weight='weight')[1]

# # Show the graph of circumcenters with edge weights
# tri.show_voronoi_tesselation(
#     'voronoi.png',
#     show_vertex_indices=True,
#     show_polygon_indices=False,
#     show_edges=True,
#     highlight_polygons=cell_path
# )
# pos = {i: tri.circumcenters[i] for i in range(len(tri.circumcenters))}
# nx.draw(lambda_graph, pos, node_size=10)
# plt.axis([-2.0, 2.0, -2.0, 2.0])
# labels = nx.get_edge_attributes(lambda_graph, 'weight')
# nx.draw_networkx_edge_labels(lambda_graph, pos, edge_labels=labels)
# plt.show()


# DEPRECATED
# # Show the graph of cells with edge weights
# tri.show_voronoi_tesselation(
#     'voronoi.png',
#     show_vertex_indices=False,
#     show_polygon_indices=False,
#     show_edges=True,
# )
# polygon_coordinates = [
#     np.array(
#         list(map(lambda x: tri.circumcenters[x], polygon))
#     )
#     for polygon in tri.contained_polygons
# ]
# barycenters = np.array(list(map(
#     lambda x: np.mean(x, axis=0),
#     polygon_coordinates
# )))
# pos = {i: barycenters[i] for i in range(len(barycenters))}
# nx.draw(lambda_graph, pos, node_size=10)
# plt.axis([-2.0, 2.0, -2.0, 2.0])
# labels = nx.get_edge_attributes(lambda_graph, 'weight')
# nx.draw_networkx_edge_labels(lambda_graph, pos, edge_labels=labels)
# plt.show()


# Find the perpendicular edges to the lambda path
def build_path_edges(vertices):
    return [[vertices[i], vertices[i + 1]] for i in range(len(vertices) - 1)]


# # Make poly_to_right_of_edge dict
# poly_to_right_of_edge = {}
# for i, poly in enumerate(tri.contained_polygons):
#     edges = tri.make_polygon_edges(poly)
#     for edge in edges:
#         poly_to_right_of_edge[tuple(edge)] = i

# Make mapping from edges on the cells to the perpendicular edges in triangulation
@numba.njit
def position(x, array):
    for i in range(len(array)):
        if x == array[i]:
            return i
    return -1


def get_perpendicular_edge(edge):
    """Think of the omega path as being triangles instead. This finds which edge of the triangle
    edge[0] is adjacent to triangle edge[1]"""
    triangle_edges = tri.make_polygon_edges(tri.triangles[edge[0]])
    edge_index = position(edge[1], tri.topology[edge[0]])
    perpendicular_edge = triangle_edges[edge_index]
    return perpendicular_edge


num_contained_polygons = len(tri.contained_polygons)
g_star_bar = np.zeros(tri.num_triangles, dtype=np.float64)
perpendicular_edges_dict = {}
for omega in range(tri.num_triangles):
    if omega in shortest_paths:
        edges = build_path_edges(shortest_paths[omega])
    else:
        edges = []
    flux_contributing_edges = []
    for edge in edges:
        flux_contributing_edges.append(tuple(get_perpendicular_edge(edge)))
    perpendicular_edges_dict[omega] = flux_contributing_edges
    g_star_bar[omega] = flux_on_contributing_edges(flux_contributing_edges)


def compute_period():
    omega_0_cross_ray_edge_position = position(True, np.array([(omega_0 in edge) for edge in edges_to_weight]))
    omega_0_cross_ray_edge = tuple(edges_to_weight[omega_0_cross_ray_edge_position])
    if omega_0_cross_ray_edge[1] == omega_0:
        omega_0_clockwise_neighbor = omega_0_cross_ray_edge[0]
    else:
        omega_0_clockwise_neighbor = omega_0_cross_ray_edge[1]

    last_flux_contributing_edge = tuple(get_perpendicular_edge(omega_0_cross_ray_edge))
    closed_loop_flux = g_star_bar[omega_0_clockwise_neighbor] + (
        tri.conductance[last_flux_contributing_edge] * np.abs(
            tri.pde_values[last_flux_contributing_edge[0]] - tri.pde_values[last_flux_contributing_edge[1]]
        )
    )
    return closed_loop_flux


# # Show shortest paths for a particular circumcenter
# omega = 419
# fig, axes = plt.subplots()
# tri.show_voronoi_tesselation(
#     'voronoi.png',
#     show_vertex_indices=False,
#     show_polygon_indices=False,
#     show_edges=True,
#     highlight_vertices=shortest_paths[omega],
#     show_polygons=False,
#     fig=fig,
#     axes=axes
# )
# # axes = plt.gca()
# add_voronoi_edges_to_axes(build_path_edges(shortest_paths[omega]), axes, color=[1, 0, 0])
# tri.show(
#     show_vertex_indices=False,
#     show_triangle_indices=False,
#     show_edges=True,
#     show_triangles=False,
#     fig=fig,
#     axes=axes,
# )
# add_edges_to_axes(perpendicular_edges_dict[omega], axes, [0, 1, 0])
# axes = plt.gca()
# add_voronoi_edges_to_axes(connected_component, axes, [1, 1, 0])
# plt.show()


# Interpolate the value of pde_solution to get its values on the omegas
@numba.njit
def cartesian_to_barycentric(x, y, x_1, y_1, x_2, y_2, x_3, y_3):
    det_T_inverse = 1 / ((x_1 - x_3) * (y_2 - y_3) + (x_3 - x_2) * (y_1 - y_3))
    lambda_1 = ((y_2 - y_3) * (x - x_3) + (x_3 - x_2) * (y - y_3)) * det_T_inverse
    lambda_2 = ((y_3 - y_1) * (x - x_3) + (x_1 - x_3) * (y - y_3)) * det_T_inverse
    return lambda_1, lambda_2


@numba.njit
def barycentric_to_cartesian(lambda_1, lambda_2, x_1, y_1, x_2, y_2, x_3, y_3):
    lambda_3 = 1 - lambda_1 - lambda_2
    x = lambda_1 * x_1 + lambda_2 * x_2 + lambda_3 * x_3
    y = lambda_1 * y_1 + lambda_2 * y_2 + lambda_3 * y_3
    return x, y


@numba.njit
def barycentric_interpolation(x, y, x_1, y_1, x_2, y_2, x_3, y_3, f_1, f_2, f_3):
    lambda_1, lambda_2 = cartesian_to_barycentric(x, y, x_1, y_1, x_2, y_2, x_3, y_3)
    lambda_3 = 1 - lambda_1 - lambda_2
    return lambda_1 * f_1 + lambda_2 * f_2 + lambda_3 * f_3


# # START BARY TO CART TESTING
# root_3_over_2 = np.sqrt(3) / 2
# r_1 = np.array([0.0, 1.0])
# r_2 = np.array([-root_3_over_2, -0.5])
# r_3 = np.array([root_3_over_2, -0.5])
# n = 5
# barycentric_coor_grid = np.vstack([np.array([i, j]) / (n - 1) for i in range(n) for j in range(n - i)])
# plt.scatter(
#     barycentric_coor_grid[:, 0],
#     barycentric_coor_grid[:, 1]
# )
# plt.show()

# cartesian_coor_tri_grid = np.vstack(
#     [
#         lambda_1 * r_1 + lambda_2 * r_2 + (1 - lambda_1 - lambda_2) * r_3
#         for lambda_1, lambda_2 in barycentric_coor_grid
#     ]
# )
# plt.scatter(
#     cartesian_coor_tri_grid[:, 0],
#     cartesian_coor_tri_grid[:, 1]
# )
# plt.scatter(
#     [r_1[0], r_2[0], r_3[0]],
#     [r_1[1], r_2[1], r_3[1]],
#     c='r',
#     alpha=0.5
# )
# plt.show()

# for lambda_1, lambda_2 in barycentric_coor_grid:
#     lambda_3 = 1 - lambda_1 - lambda_2
#     print(f'Barycentric coordinates are: ({lambda_1}, {lambda_2}, {lambda_3})')
#     x, y = barycentric_to_cartesian(lambda_1, lambda_2, r_1[0], r_1[1], r_2[0], r_2[1], r_3[0], r_3[1])
#     print(f'Cartesian coordinates are: ({x}, {y})')
#     lambda_hat_1, lambda_hat_2 = cartesian_to_barycentric(x, y, r_1[0], r_1[1], r_2[0], r_2[1], r_3[0], r_3[1])
#     lambda_hat_3 = lambda_hat_1 + lambda_hat_2
#     print(f'Recovered barycentric coordinates are: ({lambda_hat_1}, {lambda_hat_2}, {lambda_hat_3})')
#     print()
# END BARY TO CART TESTING

# # Test barycentric interpolation
# root_3_over_2 = np.sqrt(3) / 2
# r_1 = np.array([0.0, 1.0])
# r_2 = np.array([-root_3_over_2, -0.5])
# r_3 = np.array([root_3_over_2, -0.5])
# n = 10
# barycentric_coor_grid = np.vstack([np.array([i, j]) / (n - 1) for i in range(n) for j in range(n - i)])
# cartesian_coor_tri_grid = np.vstack(
#     [
#         barycentric_to_cartesian(lambda_1, lambda_2, r_1[0], r_1[1], r_2[0], r_2[1], r_3[0], r_3[1])
#         for lambda_1, lambda_2 in barycentric_coor_grid
#     ]
# )

# z_values = np.array([
#     barycentric_interpolation(
#         x, y,
#         r_1[0], r_1[1],
#         r_2[0], r_2[1],
#         r_3[0], r_3[1],
#         15.0,
#         0.0,
#         30.0
#     )
#     for x, y in cartesian_coor_tri_grid
# ])
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(
#     cartesian_coor_tri_grid[:, 0],
#     cartesian_coor_tri_grid[:, 1],
#     z_values,
# )
# plt.show()
# # END TEST OF BARYCENTRIC INTERPOLATION


pde_on_omega_values = [
    barycentric_interpolation(
        tri.circumcenters[i][0], tri.circumcenters[i][1],
        tri.triangle_coordinates[i][0][0], tri.triangle_coordinates[i][0][1],
        tri.triangle_coordinates[i][1][0], tri.triangle_coordinates[i][1][1],
        tri.triangle_coordinates[i][2][0], tri.triangle_coordinates[i][2][1],
        tri.pde_values[tri.triangles[i][0]],
        tri.pde_values[tri.triangles[i][1]],
        tri.pde_values[tri.triangles[i][2]],
    ) for i in range(tri.num_triangles)
]

# # Plot the interpolated PDE values
# tri.show_voronoi_tesselation(
#     'test.png',
#     show_vertex_indices=False
# )
# plt.scatter(
#     tri.circumcenters[:, 0],
#     tri.circumcenters[:, 1],
#     c=pde_on_omega_values,
#     s=100
# )
# plt.colorbar()
# plt.show()

i = 57
tri.show(
    'test.png',
    show_edges=True,
    show_triangles=False,
    show_vertex_indices=True,
    highlight_triangles=[i]
)
tri.show_voronoi_tesselation(
    'test.png',
    show_edges=True,
    show_polygons=False,
    highlight_vertices=[i],
    fig=plt.gcf(),
    axes=plt.gca()
)
plt.scatter(
    tri.circumcenters[:, 0],
    tri.circumcenters[:, 1],
    c=pde_on_omega_values,
    s=100
)
plt.show()
tri.triangles[i]
tri.pde_values[
    tri.triangles[i]
]
np.mean(
    tri.pde_values[
        tri.triangles[i]
    ]
)
pde_on_omega_values[i]

# Test barycentric interpolation on ith triangle with pde values
tri.triangle_coordinates[i]
r_1 = tri.triangle_coordinates[i][0]
r_2 = tri.triangle_coordinates[i][1]
r_3 = tri.triangle_coordinates[i][2]
n = 10
barycentric_coor_grid = np.vstack([np.array([i, j]) / (n - 1) for i in range(n) for j in range(n - i)])
cartesian_coor_tri_grid = np.vstack(
    [
        barycentric_to_cartesian(lambda_1, lambda_2, r_1[0], r_1[1], r_2[0], r_2[1], r_3[0], r_3[1])
        for lambda_1, lambda_2 in barycentric_coor_grid
    ]
)
z_values = np.array([
    barycentric_interpolation(
        x, y,
        r_1[0], r_1[1],
        r_2[0], r_2[1],
        r_3[0], r_3[1],
        tri.pde_values[tri.triangles[i][0]],
        tri.pde_values[tri.triangles[i][1]],
        tri.pde_values[tri.triangles[i][2]]
    )
    for x, y in cartesian_coor_tri_grid
])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    np.concatenate([cartesian_coor_tri_grid[:, 0], np.array([tri.circumcenters[i][0]])]),
    np.concatenate([cartesian_coor_tri_grid[:, 1], np.array([tri.circumcenters[i][1]])]),
    np.concatenate([z_values, np.array([pde_on_omega_values[i]])]),
)
plt.show()
# END TEST OF BARYCENTRIC INTERPOLATION


period_gsb = compute_period()
uniformization = np.exp(2 * np.pi / period_gsb * (pde_on_omega_values + 1j * g_star_bar))


plt.clf()
plt.cla()
plt.scatter(
    np.real(uniformization),
    np.imag(uniformization),
    s=50
)
plt.title('Conformal Model')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.gca().set_aspect('equal')
plt.savefig(path.with_suffix('.png'))
# plt.show()

# flux_color_array = np.zeros(tri.num_triangles, dtype=np.float64)
# for i in range(num_contained_polygons):
#     index = tri.contained_to_original_index[i]
#     flux_color_array[index] = g_star_bar[i]

# tri.show_voronoi_tesselation(
#     'test.png',
#     show_vertex_indices=True
# )
# plt.scatter(
#     tri.circumcenters[:, 0],
#     tri.circumcenters[:, 1],
#     c=g_star_bar,
#     s=500
# )
# plt.colorbar()
# plt.show()



# Level curves for gsb
g_star_bar_interpolated_interior = np.array([np.mean(g_star_bar[poly]) for poly in tri.contained_polygons])
min_, max_ = np.min(g_star_bar_interpolated_interior), np.max(g_star_bar_interpolated_interior)
heights = np.linspace(min_, max_, num=100)

# tri.show_voronoi_tesselation(
#     'test.png',
#     show_vertex_indices=True
# )
# color_array = np.array((g_star_bar_interpolated_interior - min_) / (max_ - min_))
# plt.scatter(
#     tri.vertices[tri.contained_to_original_index][:, 0],
#     tri.vertices[tri.contained_to_original_index][:, 1],
#     c=color_array,
#     s=500
# )
# plt.show()


def flatten_list_of_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


contained_triangle_indicator = np.all(tri.vertex_boundary_markers[tri.triangles] == 0, axis=1)
contained_triangles = np.where(contained_triangle_indicator)[0]
slit_cell_vertices = set(flatten_list_of_lists([tri.contained_polygons[cell] for cell in cell_path]))

fig, axes = plt.subplots()
tri.show(
    show_edges=True,
    show_triangles=False,
    fig=fig,
    axes=axes
)
tri.show_voronoi_tesselation(
    show_vertex_indices=True,
    show_polygons=False,
    highlight_polygons=cell_path,
    highlight_vertices=list(slit_cell_vertices),
    fig=fig,
    axes=axes
)
plt.show()

contained_triangle_minus_slit = list(set(contained_triangles).difference(slit_cell_vertices))

tri.show(
    show_triangle_indices=True,
    highlight_triangles=contained_triangle_minus_slit
)
plt.show()

level_set = []
for i in range(len(contained_triangle_minus_slit)):
    triangle = tri.triangles[contained_triangle_minus_slit[i]]
    level_set_triangle = tri_level_sets(
        tri.vertices[triangle],
        g_star_bar_interpolated_interior[tri.original_to_contained_index[triangle]],
        heights
    )
    level_set.append(level_set_triangle)

level_set_flattened = flatten_list_of_lists(level_set)
level_set_filtered = [
    line_segment for line_segment in level_set_flattened if len(line_segment) > 0
]
lines = [
    [
        tuple(line_segment[0]),
        tuple(line_segment[1])
    ] for line_segment in level_set_filtered
]
line_collection = mc.LineCollection(lines, linewidths=1)
line_collection.set(color=[1, 0, 0])
tri.show(
    highlight_triangles=contained_triangle_minus_slit
)
axes = plt.gca()
axes.add_collection(line_collection)
plt.show()


tri.show(
    'test.png',
    show_edges=True,
    show_triangles=False
)
tri.show_voronoi_tesselation(
    'test.png',
    show_edges=True,
    show_polygons=False,
    fig=plt.gcf(),
    axes=plt.gca()
)
plt.show()

fig, axes = plt.subplots()
tri.show(
    show_level_curves=True,
    fig=fig,
    axes=axes
)
axes.add_collection(line_collection)
plt.show()


# Conjugate level curves with color
fig, axes = plt.subplots()
def subsample_color_map(colormap, num_samples, start_color=0, end_color=255, reverse=False):
    sample_points_float = np.linspace(start_color, end_color, num_samples)
    sample_points = np.floor(sample_points_float).astype(np.int64)
    all_colors = colormap.colors
    if reverse:
        all_colors = np.flip(all_colors, axis=0)
    return all_colors[sample_points]

# graded_level_curve_color_map = cm.lajolla
conjugate_level_curve_color_map = cm.buda
conjugate_level_curve_colors = subsample_color_map(
    conjugate_level_curve_color_map,
    len(heights),
    start_color=32,
    end_color=223,
    reverse=True
)
for height_index, height in enumerate(heights):
    level_set = []
    for i in range(len(contained_triangle_minus_slit)):
        triangle = tri.triangles[contained_triangle_minus_slit[i]]
        level_set_triangle = tri_level_sets(
            tri.vertices[triangle],
            g_star_bar_interpolated_interior[tri.original_to_contained_index[triangle]],
            [height]
        )
        level_set.append(level_set_triangle)

    level_set_flattened = flatten_list_of_lists(level_set)
    level_set_filtered = [
        line_segment for line_segment in level_set_flattened if len(line_segment) > 0
    ]
    lines = [
        [
            tuple(line_segment[0]),
            tuple(line_segment[1])
        ] for line_segment in level_set_filtered
    ]
    line_collection = mc.LineCollection(lines, linewidths=1)
    line_collection.set(color=conjugate_level_curve_colors[height_index])
    axes.add_collection(line_collection)
tri.show(
    show_level_curves=True,
    fig=fig,
    axes=axes
)
axes.add_collection(line_collection)
plt.show()
