from region import Region
from triangulation import (
    Triangulation,
    point_to_right_of_line_compiled,
    polygon_oriented_counterclockwise
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



NUM_TRIANGLES = 2000
USE_WOLFRAM_SOLVER = True

file_stem = "genus_2"
path = Path(f'regions/{file_stem}/{file_stem}')

#file_stem = 'No_3_fold_sym'
#file_stem = '3_fold_sym'
#file_stem = '3_fold_sym'


# tri = Triangulation.read(path)


# domain = Region.region_from_components(
#     [
#         [
#             (2.0, 0.0),
#             (1.7320508075688774, 0.9999999999999999),
#             (1.0000000000000002, 1.7320508075688772),
#             (0.0, 2.0),
#             (-0.9999999999999996, 1.7320508075688776),
#             (-1.7320508075688774, 0.9999999999999999),
#             (-2.0, 0.0),
#             (-1.7320508075688776, -0.9999999999999996),
#             (-1.0000000000000009, -1.7320508075688767),
#             (0.0, -2.0),
#             (1.0, -1.7320508075688772),
#             (1.7320508075688767, -1.0000000000000009)
#         ],
#         [
#             (-0.1, 0.0),
#             (-0.08660254037844388, 0.049999999999999996),
#             (-0.05000000000000002, 0.08660254037844387),
#             (0.0, 0.1),
#             (0.049999999999999996, 0.08660254037844388),
#             (0.08660254037844385, 0.050000000000000024),
#             (0.1, 0.0),
#             (0.08660254037844388, -0.04999999999999999),
#             (0.05000000000000004, -0.08660254037844385),
#             (0.0, -0.1),
#             (-0.04999999999999994, -0.0866025403784439),
#             (-0.08660254037844385, -0.05000000000000004)
#         ],
#         [
#             (0.40000000000000013, 0.8660254037844386),
#             (0.4133974596215562, 0.9160254037844386),
#             (0.45000000000000007, 0.9526279441628824),
#             (0.5000000000000001, 0.9660254037844386),
#             (0.5500000000000002, 0.9526279441628824),
#             (0.586602540378444, 0.9160254037844386),
#             (0.6000000000000001, 0.8660254037844386),
#             (0.586602540378444, 0.8160254037844386),
#             (0.5500000000000002, 0.7794228634059948),
#             (0.5000000000000001, 0.7660254037844386),
#             (0.4500000000000002, 0.7794228634059946),
#             (0.4133974596215563, 0.8160254037844386)
#         ],
#         [
#             (-0.6000000000000004, -0.8660254037844384),
#             (-0.5866025403784443, -0.8160254037844383),
#             (-0.5500000000000005, -0.7794228634059945),
#             (-0.5000000000000004, -0.7660254037844384),
#             (-0.45000000000000046, -0.7794228634059945),
#             (-0.4133974596215566, -0.8160254037844383),
#             (-0.40000000000000047, -0.8660254037844384),
#             (-0.41339745962155655, -0.9160254037844384),
#             (-0.4500000000000004, -0.9526279441628822),
#             (-0.5000000000000004, -0.9660254037844384),
#             (-0.5500000000000004, -0.9526279441628823),
#             (-0.5866025403784443, -0.9160254037844384)
#         ]
#     ]
# )

# with open(f"regions/{file_stem}/{file_stem}.poly", 'w', encoding='utf-8') as f:
#     domain.write(f)

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

    subprocess.run([
        'python',
        'mesh_conversion/fenicsx_solver.py',
        file_stem,
    ])

tri = Triangulation.read(f'regions/{file_stem}/{file_stem}.poly')
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
    show_singular_level_curves=False,
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






# from region import Region
# domain = Region.region_from_components(
#     [
#         [
#             (2.0, 0.0),
#             (1.0000000000000002, 1.7320508075688772),
#             (-0.9999999999999996, 1.7320508075688776),
#             (-2.0, 2.4492935982947064e-16),
#             (-1.0000000000000009, -1.7320508075688767),
#             (1.0, -1.7320508075688772)
#         ],
#         [
#             (0.9000000000000001, 2.4492935982947065e-17),
#             (1.0, 0.17320508075688773),
#             (1.2000000000000002, 0.17320508075688776),
#             (1.3, 0.0),
#             (1.2000000000000002, -0.1732050807568877),
#             (1.0000000000000002, -0.1732050807568878)
#         ],
#         [
#             (-0.7499999999999998, 0.9526279441628828),
#             (-0.6499999999999999, 1.1258330249197706),
#             (-0.44999999999999984, 1.1258330249197706),
#             (-0.3499999999999998, 0.9526279441628828),
#             (-0.44999999999999973, 0.7794228634059951),
#             (-0.6499999999999997, 0.779422863405995)
#         ],
#         [
#             (-0.7500000000000004, -0.9526279441628823),
#             (-0.6500000000000006, -0.7794228634059945),
#             (-0.4500000000000005, -0.7794228634059945),
#             (-0.3500000000000005, -0.9526279441628823),
#             (-0.4500000000000004, -1.12583302491977),
#             (-0.6500000000000004, -1.12583302491977)
#         ]
#     ]
# )


# Annulus
from region import Region
from triangulation import (
    Triangulation,
    point_to_right_of_line_compiled,
    polygon_oriented_counterclockwise,
    segment_intersects_segment
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

NUM_TRIANGLES = 200
USE_WOLFRAM_SOLVER = True

file_stem = "concentric_annulus"
path = Path(f'regions/{file_stem}/{file_stem}')
tri = Triangulation.read(f'regions/{file_stem}/{file_stem}.poly')

# Choose a base point on the inner boundary
base_point = np.where(tri.vertex_boundary_markers == 2)[0][0]


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


def get_base_cell(tri, omega_0):
    for i, cell in enumerate(tri.contained_polygons):
        if omega_0 in cell:
            return i
    return -1


hole_x, hole_y = tri.region.points_in_holes[0]
# omega_0_x, omega_0_y = 1.0, 0.0
omega_0 = 338
omega_0_x, omega_0_y = tri.circumcenters[omega_0]
base_cell = get_base_cell(tri, omega_0)

# tri.show_voronoi_tesselation(
#     'voronoi.png',
#     show_vertex_indices=True,
#     # highlight_vertices=[omega_0],
#     highlight_polygons=[base_cell]
# )
# plt.scatter(
#     [hole_x],
#     [hole_y],
#     c=[[1, 0, 0]]
# )
# plt.show()


# base_point = tri.vertices[base_cell]
# tri.show_voronoi_tesselation(
#     'voronoi.png',
#     show_vertex_indices=True,
#     # highlight_vertices=[omega_0],
#     highlight_polygons=[base_cell]
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
# plt.show()

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
    [tri.original_to_contained_index[vertex] if vertex in tri.original_to_contained_index else -1 for vertex in cell] for cell in tri.vertex_topology
]
contained_topology = [contained_topology_all[i] for i in tri.contained_to_original_index]

# TODO
# Create cell path from base_cell to boundary_a
base_vertex = tri.contained_to_original_index[base_cell]
contained_topology[base_cell]
base_cell_vertices = tri.contained_polygons[base_cell]
edges = build_polygon_edges(base_cell_vertices)
for i, edge in enumerate(edges):
    if segment_intersects_line_positive(
            tri.circumcenters[edge[0]],
            tri.circumcenters[edge[1]]
    ):
        print(i)
        break

base_cell
contained_topology[base_cell]


# tri.show_voronoi_tesselation(
#     'voronoi.png',
#     show_polygon_indices=True,
#     show_edges=True
# )
# plt.show()

# Show setup with line from point_in_hole to base_point
# tri.show_voronoi_tesselation(
#     'voronoi.png',
#     show_vertex_indices=False,
#     show_polygon_indices=True,
#     show_edges=True,
#     # highlight_polygons=[base_cell]
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
# lines = [
#     [
#         tuple(base_point),
#         tuple([hole_x, hole_y])
#     ]
# ]
# line_collection = mc.LineCollection(lines, linewidths=2)
# axes = plt.gca()
# axes.add_collection(line_collection)
# plt.show()


# Check polygon orientation
np.all([polygon_oriented_counterclockwise(poly, tri.circumcenters) for poly in tri.contained_polygons])
np.all([not polygon_oriented_counterclockwise(np.flip(poly), tri.circumcenters) for poly in tri.contained_polygons])

# # Show the changed base points for vertex_topology to align with the first edge in each polygon
# for counter, contained_poly_index in enumerate(range(len(tri.contained_polygons) - 5, len(tri.contained_polygons))):
#     poly = tri.contained_polygons[contained_poly_index]
#     edges = build_polygon_edges(poly)
#     triangle_index = tri.contained_to_original_index[contained_poly_index]
#     first_poly_edge = np.array([poly[0], poly[1]])

#     tri.show_voronoi_tesselation(
#         'voronoi.png',
#         show_vertex_indices=False,
#         show_polygon_indices=True,
#         show_edges=True,
#     )
#     poly_edge_lines = [
#         [
#             tuple(tri.circumcenters[first_poly_edge[0]]),
#             tuple(tri.circumcenters[first_poly_edge[1]])
#         ]
#     ]
#     tri_edge_lines = [
#         [
#             tuple(tri.vertices[triangle_index]),
#             tuple(tri.vertices[tri.vertex_topology[triangle_index][0]])
#         ]
#     ]
#     poly_edge_line_collection = mc.LineCollection(poly_edge_lines, linewidths=2)
#     tri_edge_line_collection = mc.LineCollection(tri_edge_lines, linewidths=2)
#     poly_edge_line_collection.set(color=[1, 0, 0])
#     tri_edge_line_collection.set(color=[0, 1, 0])
#     axes = plt.gca()
#     axes.add_collection(poly_edge_line_collection)
#     axes.add_collection(tri_edge_line_collection)
#     plt.show()
