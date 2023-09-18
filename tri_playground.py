from triangulation import Triangulation
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm
import subprocess
from region import Region
import pyvista


file_stem = "non_concentric_annulus"

#file_stem = 'No_3_fold_sym'
#file_stem = '3_fold_sym'
# file_stem = '3_fold_sym'

# path = Path(f'regions/{file_stem}/{file_stem}')
# tri = Triangulation.read(path)

from region import Region
domain = Region.region_from_components(
    [
        [
            (2.0, 0.0),
            (1.7320508075688774, 0.9999999999999999),
            (1.0000000000000002, 1.7320508075688772),
            (1.2246467991473532e-16, 2.0),
            (-0.9999999999999996, 1.7320508075688776),
            (-1.7320508075688774, 0.9999999999999999),
            (-2.0, 2.4492935982947064e-16),
            (-1.7320508075688776, -0.9999999999999996),
            (-1.0000000000000009, -1.7320508075688767),
            (-3.6739403974420594e-16, -2.0),
            (1.0, -1.7320508075688772),
            (1.7320508075688767, -1.0000000000000009)],
        [
            (-0.5, 1.2246467991473532e-16),
            (-0.3660254037844387, 0.49999999999999994),
            (-1.1102230246251565e-16, 0.8660254037844386),
            (0.5000000000000001, 1.0),
            (1.0, 0.8660254037844387),
            (1.3660254037844384, 0.5000000000000002),
            (1.5, 0.0),
            (1.3660254037844388, -0.49999999999999983),
            (1.0000000000000004, -0.8660254037844385),
            (0.5000000000000001, -1.0),
            (6.106226635438361e-16, -0.866025403784439),
            (-0.3660254037844385, -0.5000000000000003)]
    ]
)

with open(f"regions/{file_stem}/{file_stem}.poly", 'w', encoding='utf-8') as f:
    domain.write(f)

subprocess.run([
    #    'julia',
        'julia',
        'triangulate_via_julia.jl',
        file_stem,
        file_stem,
        "500"
    ])

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
print(tri.singular_heights)
print(tri.singular_vertices)

plt.scatter(
    tri.vertices[:, 0],
    tri.vertices[:, 1],
    c=tri.pde_values
)
plt.show()

#change how_singular_level_curves=True to False in case of an annulus
tri.show(
    'test.png',
    show_level_curves=True,
    show_singular_level_curves=True,
    dpi=500,
    num_level_curves=500,
    line_width=0.75
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
