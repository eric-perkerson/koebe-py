from triangulation import Triangulation
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import subprocess

from region import Region

domain = Region.region_from_components(
    [
        [
            (2.0, 0.0),
            (1.0000000000000002, 1.7320508075688772),
            (-0.9999999999999996, 1.7320508075688776),
            (-2.0, 2.4492935982947064e-16),
            (-1.0000000000000009, -1.7320508075688767),
            (1.0, -1.7320508075688772)
        ],
        [
            (0.9000000000000001, 2.4492935982947065e-17),
            (1.0, 0.17320508075688773),
            (1.2000000000000002, 0.17320508075688776),
            (1.3, 0.0),
            (1.2000000000000002, -0.1732050807568877),
            (1.0000000000000002, -0.1732050807568878)
        ],
        [
            (-0.7499999999999998, 0.9526279441628828),
            (-0.6499999999999999, 1.1258330249197706),
            (-0.44999999999999984, 1.1258330249197706),
            (-0.3499999999999998, 0.9526279441628828),
            (-0.44999999999999973, 0.7794228634059951),
            (-0.6499999999999997, 0.779422863405995)
        ],
        [
            (-0.7500000000000004, -0.9526279441628823),
            (-0.6500000000000006, -0.7794228634059945),
            (-0.4500000000000005, -0.7794228634059945),
            (-0.3500000000000005, -0.9526279441628823),
            (-0.4500000000000004, -1.12583302491977),
            (-0.6500000000000004, -1.12583302491977)
        ]
    ]
)
with open("regions/3_fold_sym/3_fold_sym.poly", 'w') as file:
    domain.write(file)

file_stem = '3_fold_sym'
path = Path(f'regions/{file_stem}/{file_stem}')
tri = Triangulation.read(path)


subprocess.run([
    'python',
    'mesh_conversion/mesh_conversion.py',
    '-p',
    f'regions/3_fold_sym/3_fold_sym.output.poly',
    '-n',
    f'regions/3_fold_sym/3_fold_sym.node',
    '-e',
    f'regions/3_fold_sym/3_fold_sym.ele',
])

# Plot the PDE solution on the vertices (from Mathematica)
pde_min = np.min(tri.pde_values)
pde_max = np.max(tri.pde_values)
pde_colors = (tri.pde_values - pde_min) / (pde_max - pde_min)

fig = tri.region.figure('test')
plt.scatter(
    tri.vertices[:, 0],
    tri.vertices[:, 1],
    c=pde_colors
)
plt.show()

# TODO: Figure out how to compute the singular vertex using the Triangulation object


# Display the region
tri.show('test.png', show_vertex_indices=False)
plt.show()

# present the region
# im = img.imread('test.png')
# plt.imshow(im)
# plt.show()


# Enumeration of neighbors of vertices; values for index computation
for neighbors in tri.vertex_topology:
    print(neighbors)

for i, neighbors in enumerate(tri.vertex_topology):
    print(i, neighbors)

for i, value in enumerate(tri.pde_values):
    print(i, value)

# Plot the PDE solution on the vertices (from Mathematica as a PDE solver)

pde_min = np.min(tri.pde_values)
pde_max = np.max(tri.pde_values)
pde_colors = (tri.pde_values - pde_min) / (pde_max - pde_min)

#fig = tri.region.figure('test')

plt.figure()
plt.scatter(
    tri.vertices[:, 0],
    tri.vertices[:, 1],
    c=pde_colors
)
plt.show()

# TODO: Figure out how to compute the singular vertex using the Triangulation object
