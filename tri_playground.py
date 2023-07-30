#

from triangulation import Triangulation
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

file_stem = 'test_example_0'
path = Path(f'regions/{file_stem}/{file_stem}')
tri = Triangulation.read(path)
tri.show('test.png', show_vertex_indices=False)

for neighbors in tri.vertex_topology:
    print(neighbors)

for i, neighbors in enumerate(tri.vertex_topology):
    print(i, neighbors)

for i, value in enumerate(tri.pde_values):
    print(i, value)

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
