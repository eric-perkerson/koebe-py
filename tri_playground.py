#

from triangulation import Triangulation
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

path = Path('regions/test_example_0/test_example_0')
tri = Triangulation.read(path)
tri.show('test.png', show_vertex_indices=True)

for i, neighbors in enumerate(tri.vertex_topology):
    print(i, neighbors)

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
