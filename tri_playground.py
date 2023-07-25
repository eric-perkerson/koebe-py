from triangulation import Triangulation
from pathlib import Path

path = Path('/Users/eric/Code/planar-domains/regions/test_example_0/test_example_0')
tri = Triangulation.read(path)

tri.pde_values
