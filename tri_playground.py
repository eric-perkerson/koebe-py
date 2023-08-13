from triangulation import Triangulation
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

file_stem = '3_fold_sym'
path = Path(f'regions/{file_stem}/{file_stem}')
tri = Triangulation.read(path)

tri.show('test.png', show_level_curves=True)
