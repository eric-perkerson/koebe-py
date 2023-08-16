from triangulation import Triangulation
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm

file_stem = '3_fold_sym'
path = Path(f'regions/{file_stem}/{file_stem}')
tri = Triangulation.read(path)

tri.show(
    'test.png',
    show_level_curves=True,
    show_singular_level_curves=True,
    dpi=500,
    num_level_curves=50,
    line_width=0.75
)
