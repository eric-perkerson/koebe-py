#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:05:49 2023

@author: joshua
"""

from triangulation import Triangulation
from pathlib import Path
import sys

file_stem = sys.argv[1]
path = Path(f'../regions/{file_stem}/{file_stem}')
tri = Triangulation.read(path)

name = Path(f'{file_stem}.jos')
fd = open(name, "w")


fd.write(str(len(tri.vertices)))
fd.write("\n")
for vertex in range(len(tri.vertices)):
    for neighbor in tri.vertex_topology[vertex]:
        fd.write(str(neighbor))
        fd.write(" ")
    fd.write("\n")
fd.close()