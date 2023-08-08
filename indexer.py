#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:05:49 2023

@author: joshua
"""

from triangulation import Triangulation
from pathlib import Path

file_stem = 'test_example_0'
path = Path(f'regions/{file_stem}/{file_stem}')
tri = Triangulation.read(path)

name = Path(f'index/{file_stem}.jos')
fd = open(name, "w")


fd.write(str(len(tri.vertices)))
fd.write("\n")
for vertex in range(len(tri.vertices)):
    for neighbor in tri.vertex_topology[vertex]:
        fd.write(str(neighbor))
        fd.write(" ")
    fd.write("\n")
fd.close()