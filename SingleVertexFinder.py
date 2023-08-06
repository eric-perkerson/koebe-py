#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 17:00:55 2023

@author: joshua
"""

from triangulation import Triangulation
import numpy as np
from pathlib import Path
import time

startTime = time.time()


file_stem = 'test_example_1'
path = Path(f'regions/{file_stem}/{file_stem}')
tri = Triangulation.read(path)

signChange = []
for vertex in range(len(tri.vertices)):
    #print(tri.vertex_topology[vertex])
    neighborChange = []
    if (tri.vertex_topology[vertex][0] != -1):
        currentVertexValue = tri.pde_values[vertex]
        for neighbor in tri.vertex_topology[vertex]:
            neighborChange.append(np.sign(tri.pde_values[neighbor] - currentVertexValue))
        count = 0
        for i in range(len(neighborChange) - 1):
            #print("The two are", neighborChange[i], "and", neighborChange[i + 1])
            if (neighborChange[i] != neighborChange[i+1]):
                count = count + 1
        if (neighborChange[0] != neighborChange[-1]):
            count = count + 1
        signChange.append(count)
    else:
        signChange.append(2)
print(signChange)
number = 0
for i in signChange:
    if (i != 2):
        print("Exists: ", number)
    number = number + 1
print("Program took ", time.time() - startTime, " to run")
