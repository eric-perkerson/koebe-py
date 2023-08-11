#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 17:00:55 2023

@author: joshua
"""

from triangulation import Triangulation
import numpy as np
from pathlib import Path
from numba import njit
from numba.typed import List

@njit
def index(vTopo, pdeVal, length):
    signChange = np.empty(length, np.int16) # sets up an array to hold index values
    y = 0
    for vertex in range(length): # loops through every vertex
        neighborChange = np.zeros(len(vTopo[vertex]), np.int16) # sets up an array to hold the sign of the difference between neighbors
        if (vTopo[vertex][0] != -1): # weeds out boundary points
            x = 0
            for neighbor in vTopo[vertex]: # Loops through each neighbor, calculating the difference between it and the home vertex
                neighborChange[x] = (np.sign(pdeVal[neighbor] - pdeVal[vertex]))
                x += 1
                # Applys the sign function for ease, appending it to the neighborChange array
            count = 0 # a counter to keep track of sign changes
            for i in range(len(neighborChange) - 1): # loops through each difference in the neighbor array
                if (neighborChange[i] != neighborChange[i+1]): # If two adjacent differences dont have the same sign
                    count = count + 1 # increment the index
            if (neighborChange[0] != neighborChange[-1]): # mops up the loop around
                count = count + 1
            signChange[y] = count# add the index to the signChange array
            y += 1 
        else:
            signChange[y] = 2 # if its a boundary point set it to be 2 to ignore it
            y += 1
    return signChange


file_stem = 'test_example_1'
path = Path(f'regions/{file_stem}/{file_stem}')
triangle = Triangulation.read(path)
topology = List()

for neighbor in range(len(triangle.vertex_topology)):
    topology.append(triangle.vertex_topology[neighbor])
    
#topology = triangle.vertex_topology
pde = triangle.pde_values
signChange = index(topology, pde, len(triangle.vertices))
number = 0 # Just a tracker to see which vertex is found
for i in signChange:
    if (i != 2):
        print("Exists: ", number)
    number = number + 1
