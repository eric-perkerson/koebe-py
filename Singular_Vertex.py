import math
from triangulation import Triangulation
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class sig():
    def sig_find(path): # takes as input a path such as Path('regions/test_example_4/test_example_4')
        tri = Triangulation.read(path)
        crit = []
        for i, neighbors in enumerate(tri.vertex_topology): # iterate through the vertex topology
            base_value = tri.pde_values[i] # the value of the candidate vertex
            sign_changes = 0
            sign_values = []
            if not(neighbors[0] == -1): # test if this is a boundary point
                for n in neighbors: # go through a list of neighbors and determine the sign changes
                    diff = math.copysign(1,base_value-tri.pde_values[n]) # 1 if niegbor value is larger , -1 if smaller
                    sign_values.append(diff) # creates the list to track which niegbors are bigger or smaller
                for j in range(0,len(sign_values)-1): # iterates throught the list to check if there is a sign change
                    if not((sign_values[j] == sign_values[j+1])):
                        sign_changes += 1
                if not((sign_values[len(sign_values)-1] == sign_values[0])): # checks if there is a sign change looping back from the end to the start
                    sign_changes += 1
            if sign_changes > 2: # reports if there are more than 2 sign changes
                print("Vertex " + str(i) + " is a critical vertex.")
                nei = []
                for n in neighbors:
                    nei.append(n)
                crit.append([i,nei])

        return(crit) # returns a list of sigular vertices and a list of their neighbors such as 
        #Example return would be [[3653, [3652, 3662, 3663, 3654, 4818, 4819]], [4354, [2216, 2221, 4349, 4348, 4353, 4358]]]


