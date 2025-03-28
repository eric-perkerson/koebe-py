import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc

directory = "/Users/joshua/Desktop/programs/TrashStuff/SaarResearch/planar-domains"

colors_filtered = np.load(directory + "/colors.npy")
level_set_filtered = np.load(directory + "/levelSetFiltered.npy")
# x = np.linspace(0, 1, num = len(colors))
# plt.scatter(x, np.ones(len(colors)), c = colors)
# plt.show()
fig, axes = plt.subplots()
line_collection_collection = []
for i, height in enumerate(level_set_filtered):

    lines = [
        [
            tuple(line_segment[0]),
            tuple(line_segment[1])
        ] for line_segment in level_set_filtered
    ]
    line_collection = mc.LineCollection(lines, linewidths=1)
    line_collection.set(color = colors_filtered[i])

    line_collection_collection.append(line_collection)
    axes.add_collection(line_collection)
    #self.axes.add_collection(line_collection)

for line_collection in line_collection_collection:
    axes.add_collection(line_collection)

plt.show()