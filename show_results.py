from region import Region
from triangulation import (
    Triangulation,
    point_to_right_of_line_compiled,
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numba
import networkx as nx
import tkinter as tk
from sys import argv
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.patches import Annulus, Circle, Polygon
from matplotlib import animation
import draw_region
import subprocess
import random
import os
from pathlib import Path
import math

BG_COLOR = '#2e2e2e'
WHITE = '#d6d6d6'
MAGENTA = '#e519cf'
BLUE = '#6c99bb'
GREEN = '#b4d273'
ORANGE = '#e87d3e'
PURPLE = '#9e86c8'
PINK = '#b05279'
YELLOW = '#e5b567'
GREY = '#797979'
BLACK = '#000000'
RAND_1 = '#87992d'
RAND_4 = '#8119f2'
RAND_5 = '#4ce7dc'
RAND_6 = '#e95501'
RAND_7 = '#2c38a7'
RAND_8 = '#8ece91'
RAND_9 = '#727986'
RAND_10 = '#68e77e'
RAND_11 = '#5d85dd'
RAND_12 = '#423da8'
RAND_13 = '#7cccfe'
RAND_14 = '#488f2e'
RAND_15 = '#54179f'

BDRY_COLORS = [
    MAGENTA,
    BLUE,
    GREEN,
    ORANGE,
    PURPLE,
    PINK,
    YELLOW,
    GREY,
    RAND_1,
    RAND_4,
    RAND_5,
    RAND_6,
    RAND_7,
    RAND_8,
    RAND_9,
    RAND_10,
    RAND_11,
    RAND_12,
    RAND_13,
    RAND_14,
    RAND_15
]

NUM_TRIANGLES = 2000

class GraphConfig(tk.Frame):

    def __init__(self, width, height):
        self.canvas_height = height
        self.canvas_width = width

        self.show_vertices_tri = tk.BooleanVar()
        self.show_edges_tri=tk.BooleanVar()
        self.show_edges_tri.set(False)
        self.show_triangles_tri=tk.BooleanVar()
        self.show_triangles_tri.set(False)
        self.show_vertex_indices_tri=tk.BooleanVar()
        self.show_triangle_indices_tri=tk.BooleanVar()
        self.show_level_curves_tri=tk.BooleanVar()
        self.show_singular_level_curves_tri=tk.BooleanVar()

        self.show_vertex_indices_vor=tk.BooleanVar()
        self.show_vertex_indices_vor.set(False)
        self.show_polygon_indices_vor=tk.BooleanVar()
        self.show_polygon_indices_vor.set(False)
        self.show_vertices_vor=tk.BooleanVar()
        self.show_edges_vor=tk.BooleanVar()
        self.show_edges_vor.set(True)
        self.show_polygons_vor=tk.BooleanVar()
        self.show_polygons_vor.set(True)
        self.show_region_vor=tk.BooleanVar()
        self.show_region_vor.set(True)

        self.showSlitBool = tk.BooleanVar()
        self.showSlitBool.set(False)

    def getConfigsVor(self):
        """ vertex indices, polygon indices, vertex, edge, polygon, region"""
        return self.show_vertex_indices_vor.get(), self.show_polygon_indices_vor.get(), self.show_vertices_vor.get(), self.show_edges_vor.get(), self.show_polygons_vor.get(), self.show_region_vor.get()

    def getConfigsTri(self):
        """ vertex, edges, triangles, vertex indices, triangle indices, level curves, singular level curves"""
        return self.show_vertices_tri.get(), self.show_edges_tri.get(), self.show_triangles_tri.get(), self.show_vertex_indices_tri.get(), self.show_triangle_indices_tri.get(), self.show_level_curves_tri.get(), self.show_singular_level_curves_tri.get()
    
    def getSlit(self):
        return self.showSlitBool.get()
    
    def setSlit(self, bool):
        self.showSlitBool.set(bool)
    
    def getFrame(self, parent):
        super().__init__(parent, width = self.canvas_width, height = self.canvas_height, bg=BG_COLOR)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.grid(column=0, row=0)
        checkButtonTri1 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_height/50), text="Show Vertices Tri", variable=self.show_vertices_tri, bg=BG_COLOR)
        checkButtonTri1.grid(column=0, row=0)
        checkButtonTri2 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_height/50), text="Show Edges Tri", variable=self.show_edges_tri, bg=BG_COLOR)
        checkButtonTri2.grid(column=1, row=0)
        checkButtonTri3 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Triangles Tri", variable=self.show_triangles_tri, bg=BG_COLOR)
        checkButtonTri3.grid(column=2, row=0)
        checkButtonTri4 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Vertex Indices Tri", variable=self.show_vertex_indices_tri, bg=BG_COLOR)
        checkButtonTri4.grid(column=3, row=0)
        checkButtonTri5 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Triangle Indices Tri", variable=self.show_triangle_indices_tri, bg=BG_COLOR)
        checkButtonTri5.grid(column=4, row=0)
        checkButtonTri6 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Level Curves Tri", variable=self.show_level_curves_tri, bg=BG_COLOR)
        checkButtonTri6.grid(column=5, row=0)
        checkButtonTri7 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Singular Level Curves Tri", variable=self.show_singular_level_curves_tri, bg=BG_COLOR)
        checkButtonTri7.grid(column=6, row=0)
        checkButtonVor1 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Vertex Indices Vor", variable=self.show_vertex_indices_vor, bg=BG_COLOR)
        checkButtonVor1.grid(column=0, row=1)
        checkButtonVor2 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Polygon Indices Vor", variable=self.show_polygon_indices_vor, bg=BG_COLOR)
        checkButtonVor2.grid(column=1, row=1)
        checkButtonVor3 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Vertices Vor", variable=self.show_vertices_vor, bg=BG_COLOR)
        checkButtonVor3.grid(column=2, row=1)
        checkButtonVor4 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Edges Vor", variable=self.show_edges_vor, bg=BG_COLOR)
        checkButtonVor4.grid(column=3, row=1)
        checkButtonVor5 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Polygons Vor", variable=self.show_polygons_vor, bg=BG_COLOR)
        checkButtonVor5.grid(column=4, row=1)
        checkButtonVor6 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Region Vor", variable=self.show_region_vor, bg=BG_COLOR)
        checkButtonVor6.grid(column=5, row=1)
        # drawButton = tk.Button(self.controls, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Display Graph", command = self.show)
        # drawButton.grid(column=6, row=1)
        slitButton = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Slit", variable=self.showSlitBool, bg=BG_COLOR)
        slitButton.grid(column=6, row=1)
        # backButton = tk.Button(self.controls, height=int(self.canvas_height/540), width=int(self.canvas_height/50), text="Back", command = self.mainMenu)
        # backButton.grid(column=1, row=2)
        return self

class DrawRegion(tk.Frame):
    def __init__(self, parent, width, height):
        super().__init__(parent, width = width, height = height, bg=BG_COLOR)
        self.canvas_width = width
        self.canvas_height = height
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.freeDraw = tk.BooleanVar()
        self.freeDraw.set(False)
        self.inEdgeNum = tk.StringVar()
        self.outEdgeNum = tk.StringVar()
        self.outRad = tk.StringVar()
        self.inRad = tk.StringVar()
        self.fileRoot = tk.StringVar()
        self.fileName = tk.StringVar()
        self.triCount = tk.StringVar()
        self.randomSet = tk.BooleanVar()
        self.inOrOut = tk.BooleanVar()

        instructLabel = tk.Label(self, height=int(self.canvas_height/540), width=int(self.canvas_height/10), text="Select option, then click calcultate to generate a new figure", bg=BG_COLOR)
        instructLabel.grid(column=2, row=0, columnspan=3)

        radiusOneLabel = tk.Label(self, width=int(self.canvas_height/50), height=int(self.canvas_height/600), text="Outer Radius", bg=BG_COLOR)
        radiusOneLabel.grid(column=0, row=1)

        radiusOneEntry = tk.Entry(self, width=int(self.canvas_height/50), textvariable=self.outRad, bg=BLACK)
        radiusOneEntry.grid(column=1, row=1)

        radiusTwoLabel = tk.Label(self, width=int(self.canvas_height/50), height=int(self.canvas_height/600), text="Inner Radius", bg=BG_COLOR)
        radiusTwoLabel.grid(column=2, row=1)

        radiusTwoEntry = tk.Entry(self, width=int(self.canvas_height/50), textvariable=self.inRad, bg=BLACK)
        radiusTwoEntry.grid(column=3, row=1)

        fileRootLabel = tk.Label(self, width=int(self.canvas_height/50), height=int(self.canvas_height/600), text="File Root", bg=BG_COLOR)
        fileRootLabel.grid(column=4, row=1)

        fileRootEntry = tk.Entry(self, width=int(self.canvas_height/50), textvariable=self.fileRoot, bg=BLACK)
        fileRootEntry.grid(column=5, row=1)

        outEdgeLabel = tk.Label(self, width=int(self.canvas_height/50), height=int(self.canvas_height/600), text="Outer Number of Edges", bg=BG_COLOR)
        outEdgeLabel.grid(column=0, row=2)

        outEdgeEntry = tk.Entry(self, width=int(self.canvas_height/50), textvariable=self.outEdgeNum, bg=BLACK)
        outEdgeEntry.grid(column=1, row=2)

        inEdgeLabel = tk.Label(self, width=int(self.canvas_height/50), height=int(self.canvas_height/600), text="Inner Number of Edges", bg=BG_COLOR)
        inEdgeLabel.grid(column=2, row=2)

        inEdgeEntry = tk.Entry(self, width=int(self.canvas_height/50), textvariable=self.inEdgeNum, bg=BLACK)
        inEdgeEntry.grid(column=3, row=2)

        fileNameLabel = tk.Label(self, width=int(self.canvas_height/50), height=int(self.canvas_height/600), text="File Name", bg=BG_COLOR)
        fileNameLabel.grid(column=4, row=2)

        fileNameEntry = tk.Entry(self, width=int(self.canvas_height/50), textvariable=self.fileName, bg=BLACK)
        fileNameEntry.grid(column=5, row=2)

        TriangleNumLabel = tk.Label(self, width=int(self.canvas_height/50), height=int(self.canvas_height/600), text="Number of Triangles", bg=BG_COLOR)
        TriangleNumLabel.grid(column=0, row=3)

        reg = self.register(self.isNumber)
        TriangleNumEntry = tk.Entry(self, width=int(self.canvas_height/50), textvariable=self.triCount, validate='key', validatecommand= (reg, '%P', '%i'), bg=BLACK)
        TriangleNumEntry.grid(column=1, row=3)

        freeDrawButton = tk.Checkbutton(self, height=int(self.canvas_height/600), width=int(self.canvas_height/50), text="Free Draw", variable=self.freeDraw, bg=BG_COLOR)
        freeDrawButton.grid(column=2, row=3)

        randomButton = tk.Checkbutton(self, height=int(self.canvas_height/600), width=int(self.canvas_height/50), text="Randomize Vertices", variable=self.randomSet, bg=BG_COLOR)
        randomButton.grid(column=3, row=3)

        inOrOutButton = tk.Checkbutton(self, height=int(self.canvas_height/600), width=int(self.canvas_height/40), text="Inscribe the polygon or Not", variable=self.inOrOut, bg=BG_COLOR)
        inOrOutButton.grid(column=4, row=3)

    def isNumber(self, input, index):
        # lets text come through if its in a valid format
        # if len(input) <= int(index):
        #     return True
        if input[int(index)].isdigit():
            return True
        
        return False
    
    def getFreeDraw(self):
        return self.freeDraw.get()
    def getOuterEdgeNo(self):
        return self.outEdgeNum.get()
    def getInnerEdgeNo(self):
        return self.inEdgeNum.get()
    def getOutRad(self):
        return self.outRad.get()
    def getInRad(self):
        return self.inRad.get()
    def getFileRoot(self):
        return self.fileRoot.get()
    def getFileName(self):
        return self.fileName.get()
    def getTriCount(self):
        return self.triCount.get()
    def getRandomSet(self):
        return self.randomSet.get()


class show_results:

    def __init__(self):
        # The following is my list of instance variables, embarrassing.
        self.flags = False # Switches the call back from determining point in hole to base point
        self.stopFlag = False # Stops the call back from adding further points
        self.matCanvas = None # This holds the tk canvas widget
        self.ax2 = None # This will eventually hold the uniformization
        self.g_star_bar = None # This is the values of the g bar function
        self.period_gsb = None # This is the period of g bar
        self.currentGValue = None # This holds the selected indices g value
        self.currentGBarValue = None # This holds the selected indices g bar value
        self.changeGBar = None # This holds the change in g bar at that index compared to the next
        self.fig = None # This is the figure displaying everything
        self.axes = None # This holds the displayed mesh
        self.graphHolder = None # This is the frame the canvas is placed into
        self.canvas = None # This holds the canvas the figure is placed in
        self.toolbar = None # This is the toolbar of the matplotlib interface
        self.callbackName = None # This holds the current call back for clicking
        self.tri = None # This holds the actual triangulation object from the figure
        self.fileRoot = None # This holds the root of the files for the selected figure
        self.fileName = None # This holds the name of the files for the selected figure
        self.base_cell = None # This is the index of the voronoi cell where the user clicked inside the graph
        self.base_point = None # This holds the actual vertex of that index in base_cell
        self.cell_path = None # This will hold the cell path of the slit, the cells comprising the slit
        self.pointInHole = None # This is the selected point inside the hole in the figure
        self.omega_0 = None # To be frank I'm not totally sure, I think its the cell from which all paths start/ minimum angle to the line of the slit
        self.lambda_graph = None # This is the network of the figure
        self.shortest_paths = None # Array of shortest paths between omega_0 and each cell
        self.editor = None # Probably first to get removed, it will store the frame that pops up when you click while editing flux
        self.selectedIndex = None # index of the edge closest to the click when editing flux
        self.averageChange = None # Average change in g bar across the whole graph
        self.enteredInfo = None # Records information about all the files contained under the current root
        self.gui, self.canvas_width, self.canvas_height, = self.basicGui()
        self.controls, self.enteredFileRoot, self.enteredFileName = self.initializeFigure()
        self.modulus = tk.StringVar()


        self.graphConfigs = GraphConfig(width=self.canvas_width, height=self.canvas_height)
        self.gifConfig = GifConfig(self.canvas_height, self.canvas_width)

    def basicGui(self):
        gui = tk.Tk() # initialized Tk
        gui.state('zoomed')
        gui['bg'] = BG_COLOR # sets the background color to that grey
        gui.title("Manipulate data") 
        gui.columnconfigure(0, weight=1)
        gui.rowconfigure(0, weight=1)
        #print(gui.winfo_height(), gui.winfo_width())
        canvas_width = gui.winfo_width() 
        canvas_height = gui.winfo_height() # this and above set height and width variables that fill the screen
        #print(canvas_height, canvas_width)
        gui.protocol("WM_DELETE_WINDOW", exit)
        return gui, canvas_width, canvas_height
    
    def initializeFigure(self):
        controls = tk.Frame(self.gui, width=self.canvas_width, height=self.canvas_height/2, relief="ridge", bg=BG_COLOR)
        controls.columnconfigure(0, weight=1)
        controls.rowconfigure(0, weight=1)
        controls.grid(column=0, row=0)
        rootText = tk.Label(controls, height=int(self.canvas_height/224), width=int(self.canvas_height/14), text="Enter a file root, leave blank for none", bg=BG_COLOR)
        rootText.grid(column=0, row=0)
        fileRoot = tk.StringVar()
        tk.Entry(controls, width=int(self.canvas_width/50), textvariable=fileRoot, bg=BLACK).grid(column=1, row=0)
        nameText = tk.Label(controls, height=int(self.canvas_height/224), width=int(self.canvas_height/10), text="Enter a file name, should be in the following format to see varying levels of approximations: fileRoot_edgeNumber_triCount_steps", bg=BG_COLOR)
        nameText.grid(column=0, row=1)
        fileName = tk.StringVar()
        tk.Entry(controls, width=int(self.canvas_width/50), textvariable=fileName, bg=BLACK).grid(column=1, row=1)
        tk.Button(controls, height=1, width=int(self.canvas_width/50), command=self.loadFigure, text="Load", bg=BG_COLOR).grid(column=0,row=2)
        return controls, fileRoot, fileName
    
    def loadFigure(self):
        self.controls.destroy()
        if self.matCanvas is not None:
            self.matCanvas.destroy()
        self.controls = tk.Frame(self.gui, width=self.canvas_width, height=self.canvas_height/40, relief="ridge", bg=BG_COLOR)
        self.controls.columnconfigure(0, weight=1)
        self.controls.rowconfigure(0, weight=1)
        self.controls.grid(column=0, row=0)
        self.stopFlag = False
        self.flags = False
        text = tk.Label(self.controls, height=int(self.canvas_height/224), width=int(self.canvas_height/14), text="Click a point inside the hole, then click a point outside the graph to choose the line.", bg=BG_COLOR)
        text.grid(column=0, row=0)
        self.fileRoot = self.enteredFileRoot.get()
        self.fileName = self.enteredFileName.get()
        if self.fileRoot == '':
            self.tri = Triangulation.read(f'regions/{self.fileName}/{self.fileName}.poly')
            #self.fileNo = None
        else:
            self.tri = Triangulation.read(f'regions/{self.fileRoot}/{self.fileName}/{self.fileName}.poly')
            #self.fileNo = self.enteredFileName.get()[-1]
        self.fig, self.axes, self.graphHolder, self.canvas, self.toolbar, self.graphHolder, self.callbackName = self.basicTkinter()
        self.matCanvas = self.canvas.get_tk_widget()
        self.matCanvas.pack()
        self.graphConfigs.setSlit(False)
        self.show(True)
        self.graphConfigs.setSlit(True)
        # self.tri.show(
        #     show_edges=False,
        #     show_triangles=False,
        #     fig=self.fig,
        #     axes=self.axes
        # )
        # self.tri.show_voronoi_tesselation(
        #     show_vertex_indices=False,
        #     show_polygons=True,
        #     show_edges= True,
        #     fig=self.fig,
        #     axes=self.axes
        # )
        # self.axes.set_aspect('equal')
        # self.canvas.draw() 
    
    def basicTkinter(self):
        fig, axes = plt.subplots()
        #print(self.canvas_width)
        fig.set_figheight(6)
        fig.set_figwidth(6)
        graphHolder = tk.Frame(self.gui, width=self.canvas_width, height=self.canvas_height , relief="ridge", bg=BG_COLOR)
        graphHolder.grid(column=0, row=1)
        canvas = FigureCanvasTkAgg(fig, master = graphHolder)   
        toolbar = NavigationToolbar2Tk(canvas, graphHolder)
        toolbar.update()
        callbackName = fig.canvas.callbacks.connect('button_press_event', self.callback)
        return fig, axes, graphHolder, canvas, toolbar, graphHolder, callbackName
    
    def callback(self, event):
        if (self.fig.canvas.toolbar.mode != ''):
            #print(self.fig.canvas.toolbar.mode)
            return
        x = event.xdata
        y = event.ydata
        if x is None or y is None:
            return
        self.plotPoint(x,y)

    def plotPoint(self, x, y, noShow = False):
        if (not self.stopFlag):
            if (self.flags):
                self.base_cell = self.determinePolygon(x, y)
                self.base_point = self.tri.vertices[self.tri.contained_to_original_index[self.base_cell]]
            else:
                self.pointInHole = [x, y]
                plt.plot(x,y, 'bo', markersize = 2)
                plt.draw()

        if (not self.stopFlag):
            if (self.flags):
                plt.plot([self.base_point[0],self.pointInHole[0]],[self.base_point[1],self.pointInHole[1]],'bo-', markersize = 2, linewidth = 1)
                plt.draw()
                self.stopFlag = True
                if not noShow:
                    self.showSlitPath()

        self.flags = True

    def showResults(self):
        tk.mainloop()

    # This next section is all the stuff the graphs need to work

    # This seems to define what the flux will be for a given set of edges
    def flux_on_contributing_edges(self, edges):
        flux = 0.0
        # Adds the conductance times the difference in pde value for each edge in the path
        # Conductance seems to be some measure of how "important" an edge is
        for edge in edges:
            flux += self.tri.conductance[edge] * np.abs(
                self.tri.pde_values[edge[0]] - self.tri.pde_values[edge[1]]
            )
        return flux

    # Seems to basically act as a test, testing whether a given edge (i think tail-head is the circumcenters forming the edge) is intersecting the hole-base line
    # it seems to be used with circumcenters, so by edge i mean edges between circumcenters, not edges of the triangulation
    def segment_intersects_line(self, tail, head):
        tail_to_right = point_to_right_of_line_compiled(
            self.pointInHole[0],
            self.pointInHole[1],
            self.base_point[0],
            self.base_point[1],
            tail[0],
            tail[1]
        )
        head_to_right = point_to_right_of_line_compiled(
            self.pointInHole[0],
            self.pointInHole[1],
            self.base_point[0],
            self.base_point[1],
            head[0],
            head[1]
        )
        return head_to_right ^ tail_to_right # exclusive or

    # right and left in this context is relative to the line itself, so right is the right side of the line viewed so the hole is at the top

    # this tests to see if an edge connecting cells has a head to the right of the line but a tail to the left
    def segment_intersects_line_positive(self, tail, head):
        tail_to_right = point_to_right_of_line_compiled(self.pointInHole[0], self.pointInHole[1], self.base_point[0], self.base_point[1], tail[0], tail[1])
        head_to_right = point_to_right_of_line_compiled(self.pointInHole[0], self.pointInHole[1], self.base_point[0], self.base_point[1], head[0], head[1])
        return (head_to_right and not tail_to_right)

    # this tests to see if an edge connecting cells has a tail to the right of the line but a head to the left
    def segment_intersects_line_negative(self, tail, head):
        tail_to_right = point_to_right_of_line_compiled(self.pointInHole[0], self.pointInHole[1], self.base_point[0], self.base_point[1], tail[0], tail[1])
        head_to_right = point_to_right_of_line_compiled(self.pointInHole[0], self.pointInHole[1], self.base_point[0], self.base_point[1], head[0], head[1])
        return (not head_to_right and tail_to_right)
    # The reason these are seperate is because if only one of the head or tail is to the right of the segment, then we know that the line intersects that edge

    # triangulation has a set of contained polygons, this takes in each polygon and forms a list of edges
    @staticmethod
    def build_polygon_edges(polygon_vertices):
        edges = []
        # for loop adds all edges
        for i in range(len(polygon_vertices) - 1):
            edge = [
                polygon_vertices[i],
                polygon_vertices[i + 1]
            ]
            edges.append(edge)
        edges.append([polygon_vertices[-1], polygon_vertices[0]])
        # adds the wraparound edge
        return edges
    
    @staticmethod
    def flatten_list_of_lists(list_of_lists): # takes 
        return [item for sublist in list_of_lists for item in sublist]

    def slitPathCalculate(self):
        # Create the contained topology
        contained_topology_all = [
            [
                self.tri.original_to_contained_index[vertex]
                if vertex in self.tri.contained_to_original_index
                else -1
                for vertex in cell
            ] for cell in self.tri.vertex_topology
        ]
        # seems to add all indicies shared between otc and cto into lists, contained in a big dictionary. Each small list is for each cell
        contained_topology = [contained_topology_all[i] for i in self.tri.contained_to_original_index]
        # then he adds all of these that are contained in cells in the cto list to form his contained_topology
        # harboring a guess, this is probably all indicies of cells inside the figure wrapped up in neat packages for each cell.
        # Create cell path from base_cell to boundary_1
        poly = self.base_cell # starting at base_cell, more like the edge at base cell?
        poly_path_outward = []
        while poly != -1:
            cell_vertices = self.tri.contained_polygons[poly] # cell verticies is the verticies of the current polygon
            edges = self.build_polygon_edges(cell_vertices) # creates a list of edges for that polygon
            for i, edge in enumerate(edges): # i is the index of the edge
                # enumerates through each edge for the polygon
                # if the tail is to the right, then add that edge to the path, and then move down one edge
                if self.segment_intersects_line_negative(
                    self.tri.circumcenters[edge[0]],
                    self.tri.circumcenters[edge[1]]
                ):
                    poly_path_outward.append(poly)
                    poly = contained_topology[poly][i]
        # I vaguely remember him mentioning that the cell is numbered in a way that sides are like adjacent cells

        # Create cell path from base_cell to boundary_0
        poly = self.base_cell
        poly_path_inward = []
        while poly != -1:
            cell_vertices = self.tri.contained_polygons[poly]
            edges = self.build_polygon_edges(cell_vertices)
            for i, edge in enumerate(edges):
                if self.segment_intersects_line_positive(
                    self.tri.circumcenters[edge[0]],
                    self.tri.circumcenters[edge[1]]
                ):
                    poly_path_inward.append(poly)
                    poly = contained_topology[poly][i]
        # This is the exact same as base to 1, except we measure when the head is to the right, not tail

        # Create slit cell path by joining
        poly_path_inward = poly_path_inward[1:]
        #removes first element in path (base cell)
        poly_path_inward.reverse()
        # reverses the list
        self.cell_path = poly_path_inward + poly_path_outward
        # combines the list
        # What I think is happening here is finding the blue section of the graph, it adds the path of cells that the line intersects from hole to outer boundary
        slit_cell_vertices = set(self.flatten_list_of_lists([self.tri.contained_polygons[cell] for cell in self.cell_path]))

        # Create poly edge path on the left of line
        connected_component = []
        perpendicular_edges = []
        #this loops through every cell in the cell path
        for cell_path_index, cell in enumerate(reversed(self.cell_path)):
            flag = False
            edges = self.tri.make_polygon_edges(self.tri.contained_polygons[cell]) # creates a list of edges that make up the cell
            num_edges = len(edges)
            edge_index = -1
            while True: # runs continuously until
                edge_index = (edge_index + 1) % num_edges  # Next edge is one more than the previous edge, wrapping around to 0 at the last
                edge = edges[edge_index] # the current edge is the one at the edge index
                if flag: # if any previous edge has a tail to the right of line
                    if (not self.segment_intersects_line_positive( # and the current edge does not have a head to the right of the line, this stops the path from getting stuck in a loop
                        self.tri.circumcenters[edge[0]],
                        self.tri.circumcenters[edge[1]]
                    )):
                        if (contained_topology[cell][edge_index] != -1):  # This most likely checks to make sure the edge is not on the outside of the figure
                            connected_component.append(edge) # add the edge as a connected component
                            perpendicular_edges.append((cell, contained_topology[cell][edge_index])) # This adds a tuplet with the original cell, and the vertex of that cells current index, not sure what for, its never used
                    else:
                        break # this is the only way to break the while loop, it happens if a previous edge has a tail to the right, and we've now gotten to an edge that has a head to the right of the line
                if self.segment_intersects_line_negative( # if the current edge has a tail to the right of the line, set flag true
                    self.tri.circumcenters[edge[0]],
                    self.tri.circumcenters[edge[1]]
                ):
                    flag = True # this will cause the NEXT edge to be considered when adding it to the path
        # This creates a path of EDGES, currently we had a path of cellsx
        
        # Edges to weight
        self.edges_to_weight = []
        for cell_path_index, cell in enumerate(reversed(self.cell_path)): # again loops over the path of cells
            edges = self.tri.make_polygon_edges(self.tri.contained_polygons[cell]) # again retrives a list of all edges in that cell
            for edge in edges: # loops over each edge
                if self.segment_intersects_line(self.tri.circumcenters[edge[0]], self.tri.circumcenters[edge[1]]): # if the segment in the cell path intersects the line
                    self.edges_to_weight.append(edge) # adds every edge that intersects the line
        self.edges_to_weight = list(set(map(lambda x: tuple(np.sort(x)), self.edges_to_weight))) # ok so best guess, this builds a list of tuples, each tuple being the edge sorted with lowest index first, idk why that is necessary

        # Create contained_edges
        # triangulation_edges_reindexed = self.tri.original_to_contained_index[self.tri.triangulation_edges]
        # contained_edges = []
        # for edge in triangulation_edges_reindexed:
        #     if -1 not in edge:
        #         contained_edges.append(list(edge))
        # I think this just creates a list of all edges that don't have a vertex on a boundary

                # Choose omega_0 as the slit vertex that has the smallest angle relative to the line from the point in hole through
        # the circumcenter of the base_cell
        slit_path = [edge[0] for edge in connected_component] # the slit path is the sequence of edges from inside to out
        slit_path.append(connected_component[-1][1]) # adds the final edge
        # Connected component goes from outer boundary to inner boundary. Reverse after making slit
        slit_path = list(reversed(slit_path))
        angles = np.array([ # builds an array of angles between the circumcenter and line
            np.arctan2(
                self.tri.circumcenters[vertex][1] - self.pointInHole[0],
                self.tri.circumcenters[vertex][0] - self.pointInHole[1]
            )
            for vertex in slit_path
        ])
        self.omega_0 = slit_path[np.argmin(angles)] # makes omega_0 the minimum of this
        # Create graph of circumcenters (Lambda[0])
        self.lambda_graph = nx.Graph() # creates empty graph
        self.lambda_graph.add_nodes_from(range(len(self.tri.circumcenters))) # adds all circumcenters, not really maintaining structure just adding a node for each
        self.lambda_graph.add_edges_from(self.tri.voronoi_edges) # adds all edges connecting these nodes
        nx.set_edge_attributes(self.lambda_graph, values=1, name='weight') # sets all edges to have a value of 1
        for edge in self.edges_to_weight: # Sets every edge that intersects the line to have effectivly infinite weight
            self.lambda_graph.edges[edge[0], edge[1]]['weight'] = np.finfo(np.float32).max

    def redraw(self):
        self.controls = self.createNewConfigFrame(self.mainMenu, "Back", "Click a point inside the hole, then click a point outside the graph to choose the line.")
        self.flags = False
        self.stopFlag = False
        self.show()

    def updateLambdaGraph(self):
        self.shortest_paths = nx.single_source_dijkstra(self.lambda_graph, self.omega_0, target=None, cutoff=None, weight='weight')[1] # finds the shortest path around the figure to every node in the figure in a MASSIVE dictionary

    def uniformizationPage(self):
        self.updateLambdaGraph()
        uniformization = self.calculateUniformization()
        self.modulus.set("Modulus: " + str(self.findModulus(uniformization)))
        self.controls = self.createNewConfigFrame(self.disconnectAndReturnAndShow, "Back", self.modulus.get())
        approxButton = tk.Button(self.controls, height=int(self.canvas_height/200), width=int(self.canvas_height/60), text="See Approximations", command = self.showIntermediate, bg=BG_COLOR)
        approxButton.grid(column=0, row=2)
        # disconnects the ability to click normally
        self.fig.canvas.callbacks.disconnect(self.callbackName)
        # and adds the same back?? Idk why I did this but I'll keep it this way for now.
        self.callbackName = self.fig.canvas.callbacks.connect('button_press_event', self.callback)
        self.showUniformization(uniformization, False)
        #self.ax2.axis('on')

        self.canvas.draw()

    def compute_period(self):
        omega_0_cross_ray_edge_position = self.position(True, np.array([(self.omega_0 in edge) for edge in self.edges_to_weight]))
        omega_0_cross_ray_edge = tuple(self.edges_to_weight[omega_0_cross_ray_edge_position])
        if omega_0_cross_ray_edge[1] == self.omega_0:
            omega_0_clockwise_neighbor = omega_0_cross_ray_edge[0]
        else:
            omega_0_clockwise_neighbor = omega_0_cross_ray_edge[1]

        last_flux_contributing_edge = tuple(self.get_perpendicular_edge(omega_0_cross_ray_edge))
        closed_loop_flux = self.g_star_bar[omega_0_clockwise_neighbor] + (
            self.tri.conductance[last_flux_contributing_edge] * np.abs(
                self.tri.pde_values[last_flux_contributing_edge[0]] - self.tri.pde_values[last_flux_contributing_edge[1]]
            )
        )
        return closed_loop_flux

    #@numba.njit
    def cartesian_to_barycentric(self, x, y, x_1, y_1, x_2, y_2, x_3, y_3):
        det_T_inverse = 1 / ((x_1 - x_3) * (y_2 - y_3) + (x_3 - x_2) * (y_1 - y_3))
        lambda_1 = ((y_2 - y_3) * (x - x_3) + (x_3 - x_2) * (y - y_3)) * det_T_inverse
        lambda_2 = ((y_3 - y_1) * (x - x_3) + (x_1 - x_3) * (y - y_3)) * det_T_inverse
        return lambda_1, lambda_2


    # @numba.njit
    # def barycentric_to_cartesian(lambda_1, lambda_2, x_1, y_1, x_2, y_2, x_3, y_3):
    #     lambda_3 = 1 - lambda_1 - lambda_2
    #     x = lambda_1 * x_1 + lambda_2 * x_2 + lambda_3 * x_3
    #     y = lambda_1 * y_1 + lambda_2 * y_2 + lambda_3 * y_3
    #     return x, y


    #@numba.njit
    def barycentric_interpolation(self, x, y, x_1, y_1, x_2, y_2, x_3, y_3, f_1, f_2, f_3):
        lambda_1, lambda_2 = self.cartesian_to_barycentric(x, y, x_1, y_1, x_2, y_2, x_3, y_3)
        lambda_3 = 1 - lambda_1 - lambda_2
        return lambda_1 * f_1 + lambda_2 * f_2 + lambda_3 * f_3

    def calculateUniformization(self):
        #num_contained_polygons = len(self.tri.contained_polygons)
        self.g_star_bar = np.zeros(self.tri.num_triangles, dtype=np.float64) # creates a vector for each triangle
        perpendicular_edges_dict = {}
        for omega in range(self.tri.num_triangles): # Loops over each triangle
            #print(omega)
            #print(35 in self.shortest_paths)
            if omega in self.shortest_paths:
                edges = self.build_path_edges(self.shortest_paths[omega]) # Takes in a list of verticies (circumcenters) connecting omega_0 to the node, and builds an edge path
            else:
                edges = [] # Temporary fix, maybe. Why isnt there a path at that index?
            # Error is because the path is not contained in the shortest path index
            flux_contributing_edges = []
            for edge in edges:
                flux_contributing_edges.append(tuple(self.get_perpendicular_edge(edge))) # This creates a sequence of verticies (triangle verticies) connecting omega_0 to the desired end vertex
            perpendicular_edges_dict[omega] = flux_contributing_edges # adds this (triangle vertex0) path to the dictionary 
            self.g_star_bar[omega] = self.flux_on_contributing_edges(flux_contributing_edges) # adds the flux for this path to whatever the g_star_bar vector is, which is apparently the harmonic conjugate to g, the pde solution

        # Interpolate the value of pde_solution to get its values on the omegas
        #pde_on_omega_values = [np.mean(self.tri.pde_values[self.tri.triangles[i]]) for i in range(self.tri.num_triangles)] # takes in the solution on the triangle vertices and gives a solution on the circumcenter/node

        pde_on_omega_values = [
            self.barycentric_interpolation(
                self.tri.circumcenters[i][0], self.tri.circumcenters[i][1],
                self.tri.triangle_coordinates[i][0][0], self.tri.triangle_coordinates[i][0][1],
                self.tri.triangle_coordinates[i][1][0], self.tri.triangle_coordinates[i][1][1],
                self.tri.triangle_coordinates[i][2][0], self.tri.triangle_coordinates[i][2][1],
                self.tri.pde_values[self.tri.triangles[i][0]],
                self.tri.pde_values[self.tri.triangles[i][1]],
                self.tri.pde_values[self.tri.triangles[i][2]],
            ) for i in range(self.tri.num_triangles)
        ]

        #self.period_gsb = np.max(self.g_star_bar) # the maximum value in the vector is stored, so i guess the largest flux
        
        self.period_gsb = self.compute_period()
        
        # TODO: allow the last edge so we get all the
        uniformization = np.exp(2 * np.pi / self.period_gsb * (pde_on_omega_values + 1j * self.g_star_bar)) # Uniformizes the triangulation into an approximation of the annulus

        # Level curves for gsb
        g_star_bar_interpolated_interior = np.array([np.mean(self.g_star_bar[poly]) for poly in self.tri.contained_polygons]) # vector of the average fluxes for each contained cell
        min_, max_ = np.min(g_star_bar_interpolated_interior), np.max(g_star_bar_interpolated_interior) # saves the maximum and minimum average flux for each contained cell
        heights = np.linspace(min_, max_, num=100) # creates an array of 100 evenely spaced numbers between the max and minimum flux for the contained polygons

        contained_triangle_indicator = np.all(self.tri.vertex_boundary_markers[self.tri.triangles] == 0, axis=1)
        contained_triangles = np.where(contained_triangle_indicator)[0]
        slit_cell_vertices = set(self.flatten_list_of_lists([self.tri.contained_polygons[cell] for cell in self.cell_path]))
        contained_triangle_minus_slit = list(set(contained_triangles).difference(slit_cell_vertices))
        #print(uniformization)
        return uniformization
    
    @staticmethod
    def findModulus(uni):
        reals = np.real(uni)
        imags = np.imag(uni)
        reals = np.square(reals)
        imags = np.square(imags)
        radii = np.sqrt(np.add(reals, imags))
        radii = np.sort(radii)
        averageSmall = (radii[0] + radii[1] + radii[2] + radii[3]) / 4
        averageLarge = (radii[-1] + radii[-2] + radii[-3] + radii[-4]) / 4
        #print(averageLarge, averageSmall)
        modulus = (1 / (2 * np.pi)) * np.log10(averageLarge / averageSmall)
        return modulus

    def showUniformization(self, uniformization, animationFlag):
        # refreshes graph with updated information
        xZoom = self.axes.get_xlim()
        yZoom = self.axes.get_ylim()
        plt.close("all")
        self.fig, (self.axes, self.ax2) = plt.subplots(1,2, sharex=False, sharey=False)
        self.fig.set_figheight(5)
        self.fig.set_figwidth(14)
        self.graphHolder.destroy()
        self.graphHolder = tk.Frame(self.gui, width=self.canvas_width, height=self.canvas_height , relief="ridge", bg=BG_COLOR)
        self.graphHolder.grid(column=0, row=1)
        self.canvas = FigureCanvasTkAgg(self.fig, master = self.graphHolder)   
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.graphHolder)
        self.toolbar.update()
        self.matCanvas = self.canvas.get_tk_widget()
        self.matCanvas.pack()

        # Needs functionality to help show if the figure was inscribed or not, basically just check to see if all the minimums are the same or if all the maximums are the same, if neither, show minimums
        # innerRad = self.findMinRadius(1)
        # outerRad = self.findMinRadius(2)

        # point0 = self.tri.region.points_in_holes[0]
        # self.axes.add_patch(Circle(point0, radius =outerRad,fill = False, edgecolor = 'g',
        #             linestyle = '--',linewidth = 1.25))

        # self.axes.add_patch(Circle(point0, radius =innerRad,fill = False, edgecolor = 'g',
        #             linestyle = '--',linewidth = 1.25))

        vert, edge, triangle, vInd, tInd, level, singLevel = self.graphConfigs.getConfigsTri()
        self.tri.show(
            show_vertices=vert,
            show_edges=edge,
            show_triangles=triangle,
            show_vertex_indices=vInd,
            show_triangle_indices=tInd,
            show_level_curves=level,
            #show_singular_level_curves=self.show_singular_level_curves_tri,
            highlight_vertices=None,
            fig=self.fig,
            axes=self.axes
        )
        
        vInd, pInd, vert, edge, poly, region = self.graphConfigs.getConfigsVor()
        self.tri.show_voronoi_tesselation(
            show_vertex_indices=vInd,
            show_polygon_indices=pInd,
            show_vertices=vert,
            show_edges=edge,
            show_polygons=poly,
            show_region=region,
            fig=self.fig,
            axes=self.axes
        )
        self.axes.set_aspect('equal')
        self.ax2.scatter(
            1 * np.real(uniformization),
            1 * np.imag(uniformization),
            s=50,
            linewidths = .1
        ) #Plots image of uniformization map
        self.ax2.set_aspect('equal')
        self.fig.tight_layout()
        self.ax2.axis('on')
        self.axes.axis('on')
        self.canvas.draw()
        if animationFlag is False:
            self.toolbar.push_current() # Allows the home button to center the figure to original, yay!
            self.axes.set_xlim(xZoom)
            self.axes.set_ylim(yZoom)
        

    def add_voronoi_edges_to_axes(self, edge_list, axes, color): # I think this is exactly what it is called
        lines = [
            [
                tuple(self.tri.circumcenters[edge[0]]),
                tuple(self.tri.circumcenters[edge[1]])
            ] for edge in edge_list
        ]
        colors = np.tile(color, (len(edge_list), 1))
        line_collection = mc.LineCollection(lines, linewidths=2, colors=colors)
        axes.add_collection(line_collection)

    # Find the perpendicular edges to the lambda path
    @staticmethod
    def build_path_edges(vertices): # specifically this takes in a list of verticies, and creates a list of edges connecting the verticies in order that they are stored
        #print(vertices)
        edges = []
        for i in range(len(vertices) - 1):
            edge = [
                vertices[i],
                vertices[i + 1]
            ]
            edges.append(edge)
        return edges

    # Make mapping from edges on the cells to the perpendicular edges in triangulation
    @staticmethod
    @numba.njit 
    def position(x, array): # specifically finds the index of the inputted parameter in the inputted array
        for i in range(len(array)):
            if x == array[i]:
                return i
        return -1

    def get_perpendicular_edge(self, edge):
        """Think of the omega path as being triangles instead. This finds which edge of the triangle
        edge[0] is adjacent to triangle edge[1]"""
        triangle_edges = self.tri.make_polygon_edges(self.tri.triangles[edge[0]]) # retrieves all edges around the cell with circumcenter that is the head of the input edge
        edge_index = self.position(edge[1], self.tri.topology[edge[0]]) # Finds the triangle number that is equal to the tail circumcenter
        perpendicular_edge = triangle_edges[edge_index] # The perpendicuar edge is thus the edge of the cell with the triangle edge
        return perpendicular_edge

    def determinePolygon(self, x, y):
        # Finds the cell closest to the click
        polygon_coordinates = [
                np.array(
                    list(map(lambda x: self.tri.circumcenters[x], polygon))
                )
                for polygon in self.tri.contained_polygons
            ]
        
        voronoiCenter = np.array(list(map(
            lambda x: np.mean(x, axis=0),
            polygon_coordinates
        )))
        distanceToCenter = np.array([ # builds an array of distances between barycenters and click
            (center[0] - x)**2 + (center[1] - y)**2
            for center in voronoiCenter
        ])
        return np.argmin(distanceToCenter)
    
    def findMaxRadius(self, boundary):
        array = []
        for i in range(len(self.tri.vertex_boundary_markers)):
            if (self.tri.vertex_boundary_markers[i] == boundary):
                array.append(self.tri.vertices[i])
        ys = []
        xs = []
        for vertex in array:
            xs.append(vertex[0])
            ys.append(vertex[1])
        xsa = np.array(xs)
        ysa = np.array(ys)
        rads = np.add(np.square(np.subtract(xsa, self.tri.region.points_in_holes[0][0])), np.square(np.subtract(ysa, self.tri.region.points_in_holes[0][1])))
        rads = np.sqrt(rads)
        return np.max(rads)
    
    def findMinRadius(self, boundary):
        array = []
        for i in range(len(self.tri.vertex_boundary_markers)):
            if (self.tri.vertex_boundary_markers[i] == boundary):
                array.append(self.tri.vertices[i])
        ys = []
        xs = []
        for vertex in array:
            xs.append(vertex[0])
            ys.append(vertex[1])
        xsa = np.array(xs)
        ysa = np.array(ys)
        rads = np.add(np.square(np.subtract(xsa, self.tri.region.points_in_holes[0][0])), np.square(np.subtract(ysa, self.tri.region.points_in_holes[0][1])))
        rads = np.sqrt(rads)
        return np.min(rads)
    
    def findAverageRad(self, maxOrMin = False, boundary = 0):
        array = []
        for i in range(len(self.tri.vertex_boundary_markers)):
            if (self.tri.vertex_boundary_markers[i] >= 1):
                array.append(self.tri.vertices[i])
        ys = []
        xs = []
        for vertex in array:
            xs.append(vertex[0])
            ys.append(vertex[1])
        xsa = np.array(xs)
        ysa = np.array(ys)
        rads = np.add(np.square(np.subtract(xsa, self.tri.region.points_in_holes[0][0])), np.square(np.subtract(ysa, self.tri.region.points_in_holes[0][1])))
        rads = np.sqrt(rads)
        rads.sort()
        average = 0
        for i in range(4):
            if maxOrMin:
                average += rads[len(rads) - i - 1]
            else:
                average += rads[i]
        average /= 4
        return average
    
    def findInputtedRadius(self):
        vertex = self.tri.vertices[self.findMaxRadius(2)]
        radiusIn = math.sqrt(vertex[0] ** 2 + vertex[1] ** 2)
        vertex = self.tri.vertices[self.findMaxRadius(1)]
        radiusOut = math.sqrt(vertex[0] ** 2 + vertex[1] ** 2)
        return (radiusIn, radiusOut)
    
    def show(self, first = False): 
        xZoom = self.axes.get_xlim()
        yZoom = self.axes.get_ylim()
        plt.close("all")
        # TODO try and preserve zoom information for the display
        # refreshes graph with updated information
        self.fig, self.axes = plt.subplots()
        self.fig.set_figheight(6)
        self.fig.set_figwidth(6)
        self.graphHolder.destroy()
        self.graphHolder = tk.Frame(self.gui, width=self.canvas_width, height=self.canvas_height , relief="ridge", bg=BG_COLOR)
        self.graphHolder.grid(column=0, row=1)
        self.canvas = FigureCanvasTkAgg(self.fig, master = self.graphHolder)   
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.graphHolder)
        self.toolbar.update()
        self.matCanvas = self.canvas.get_tk_widget()
        self.matCanvas.pack()

        self.callbackName = self.fig.canvas.callbacks.connect('button_press_event', self.callback)
        
        vert, edge, triangle, vInd, tInd, level, singLevel = self.graphConfigs.getConfigsTri()
        self.tri.show(
            show_vertices=vert,
            show_edges=edge,
            show_triangles=triangle,
            show_vertex_indices=vInd,
            show_triangle_indices=tInd,
            show_level_curves=level,
            #show_singular_level_curves=self.show_singular_level_curves_tri,
            highlight_vertices=None,
            fig=self.fig,
            axes=self.axes
        )
        
        vInd, pInd, vert, edge, poly, region = self.graphConfigs.getConfigsVor()
        self.tri.show_voronoi_tesselation(
            show_vertex_indices=vInd,
            show_polygon_indices=pInd,
            show_vertices=vert,
            show_edges=edge,
            show_polygons=poly,
            show_region=region,
            fig=self.fig,
            axes=self.axes
        )
        self.axes.set_aspect('equal')
        if self.graphConfigs.getSlit():
            self.showSlit()
        else:
            self.canvas.draw()
        if not first:
            self.toolbar.push_current()
            self.axes.set_xlim(xZoom)
            self.axes.set_ylim(yZoom)

    def showSlit(self):
        # highlights the slit
        xZoom = self.axes.get_xlim()
        yZoom = self.axes.get_ylim()
        slit_cell_vertices = set(self.flatten_list_of_lists([self.tri.contained_polygons[cell] for cell in self.cell_path]))
        # This is causing problems?
        self.tri.show_voronoi_tesselation(
            highlight_polygons=self.cell_path,
            highlight_vertices=list(slit_cell_vertices),
            fig=self.fig,
            axes=self.axes
        )
        self.canvas.draw()
        self.toolbar.push_current()
        self.axes.set_xlim(xZoom)
        self.axes.set_ylim(yZoom)



    def mainMenu(self):
        self.show()
        if self.graphConfigs.getSlit():
            self.showSlit()
        self.controls.grid_remove()
        # adds buttons to various modes, currently just graph edit and edit flux
        mainMenu = tk.Frame(self.gui, width=self.canvas_width, height=self.canvas_height, bg=BG_COLOR)
        mainMenu.columnconfigure(0, weight=1)
        mainMenu.rowconfigure(0, weight=1)
        mainMenu.grid(column=0, row=0)

        graphButton = tk.Button(mainMenu, height=int(self.canvas_height/200), width=int(self.canvas_height/60), text="Graph Display Options", command = self.graphConfig, bg=BG_COLOR)
        graphButton.grid(column=0, row=0)

        fluxButton = tk.Button(mainMenu, height=int(self.canvas_height/200), width=int(self.canvas_height/60), text="Edit Flux", command = self.fluxConfig, bg=BG_COLOR)
        fluxButton.grid(column=1, row=0)

        pathButton = tk.Button(mainMenu, height=int(self.canvas_height/200), width=int(self.canvas_height/60), text="Show Paths", command = self.pathFinder, bg=BG_COLOR)
        pathButton.grid(column=2, row=0)

        uniformButton = tk.Button(mainMenu, height=int(self.canvas_height/200), width=int(self.canvas_height/60), text="Show Uniformization and Approximations", command = self.uniformizationPage, bg=BG_COLOR)
        uniformButton.grid(column=3, row=0)

        redrawButton = tk.Button(mainMenu, height=int(self.canvas_height/200), width=int(self.canvas_height/60), text="Choose another slit path", command = self.redraw, bg=BG_COLOR)
        redrawButton.grid(column=4, row=0)

        newDrawButton = tk.Button(mainMenu, height=int(self.canvas_height/200), width=int(self.canvas_height/60), text="Draw another figure", command = self.showDraw, bg=BG_COLOR)
        newDrawButton.grid(column=5, row=0)

        animButton = tk.Button(mainMenu, height=int(self.canvas_height/200), width=int(self.canvas_height/60), text="Create Animation", command = self.animationConfig, bg=BG_COLOR)
        animButton.grid(column=0, row=1)

        numericalButton = tk.Button(mainMenu, height=int(self.canvas_height/200), width=int(self.canvas_height/60), text="Show Numerical Values", command = self.showFunction, bg=BG_COLOR)
        numericalButton.grid(column=1, row=1)

        numericalButton = tk.Button(mainMenu, height=int(self.canvas_height/200), width=int(self.canvas_height/60), text="Load a new Figure", command = self.loadNew, bg=BG_COLOR)
        numericalButton.grid(column=2, row=1)
        


        self.controls = mainMenu

    def loadNew(self):
        self.controls.grid_remove()
        self.stopFlag = False
        self.flags = False
        self.controls, self.enteredFileRoot, self.enteredFileName = self.initializeFigure()

    def showSlitPath(self):
        # displays slit path and takes to the main menu
        self.slitPathCalculate()
        self.mainMenu()

    def nearestEdge(self, x, y):
        distanceToMidPoints = np.array([ # builds an array of distance between click and edge midpoints
            ((((self.tri.circumcenters[edge[0]][0] + self.tri.circumcenters[edge[1]][0]) * .5) - x)**2 +
            (((self.tri.circumcenters[edge[0]][1] + self.tri.circumcenters[edge[1]][1]) * .5) - y)**2)
            for edge in self.tri.voronoi_edges
        ])

        # finds the smallest distance
        index = np.argmin(distanceToMidPoints)

        # Debug mode stuff that I will remove later/use to highlight and remove
        plt.plot(x,y, 'bo', markersize = 2)
        plt.plot(self.tri.circumcenters[self.tri.voronoi_edges[index][0]][0], self.tri.circumcenters[self.tri.voronoi_edges[index][0]][1], 'bo', markersize = 2)
        plt.plot(self.tri.circumcenters[self.tri.voronoi_edges[index][1]][0], self.tri.circumcenters[self.tri.voronoi_edges[index][1]][1], 'bo', markersize = 2)
        plt.draw()
        return index

    @staticmethod
    def validateText(input, index):
        # lets text come through if its in a valid format
        if input.count(".") > 1:
            return False
        if len(input) <= int(index):
            return True
        if input[int(index)].isdigit():
            return True
        elif input[int(index)] == '.':
            return True
        
        return False
    
    def editFluxGraph(self):
        if self.editor.children['!entry'] is None:
            return
        if self.editor.children['!entry'].get() != '':
            #print('a', self.editor.children['!entry'].get(), 'b')
            # edits edge flux in lambda graph
            self.lambda_graph.edges[self.tri.voronoi_edges[self.selectedIndex][0], self.tri.voronoi_edges[self.selectedIndex][1]]['weight'] = float(self.editor.children['!entry'].get())
            # removes popup, and connects call back back to the edge finder, renables back button
            self.editor.destroy()
            self.editor = None
            #self.callbackName = self.fig.canvas.callbacks.connect('button_press_event', self.fluxFinder)
            self.controls.children['!button']['state'] = 'normal'
            self.show()
            #self.fig.canvas.callbacks.disconnect(self.callbackName)
            self.callbackName = self.fig.canvas.callbacks.connect('button_press_event', self.fluxFinder)

    def fluxFinder(self, event):
        if (self.fig.canvas.toolbar.mode != ''):
            #print(self.fig.canvas.toolbar.mode)
            return
        self.updateLambdaGraph()
        x = event.xdata
        y = event.ydata        
        if x is None or y is None:
            return
        # finds index of the edge closest to the mouse click
        self.selectedIndex = self.nearestEdge(x, y)

        # adds a entry and button to input user data to the flux graph, and places them at a point in the middle of the graph
        self.editor = tk.Frame(self.gui, height = int(self.canvas_height/50), width=int(self.canvas_height/50), bg=BG_COLOR)
        fluxValue = tk.StringVar()
        reg = self.gui.register(self.validateText)
        currentFlux = self.lambda_graph.edges[self.tri.voronoi_edges[self.selectedIndex][0], self.tri.voronoi_edges[self.selectedIndex][1]]['weight']
        fluxValue.set(str(currentFlux))
        fluxInput = tk.Entry(self.editor, width=int(self.canvas_height/50), bg=BLACK, validate='key', validatecommand= (reg, '%P', '%i'), textvariable = fluxValue)
        fluxInput.grid(column=0, row=0)
        sendButton = tk.Button(self.editor, height=1, width=1, bg=BG_COLOR, command= self.editFluxGraph)
        sendButton.grid(column=1, row=0)
        self.editor.place(x=int(self.canvas_width / 2), y=int(self.canvas_height/2))
        # disables back button until data is entered
        self.controls.children['!button']['state'] = 'disabled'
        # disables clicking entirely
        self.fig.canvas.callbacks.disconnect(self.callbackName)

    def disconnectAndReturn(self):
        # disconnects weird callbacks and returns to main menu
        self.fig.canvas.callbacks.disconnect(self.callbackName)
        self.callbackName = self.fig.canvas.callbacks.connect('button_press_event', self.callback)
        self.mainMenu()

    def disconnectAndReturnAndShow(self):
        self.show()
        self.disconnectAndReturn()

    def fluxConfig(self):
        # removes old controls and adds new scene
        self.controls = self.createNewConfigFrame(self.disconnectAndReturn, "Back", "Click on an edge to edit it's flux. Press enter to set value.")
        # disconnects the ability to click normally
        self.fig.canvas.callbacks.disconnect(self.callbackName)
        # and adds a new click that finds nearest edge in the voronai graph
        self.callbackName = self.fig.canvas.callbacks.connect('button_press_event', self.fluxFinder)

    def createNewConfigFrame(self, commandB, textB, textL):
        self.controls.grid_remove()
        configs = tk.Frame(self.gui, width=self.canvas_width, height=self.canvas_height, bg=BG_COLOR)
        configs.columnconfigure(0, weight=1)
        configs.rowconfigure(0, weight=1)
        configs.grid(column=0, row=0)
        i = 0
        if textL is not None:
            instructions = tk.Label(configs, height=int(self.canvas_height/540), width=int(self.canvas_height/10), text=textL, bg=BG_COLOR)
            instructions.grid(column=0, row=0)
            i = 1
        if commandB is not None:
            backButton = tk.Button(configs, height=int(self.canvas_height/540), width=int(self.canvas_height/10), text=textB, command = commandB, bg=BG_COLOR)
            backButton.grid(column=0, row=i)
        return configs

    def graphConfig(self):
        # removes the old controls and adds a new scene
        self.controls.grid_remove()
        self.controls = self.graphConfigs.getFrame(parent = self.gui)

        drawButton = tk.Button(self.controls, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Display Graph", command = self.show, bg=BG_COLOR)
        drawButton.grid(column=6, row=2)
        backButton = tk.Button(self.controls, height=int(self.canvas_height/540), width=int(self.canvas_height/50), text="Back", command = self.mainMenu, bg=BG_COLOR)
        backButton.grid(column=0, row=2)

    # actual controller for displaying paths, sets what happens when you click
    def pathSelector(self, event):
        if (self.fig.canvas.toolbar.mode != ''):
            #print(self.fig.canvas.toolbar.mode)
            return
        self.show()
        self.callbackName = self.fig.canvas.callbacks.connect('button_press_event', self.pathSelector)
        x = event.xdata
        y = event.ydata

        if x is None or y is None:
            return
        distanceToVerticies = np.array([ # builds an array of distance between click and edge midpoints
            ((vertex[0] - x)**2 +
            (vertex[1] - y)**2)
            for vertex in self.tri.circumcenters
        ])

        omega = distanceToVerticies.argmin()
        self.add_voronoi_edges_to_axes(self.build_path_edges(self.shortest_paths[omega]), self.axes, color=[1, 0, 0])
        self.canvas.draw()

    # changes mode to display paths
    def pathFinder(self):
        self.updateLambdaGraph()
        self.controls = self.createNewConfigFrame(self.mainMenu, "Back", "Click on a vertex to see the path from omega0 to that vertex.")
        self.fig.canvas.callbacks.disconnect(self.callbackName)
        # and adds a new click that finds nearest edge in the voronai graph
        self.callbackName = self.fig.canvas.callbacks.connect('button_press_event', self.pathSelector)

    def nextGraph(self):
        root, edge, triCount, step = self.fileName.split("_")
        edgeNum = int(edge)
        stepNum = int(step)
        triNum = int(triCount)
        if edgeNum < int(self.enteredInfo[3]):
            edgeNum += 1
        elif triNum < int(self.enteredInfo[8]):
            triNum += int(int(self.enteredInfo[8]) / int(self.enteredInfo[9]))
        else:
            stepNum += 1
        name = root + "_" + str(edgeNum) + "_" + str(triNum) + "_" + str(stepNum)
        try:
            self.tri = Triangulation.read(f'regions/{self.fileRoot}/{name}/{name}.poly')
        except FileNotFoundError:
            print("File Not Found: ", name)
            return
        self.fileName = name
        self.flags = False
        self.stopFlag = False
        hole_x, hole_y = self.tri.region.points_in_holes[0]
        self.plotPoint(hole_x, hole_y)
        self.plotPoint(hole_x + 1000, hole_y)
        self.slitPathCalculate()
        self.updateLambdaGraph()
        uniformization = self.calculateUniformization()
        self.showUniformization(uniformization, False)
        self.modulus.set("Modulus: " + str(self.findModulus(uniformization)))
        self.updateMod()
    
    def prevGraph(self):
        root, edge, triCount, step = self.fileName.split("_")
        edgeNum = int(edge)
        stepNum = int(step)
        triNum = int(triCount)
        if stepNum > 0:
            stepNum -= 1
        elif triNum > int(self.enteredInfo[7]):
            triNum -= int(int(self.enteredInfo[8]) / int(self.enteredInfo[9]))
        elif edgeNum > int(self.enteredInfo[2]):
            edgeNum -= 1
        name = root + "_" + str(edgeNum) + "_" + str(triNum) + "_" + str(stepNum)
        try:
            self.tri = Triangulation.read(f'regions/{self.fileRoot}/{name}/{name}.poly')
        except FileNotFoundError:
            print("File Not Found: ", name)
            return
        self.fileName = name
        self.flags = False
        self.stopFlag = False
        hole_x, hole_y = self.tri.region.points_in_holes[0]
        self.plotPoint(hole_x, hole_y)
        self.plotPoint(hole_x + 1000, hole_y)
        self.slitPathCalculate()
        self.updateLambdaGraph()
        uniformization = self.calculateUniformization()
        self.showUniformization(uniformization, False)
        self.modulus.set("Modulus: " + str(self.findModulus(uniformization)))
        self.updateMod()

    def updateMod(self):
        self.controls = self.createNewConfigFrame(self.mainMenu, "Back", "Click buttons to switch between approximations.")
        buttonHolder = tk.Frame(self.controls, width=self.canvas_width, height=self.canvas_height, bg=BG_COLOR)
        buttonHolder.columnconfigure(0, weight=1)
        buttonHolder.rowconfigure(0, weight=1)
        buttonHolder.grid(column=0, row=2)
        nextButton = tk.Button(buttonHolder, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Next Graph", command = self.nextGraph, bg=BG_COLOR)
        nextButton.grid(column=2, row=0)
        pictureButton = tk.Button(buttonHolder, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Save Photo", command = self.showNSave, bg=BG_COLOR)
        pictureButton.grid(column=1, row=0)
        previousButton = tk.Button(buttonHolder, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Previous Graph", command = self.prevGraph, bg=BG_COLOR)
        previousButton.grid(column=0, row=0)
        modulusLabel = tk.Label(buttonHolder, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text=self.modulus.get(), bg=BG_COLOR)
        modulusLabel.grid(column=0, row=3, columnspan=3)
    
    def showIntermediate(self):
        if self.fileRoot == '':
            self.controls = self.createNewConfigFrame(self.mainMenu, "Back", "Your File does not have a root!")
            return
        directory = Path('regions/' + self.fileRoot)
        directory_info = directory / (self.fileRoot + "_info.txt")
        if (not directory.is_dir()) or (not directory_info.is_file()):
            self.controls = self.createNewConfigFrame(self.mainMenu, "Back", "The inputted file is not part of a chain!")
            return
        self.enteredInfo = []
        with open(directory_info, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i != 0:
                    self.enteredInfo.append(line)
        self.flags = False
        self.stopFlag = False
        hole_x, hole_y = self.tri.region.points_in_holes[0]
        self.plotPoint(hole_x, hole_y)
        self.plotPoint(hole_x + 1000, hole_y)
        self.slitPathCalculate()
        self.show()
        self.updateLambdaGraph()
        uniformization = self.calculateUniformization()
        self.showUniformization(uniformization, False)
        self.modulus.set("Modulus: " + str(self.findModulus(uniformization)))
        self.updateMod()

    def nearestVertexInPoly(self, x, y, polygon):
        distanceToVertices = np.array([ # builds an array of distance between click and vertices
            ((self.tri.circumcenters[index][0] - x)**2 +
            (self.tri.circumcenters[index][1] - y)**2)
            for index in self.tri.contained_polygons[polygon]
        ])
        return self.tri.contained_polygons[polygon][np.argmin(distanceToVertices)]
    
    def voroniFinder(self, event):
        if (self.fig.canvas.toolbar.mode != ''):
            #print(self.fig.canvas.toolbar.mode)
            return
        self.show()
        x = event.xdata
        y = event.ydata
        if x is None or y is None:
            self.showFunction()
        voronoiIndex = self.determinePolygon(x, y)
        vertexIndex = self.nearestVertexInPoly(x, y, voronoiIndex)
        cell = self.tri.vertices[self.tri.contained_to_original_index[voronoiIndex]]
        vertex = self.tri.circumcenters[vertexIndex]
        plt.plot(vertex[0], vertex[1], 'bo', markersize = 2)
        plt.plot(cell[0], cell[1], 'bo', markersize = 2)
        # for vertex in self.tri.contained_polygons[voronoiIndex]:
        #     plt.plot(self.tri.circumcenters[vertex][0], self.tri.circumcenters[vertex][1], 'bo', markersize = 2)
        # COME BACK TO TODO, not finding period thats on eric, though displaying the rest is fine, even if the difference will be handled elsewhere (with sample of 100 random vertices).
        xZoom = self.axes.get_xlim()
        yZoom = self.axes.get_ylim()
        plt.draw()
        self.toolbar.push_current()
        self.axes.set_xlim(xZoom)
        self.axes.set_ylim(yZoom)
        self.currentGValue = self.tri.pde_values[self.tri.contained_to_original_index[voronoiIndex]]
        self.currentGBarValue = self.g_star_bar[vertexIndex]
        if vertexIndex in self.tri.contained_polygons[voronoiIndex]:
            adjacentBarInd = self.tri.contained_polygons[voronoiIndex].index(vertexIndex) + 1
            # print(adjacentBarInd)
            if adjacentBarInd >= len(self.tri.contained_polygons[voronoiIndex]):
                adjacentBarInd = 0
            # print(self.tri.contained_polygons[voronoiIndex][adjacentBarInd])
            self.changeGBar = abs(self.g_star_bar[vertexIndex] - self.g_star_bar[self.tri.contained_polygons[voronoiIndex][adjacentBarInd]])
        else:
            self.changeGBar = None
        # print("Hopefully correct polygon: ", self.tri.contained_polygons[voronoiIndex], self.tri.contained_to_original_index[voronoiIndex])
        # print("Using Circumcenters: ", self.tri.circumcenters[vertexIndex])
        # print("Hopefully correct center: ", cell, voronoiIndex)
        # print("Hopefully correct vertex, using circumcenters: ", vertex, vertexIndex)
        self.showGBarAfter()
    
    def showFunction(self):
        self.updateLambdaGraph()
        self.calculateUniformization()
        self.fig.canvas.callbacks.disconnect(self.callbackName)
        self.callbackName = self.fig.canvas.callbacks.connect('button_press_event', self.voroniFinder)
        self.controls.grid_remove()
        self.controls = tk.Frame(self.gui, width=self.canvas_width, height=self.canvas_height, bg=BG_COLOR)
        self.controls.grid(column=0, row=0)
        self.controls.columnconfigure(0, weight=1)
        self.controls.rowconfigure(0, weight=1)
        if len(self.tri.contained_polygons) >= 100:
            size = 100
        else:
            size = len(self.tri.contained_polygons)
        changeArray = np.zeros(size)
        for i in range(size):
            index = random.randint(0, len(self.tri.contained_polygons) - 1)
            vertexInd = random.randint(0, len(self.tri.contained_polygons[index]) - 1)
            adjacentBarInd = vertexInd + 1
            if adjacentBarInd >= len(self.tri.contained_polygons[index]):
                adjacentBarInd = 0
            changeArray[i] = abs(self.g_star_bar[self.tri.contained_polygons[index][vertexInd]] - self.g_star_bar[self.tri.contained_polygons[index][adjacentBarInd]])
        self.averageChange = np.mean(changeArray)
        instructionLabel = tk.Label(self.controls, height=int(self.canvas_height/540), width=int(self.canvas_height/20), text="Click on a vertex to display function information")
        instructionLabel.grid(column=0, row=1)
        backButton = tk.Button(self.controls, height=int(self.canvas_height/540), width=int(self.canvas_height/50), text="back", command = self.disconnectAndReturn, bg=BG_COLOR)
        backButton.grid(column=0, row=2)
    
    def showGBarAfter(self):
        self.fig.canvas.callbacks.disconnect(self.callbackName)
        self.callbackName = self.fig.canvas.callbacks.connect('button_press_event', self.voroniFinder)
        self.controls.grid_remove()
        self.controls = tk.Frame(self.gui, width=self.canvas_width, height=self.canvas_height, bg=BG_COLOR)
        self.controls.grid(column=0, row=0)
        self.controls.columnconfigure(0, weight=1)
        self.controls.rowconfigure(0, weight=1)
        gLabel = tk.Label(self.controls, height=int(self.canvas_height/540), width=int(self.canvas_height/30), text="g = " + str(self.currentGValue), bg=BG_COLOR)
        gLabel.grid(column=0, row=1)
        gBarLabel = tk.Label(self.controls, height=int(self.canvas_height/540), width=int(self.canvas_height/30), text="g bar = " + str(self.currentGBarValue), bg=BG_COLOR)
        gBarLabel.grid(column=2, row=1)
        changeLabel = tk.Label(self.controls, height=int(self.canvas_height/540), width=int(self.canvas_height/30), text="Change in g bar = " + str(self.changeGBar), bg=BG_COLOR)
        changeLabel.grid(column=0, row=2)
        averageChangeLabel = tk.Label(self.controls, height=int(self.canvas_height/540), width=int(self.canvas_height/20), text="Average change in g bar = " + str(self.averageChange), bg=BG_COLOR)
        averageChangeLabel.grid(column=2, row=2)
        periodLabel = tk.Label(self.controls, height=int(self.canvas_height/540), width=int(self.canvas_height/30), text="Period of g bar = " + str(self.period_gsb), bg=BG_COLOR)
        periodLabel.grid(column=0, row=3)

        backButton = tk.Button(self.controls, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="back", command = self.disconnectAndReturn, bg=BG_COLOR)
        backButton.grid(column=1, row=4)

    def showDraw(self):
        self.controls.grid_remove()
        self.drawRegion = DrawRegion(self.gui, self.canvas_width, self.canvas_height)
        self.controls = self.drawRegion
        self.controls.grid(column=0, row=0)
        drawButton = tk.Button(self.controls, height=int(self.canvas_height/540), width=int(self.canvas_height/50), text="Draw", command = self.createNewCommand, bg=BG_COLOR)
        drawButton.grid(column=5, row=4)
        backButton = tk.Button(self.controls, height=int(self.canvas_height/540), width=int(self.canvas_height/50), text="Back", command = self.mainMenu, bg=BG_COLOR)
        backButton.grid(column=4, row=4)

    def createNewCommand(self):
        triCount = int(self.drawRegion.getTriCount())
        if triCount < 3 * int(self.drawRegion.getOuterEdgeNo()):
            triCount = 3 * int(self.drawRegion.getOuterEdgeNo()) + 10
        self.fileName = self.drawRegion.getFileName()
        self.fileRoot = self.drawRegion.getFileRoot()
        self.createNew(self.drawRegion.getFreeDraw(), self.drawRegion.getFileRoot(), self.drawRegion.getFileName(), triCount, int(self.drawRegion.getInnerEdgeNo()), int(self.drawRegion.getOuterEdgeNo()), int(self.drawRegion.getInRad()), int(self.drawRegion.getOutRad()), self.drawRegion.getRandomSet())
        self.flags = False
        self.stopFlag = False
        hole_x, hole_y = self.tri.region.points_in_holes[0]
        self.plotPoint(hole_x, hole_y)
        self.plotPoint(hole_x + 1000, hole_y)
        self.slitPathCalculate()
        self.show(first=True)

    def createNew(self, freeDraw, fileRoot, fileName, triCount, inEdgeNum, outEdgeNum, inRad = None, outRad = None, randomOrNot = False):
        if freeDraw:
            if fileRoot != None:
                subprocess.run([
                    'python',
                    'draw_region.py',
                    fileName,
                    fileRoot
                ])
            else:
                subprocess.run([
                    'python',
                    'draw_region.py',
                    fileName
                ])
        else:
            draw_region.draw_region_back(fileName, inEdgeNum, outEdgeNum, inRad, outRad, randomSet = randomOrNot, fileRoot=fileRoot)
        print("drew region")
        subprocess.run([
            'julia',
            'triangulate_via_julia.jl',
            fileName,
            fileRoot,
            fileName,
            str(int(triCount))
        ])
        print("triangulated region")
        if fileRoot != None:
            t = Triangulation.read(f'regions/{fileRoot}/{fileName}/{fileName}.poly')
            t.write(f'regions/{fileRoot}/{fileName}/{fileName}.output.poly')
        else:
            t = Triangulation.read(f'regions/{fileName}/{fileName}.poly')
            t.write(f'regions/{fileName}/{fileName}.output.poly')
        print("made output.poly")
        if fileRoot != None:
            subprocess.run([
                'python',
                'mesh_conversion/mesh_conversion.py',
                '-p',
                f'regions/{fileRoot}/{fileName}/{fileName}.output.poly',
                '-n',
                f'regions/{fileRoot}/{fileName}/{fileName}.node',
                '-e',
                f'regions/{fileRoot}/{fileName}/{fileName}.ele',
            ])
        else:
            subprocess.run([
                'python',
                'mesh_conversion/mesh_conversion.py',
                '-p',
                f'regions/{fileName}/{fileName}.output.poly',
                '-n',
                f'regions/{fileName}/{fileName}.node',
                '-e',
                f'regions/{fileName}/{fileName}.ele',
            ])
        print("converted mesh")
        if fileRoot != None:
            subprocess.run([
                'python',
                'mesh_conversion/fenicsx_solver.py',
                fileName,
                fileRoot
            ])
        else:
            subprocess.run([
                'python',
                'mesh_conversion/fenicsx_solver.py',
                fileName,
                fileRoot
            ])
        print("solved pde")
        if fileRoot != None:
            self.tri = Triangulation.read(f'regions/{fileRoot}/{fileName}/{fileName}.poly')
        else:
            self.tri = Triangulation.read(f'regions/{fileName}/{fileName}.poly')
        print("finished")

    def animationConfig(self):
        self.controls.grid_remove()
        self.controls = self.gifConfig.getFrame(self.gui)
        drawButton = tk.Button(self.controls, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Create", command = self.createAnimation, bg=BG_COLOR)
        backButton = tk.Button(self.controls, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Back", command = self.mainMenu, bg=BG_COLOR)
        drawButton.grid(column=4, row=4)
        backButton.grid(column=5, row=4)

    def showNSave(self, name=None):
        self.flags = False
        self.stopFlag = False
        hole_x, hole_y = self.tri.region.points_in_holes[0]
        self.plotPoint(hole_x, hole_y)
        self.plotPoint(hole_x + 1000, hole_y)
        self.slitPathCalculate()
        self.updateLambdaGraph()
        uniformization = self.calculateUniformization()
        self.showUniformization(uniformization, True)
        self.ax2.axis('off')
        if name == None:
            name = self.fileName
        path = "outputImg/" + name
        self.fig.savefig(path)
        self.canvas.draw()

    def createAnimation(self):
        firstFileName = self.gifConfig.getFileRoot() + "_" + str(self.gifConfig.getInitEdge()) + "_" + str(int(self.gifConfig.getTriCountInit())) + "_0"
        # The following is the steps increasing edge count
        for i in range(self.gifConfig.getFinEdge() - self.gifConfig.getInitEdge()):
            name = self.gifConfig.getFileRoot() + "_" + str(self.gifConfig.getInitEdge()+i) + "_" + str(int(self.gifConfig.getTriCountInit())) + "_0"
            #print(name)
            if int(self.gifConfig.getTriCountInit()) < int(self.gifConfig.getInitEdge()+i):
                triCount = int(self.gifConfig.getInitEdge()+i) * 3 + 10
            else:
                triCount = int(self.gifConfig.getTriCountInit())
            self.createNew(False, 
                           self.gifConfig.getFileRoot(), 
                           name, 
                           int(triCount), 
                           int(self.gifConfig.getInitEdge()+i), 
                           int(self.gifConfig.getInitEdge()+i), 
                           self.gifConfig.getInitInRad(), 
                           self.gifConfig.getOutRad()
                           )
            self.tri = Triangulation.read(f'regions/{self.gifConfig.getFileRoot()}/{name}/{name}.poly')
            self.showNSave(name)
        # The following is the steps refining triangulation
        for i in range(self.gifConfig.getTriCountSteps()):
            triCount = int(int(self.gifConfig.getTriCountInit()) + i * (int(self.gifConfig.getTriCountFinal()) - int(self.gifConfig.getTriCountInit())) / self.gifConfig.getTriCountSteps())
            name = self.gifConfig.getFileRoot() + "_" + str(self.gifConfig.getFinEdge()) + "_" + str(int(triCount)) + "_0"
            #print(name)
            if triCount < int(self.gifConfig.getFinEdge()) * 3:
                triCount = int(self.gifConfig.getFinEdge()) * 3 + 10
            self.createNew(
                False,
                self.gifConfig.getFileRoot(),
                name,
                int(triCount),
                int(self.gifConfig.getFinEdge()), 
                int(self.gifConfig.getFinEdge()), 
                self.gifConfig.getInitInRad(), 
                self.gifConfig.getOutRad()
                )
            self.tri = Triangulation.read(f'regions/{self.gifConfig.getFileRoot()}/{name}/{name}.poly')
            self.showNSave(name)
        # The following is the steps shrinking the inner radius
        for i in range(self.gifConfig.getStepCount()):
            name = self.gifConfig.getFileRoot() + "_" + str(self.gifConfig.getFinEdge()) + "_" + str(int(self.gifConfig.getTriCountFinal())) + "_" + str(i)
            #print(name)
            if int(self.gifConfig.getTriCountFinal()) < int(self.gifConfig.getFinEdge()) * 3:
                triCount = int(self.gifConfig.getFinEdge()) * 3 + 10
            else:
                triCount = int(self.gifConfig.getTriCountFinal())
            self.createNew(
                False,
                self.gifConfig.getFileRoot(),
                name,
                int(triCount), 
                int(self.gifConfig.getFinEdge()), 
                int(self.gifConfig.getFinEdge()), 
                self.gifConfig.getInitInRad() + i * ((self.gifConfig.getFinInRad() - self.gifConfig.getInitInRad()) / self.gifConfig.getStepCount()), 
                self.gifConfig.getOutRad()
                )
            self.tri = Triangulation.read(f'regions/{self.gifConfig.getFileRoot()}/{name}/{name}.poly')
            self.showNSave(name)
        directory = Path('regions/' + self.gifConfig.getFileRoot())
        directory_info = directory / (self.gifConfig.getFileRoot() + "_info.txt")
        if not directory.is_dir():
            directory.mkdir(parents=True, exist_ok=True)
        with open(directory_info, 'w', encoding='utf-8') as file:
            file.write("The lines (starting line 2) in order are: the root name for the sequence, how many steps to get after reaching final edge count, the beginning edge count, the final edge count, the outer radius, the beginning inner radius, the final inner radius, the triangle count at the beginning, the triangle count at the end, the number of steps to get from the beginning to end amount of triangles\n" +
                str(self.gifConfig.getFileRoot()) + "\n" +
                str(self.gifConfig.getStepCount()) + "\n" +
                str(self.gifConfig.getInitEdge()) + "\n" +
                str(self.gifConfig.getFinEdge()) + "\n" +
                str(self.gifConfig.getOutRad()) + "\n" +
                str(self.gifConfig.getInitInRad()) + "\n" +
                str(self.gifConfig.getFinInRad()) + "\n" +
                str(self.gifConfig.getTriCountInit()) + "\n" +
                str(self.gifConfig.getTriCountFinal()) + "\n" +
                str(self.gifConfig.getTriCountSteps())
            )
        self.tri = Triangulation.read(f'regions/{self.gifConfig.getFileRoot()}/{firstFileName}/{firstFileName}.poly')
        self.fileName = firstFileName
        self.fileRoot = self.gifConfig.getFileRoot()
        self.showIntermediate()
            

##################
# GOAL
# Want to let user input: Initial Edge count, final edge count, Outer radius, Initial inner radius, final internal radius, number of steps in between
# After this is implemented I should also see if I can add support for moving the hole around, different internal external edge counts
# So I should create a new class that deals with these inputs, like drawregion and show. This will then allow my program access to these values, so when the button to start begins it can 
# be plugged into a for loop. Definitly should include a warning about how long it will take. 

class GifConfig():

    def __init__(self, height, width):
        self.canvas_height = height
        self.canvas_width = width
        self.initEdge = tk.IntVar()
        self.initEdge.set(3)
        self.finEdge = tk.IntVar()
        self.finEdge.set(12)
        self.outRad = tk.DoubleVar()
        self.initInRad = tk.DoubleVar()
        self.finInRad = tk.DoubleVar()
        self.stepCount = tk.IntVar()
        self.fileRoot = tk.StringVar()
        self.triCountInit = tk.IntVar()
        self.triCountFinal = tk.IntVar()
        self.triCountSteps = tk.IntVar()
        self.controls = None

    def getFrame(self, parent):
        # if self.controls is not None:
        #     return None
        controls = tk.Frame(parent, width=self.canvas_width, height=self.canvas_height, bg=BG_COLOR)
        controls.columnconfigure(0, weight=1)
        controls.rowconfigure(0, weight=1)
        controls.grid(column=0, row=0)

        instructLabel = tk.Label(controls, height=int(self.canvas_height/540), width=int(self.canvas_height/10), text="Select options, then click start to generate a new sequence of figures, WARNING, it takes about 15-30 seconds per step to generate", bg=BG_COLOR)
        instructLabel.grid(column=2, row=0, columnspan=3)

        iEdgeLabel = tk.Label(controls, width=int(self.canvas_height/50), height=int(self.canvas_height/600), text="Starting Edge Count", bg=BG_COLOR)
        iEdgeLabel.grid(column=0, row=1)

        iEdgeEntry = tk.Entry(controls, width=int(self.canvas_height/50), textvariable=self.initEdge, bg=BLACK)
        iEdgeEntry.grid(column=1, row=1)

        fEdgeLabel = tk.Label(controls, width=int(self.canvas_height/50), height=int(self.canvas_height/600), text="Final Edge Count", bg=BG_COLOR)
        fEdgeLabel.grid(column=2, row=1)

        fEdgeEntry = tk.Entry(controls, width=int(self.canvas_height/50), textvariable=self.finEdge, bg=BLACK)
        fEdgeEntry.grid(column=3, row=1)

        outRadiusLabel = tk.Label(controls, width=int(self.canvas_height/50), height=int(self.canvas_height/600), text="Outer Radius", bg=BG_COLOR)
        outRadiusLabel.grid(column=4, row=1)

        outRadiusLabel = tk.Entry(controls, width=int(self.canvas_height/50), textvariable=self.outRad, bg=BLACK)
        outRadiusLabel.grid(column=5, row=1)

        initInRadiusLabel = tk.Label(controls, width=int(self.canvas_height/50), height=int(self.canvas_height/600), text="Initial Inner Radius", bg=BG_COLOR)
        initInRadiusLabel.grid(column=0, row=2)

        initInRadiusEntry = tk.Entry(controls, width=int(self.canvas_height/50), textvariable=self.initInRad, bg=BLACK)
        initInRadiusEntry.grid(column=1, row=2)

        finInRadiusLabel = tk.Label(controls, width=int(self.canvas_height/50), height=int(self.canvas_height/600), text="Final Inner Radius", bg=BG_COLOR)
        finInRadiusLabel.grid(column=2, row=2)

        finInRadiusEntry = tk.Entry(controls, width=int(self.canvas_height/50), textvariable=self.finInRad, bg=BLACK)
        finInRadiusEntry.grid(column=3, row=2)

        stepCountLabel = tk.Label(controls, width=int(self.canvas_height/50), height=int(self.canvas_height/600), text="Number of Steps to shrink Inner Radius", bg=BG_COLOR)
        stepCountLabel.grid(column=4, row=2)

        stepCountEntry = tk.Entry(controls, width=int(self.canvas_height/50), textvariable=self.stepCount, bg=BLACK)
        stepCountEntry.grid(column=5, row=2)

        fileRootLabel = tk.Label(controls, width=int(self.canvas_height/50), height=int(self.canvas_height/600), text="File Root", bg=BG_COLOR)
        fileRootLabel.grid(column=0, row=3)

        fileRootEntry = tk.Entry(controls, width=int(self.canvas_height/50), textvariable=self.fileRoot, bg=BLACK)
        fileRootEntry.grid(column=1, row=3)

        triCountInitLabel = tk.Label(controls, width=int(self.canvas_height/50), height=int(self.canvas_height/600), text="Triangle Count Initial", bg=BG_COLOR)
        triCountInitLabel.grid(column=2, row=3)

        triCountInitEntry = tk.Entry(controls, width=int(self.canvas_height/50), textvariable=self.triCountInit, bg=BLACK)
        triCountInitEntry.grid(column=3, row=3)

        triCountFinLabel = tk.Label(controls, width=int(self.canvas_height/50), height=int(self.canvas_height/600), text="Triangle Count Final", bg=BG_COLOR)
        triCountFinLabel.grid(column=4, row=3)

        triCountFinEntry = tk.Entry(controls, width=int(self.canvas_height/50), textvariable=self.triCountFinal, bg=BLACK)
        triCountFinEntry.grid(column=5, row=3)

        triCountStepsLabel = tk.Label(controls, width=int(self.canvas_height/50), height=int(self.canvas_height/600), text="Triangle Count Steps", bg=BG_COLOR)
        triCountStepsLabel.grid(column=0, row=4)

        triCountStepsEntry = tk.Entry(controls, width=int(self.canvas_height/50), textvariable=self.triCountSteps, bg=BLACK)
        triCountStepsEntry.grid(column=1, row=4)

        self.controls = controls

        return controls

    
    def getInitEdge(self):
        return self.initEdge.get()
    def getFinEdge(self):
        return self.finEdge.get()
    def getOutRad(self):
        return self.outRad.get()
    def getInitInRad(self):
        return self.initInRad.get()
    def getFinInRad(self):
        return self.finInRad.get()
    def getStepCount(self):
        return self.stepCount.get()
    def getFileRoot(self):
        return self.fileRoot.get()
    def getTriCountInit(self):
        return self.triCountInit.get()
    def getTriCountFinal(self):
        return self.triCountFinal.get()
    def getTriCountSteps(self):
        return self.triCountSteps.get()

if __name__ == "__main__":
    a = show_results()
    a.showResults()