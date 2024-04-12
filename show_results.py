from region import Region
from triangulation import (
    Triangulation,
    point_to_right_of_line_compiled,
    polygon_oriented_counterclockwise,
    segment_intersects_segment,
    tri_level_sets
)
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_tools import ToolToggleBase, ToolBase
from cmcrameri import cm
from region import Region
import pyvista
from matplotlib import collections as mc
import numba
import networkx as nx
import tkinter as tk
from sys import argv
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import NavigationToolbar2

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

class show_results:

    def __init__(self):
        if len(argv) > 1:
            file_stem = argv[1]
        else:
            file_stem = "non_concentric_annulus"

        path = Path(f'regions/{file_stem}/{file_stem}')
        self.tri = Triangulation.read(f'regions/{file_stem}/{file_stem}.poly')

        self.gui, self.controls, self.canvas_width, self.canvas_height = self.basicGui()
        self.fig, self.axes, self.graphHolder, self.canvas, self.toolbar, self.graphHolder, self.callbackName = self.basicTkinter()
        self.matCanvas = self.canvas.get_tk_widget()
        self.matCanvas.pack()
        self.show_vertices_tri = tk.BooleanVar()
        self.show_edges_tri=tk.BooleanVar()
        self.show_triangles_tri=tk.BooleanVar()
        self.show_vertex_indices_tri=tk.BooleanVar()
        self.show_triangle_indices_tri=tk.BooleanVar()
        self.show_level_curves_tri=tk.BooleanVar()
        self.show_singular_level_curves_tri=tk.BooleanVar()
        self.highlight_vertices_tri=True
        self.highlight_edges_tri=True
        self.highlight_triangles_tri=True
        self.face_color_tri=True
        self.highlight_triangles_color_tri=True
        self.num_level_curves_tri=True
        self.line_width_tri=True
        self.show_vertex_indices_vor=tk.BooleanVar()
        self.show_polygon_indices_vor=tk.BooleanVar()
        self.show_vertices_vor=tk.BooleanVar()
        self.show_edges_vor=tk.BooleanVar()
        self.show_polygons_vor=tk.BooleanVar()
        self.show_region_vor=tk.BooleanVar()
        self.highlight_vertices_vor=True
        self.highlight_edges_vor=True
        self.highlight_polygons_vor=True
        self.highlight_edges_color_vor=True
        self.highlight_vertices_color_vor=True
        self.highlight_polygons_color_vor=True

        
    def basicGui(self):
        gui = tk.Tk() # initialized Tk
        gui.state('zoomed')
        gui['bg'] = BG_COLOR # sets the background color to that grey
        gui.title("Manipulate data") 
        gui.columnconfigure(0, weight=1)
        gui.rowconfigure(0, weight=1)
        print(gui.winfo_height(), gui.winfo_width())
        canvas_width = gui.winfo_width() 
        canvas_height = gui.winfo_height() # this and above set height and width variables that fill the screen
        print(canvas_height, canvas_width)
        controls = tk.Frame(gui, width=canvas_width, height=canvas_height/2, relief="ridge", bg=BLUE)
        controls.columnconfigure(0, weight=1)
        controls.rowconfigure(0, weight=1)
        controls.grid(column=0, row=0)
        text = tk.Label(controls, height=int(canvas_height/224), width=int(canvas_height/14), text="Click a point inside the hole, then click a point outside the graph to choose the line.\n Press \" Compute Slit Path\" to calculate the intersecting cells")
        text.grid(column=0, row=0)
        button1 = tk.Button(controls, height=int(canvas_height/224), width=int(canvas_height/14), text="Compute Slit Path", command=lambda: self.showSlitPath())
        button1.grid(column=0, row=1)
        return gui, controls, canvas_width, canvas_height
    
    def dummy():
        print("Hello")
    
    def basicTkinter(self):
        fig, axes = plt.subplots()
        axes = plt.gca()
        fig.set_figheight(3)
        fig.set_figwidth(4)
        graphHolder = tk.Frame(self.gui, width=self.canvas_width, height=self.canvas_height , relief="ridge")
        graphHolder.grid(column=0, row=1)
        canvas = FigureCanvasTkAgg(fig, master = graphHolder)   
        toolbar = NavigationToolbar2Tk(canvas, graphHolder)
        toolbar.update()
        callbackName = fig.canvas.callbacks.connect('button_press_event', self.callback)
        return fig, axes, graphHolder, canvas, toolbar, graphHolder, callbackName
    
    def callback(self,event):
        if (self.fig.canvas.toolbar.__getstate__()['mode'] != ''):
            print(self.fig.canvas.toolbar.__getstate__()['mode'])
            return
        x = event.xdata
        y = event.ydata
        global flags
        global stopFlag
        print(x,y)
        if (not stopFlag):
            if (flags):
                self.base_cell = self.determinePolygon(x, y)
                print(self.base_cell)
                self.base_point = self.tri.vertices[self.tri.contained_to_original_index[self.base_cell]]
                print(self.base_point)

            else:
                self.pointInHole = [x, y]
                plt.plot(x,y, 'bo', markersize = 2)
                plt.draw()

        if (not stopFlag):
            if (flags):
                plt.plot([self.base_point[0],self.pointInHole[0]],[self.base_point[1],self.pointInHole[1]],'bo-', markersize = 2, linewidth = 1)
                plt.draw()
                stopFlag = True
                print("yippee!")

        flags = True

    def showResults(self):

        self.tri.show(
            show_edges=False,
            show_triangles=False,
            fig=self.fig,
            axes=self.axes
        )
        self.tri.show_voronoi_tesselation(
            show_vertex_indices=False,
            show_polygons=True,
            show_edges= True,
            fig=self.fig,
            axes=self.axes
        )
        #plt.show() 
        polygon_coordinates = [
                np.array(
                    list(map(lambda x: self.tri.circumcenters[x], polygon))
                )
                for polygon in self.tri.contained_polygons
            ]
        barycenters = np.array(list(map(
            lambda x: np.mean(x, axis=0),
            polygon_coordinates
        )))
        for i in range(len(barycenters)):
            plt.text(
                barycenters[i, 0],
                barycenters[i, 1],
                str(i),
                fontsize=6,
                weight='bold',
                zorder=7
            )
        self.canvas.draw() 

        global flags
        flags = False
        global stopFlag
        stopFlag = False

        tk.mainloop()

        return 

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
    def segment_intersects_line(self,tail, head):
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

    def showSlitPathCalculate(self):
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
        global contained_topology
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
        self.connected_component = []
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
                            self.connected_component.append(edge) # add the edge as a connected component
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
        triangulation_edges_reindexed = self.tri.original_to_contained_index[self.tri.triangulation_edges]
        contained_edges = []
        for edge in triangulation_edges_reindexed:
            if -1 not in edge:
                contained_edges.append(list(edge))
        # I think this just creates a list of all edges that don't have a vertex on a boundary

        self.tri.show(
            show_edges=False,
            show_triangles=False,
            fig=self.fig,
            axes=self.axes
        )
        self.tri.show_voronoi_tesselation(
            show_vertex_indices=False,
            show_polygons=True,
            show_polygon_indices=False,
            show_edges=True,
            highlight_polygons=self.cell_path,
            highlight_vertices=list(slit_cell_vertices),
            fig=self.fig,
            axes=self.axes
        )
        self.canvas.draw()

    def jargon(self):
        # Choose omega_0 as the slit vertex that has the smallest angle relative to the line from the point in hole through
        # the circumcenter of the base_cell
        slit_path = [edge[0] for edge in self.connected_component] # the slit path is the sequence of edges from inside to out
        slit_path.append(self.connected_component[-1][1]) # adds the final edge
        # Connected component goes from outer boundary to inner boundary. Reverse after making slit
        slit_path = list(reversed(slit_path))
        angles = np.array([ # builds an array of angles between the circumcenter and line
            np.arctan2(
                self.tri.circumcenters[vertex][1] - self.pointInHole[0],
                self.tri.circumcenters[vertex][0] - self.pointInHole[1]
            )
            for vertex in slit_path
        ])
        omega_0 = slit_path[np.argmin(angles)] # makes omega_0 the minimum of this
        # Create graph of circumcenters (Lambda[0])
        lambda_graph = nx.Graph() # creates empty graph
        lambda_graph.add_nodes_from(range(len(self.tri.circumcenters))) # adds all circumcenters, not really maintaining structure just adding a node for each
        lambda_graph.add_edges_from(self.tri.voronoi_edges) # adds all edges connecting these nodes
        nx.set_edge_attributes(lambda_graph, values=1, name='weight') # sets all edges to have a value of 1
        for edge in self.edges_to_weight: # Sets every edge that intersects the line to have effectivly infinite weight
            lambda_graph.edges[edge[0], edge[1]]['weight'] = np.finfo(np.float32).max
        shortest_paths = nx.single_source_dijkstra(lambda_graph, omega_0, target=None, cutoff=None, weight='weight')[1] # finds the shortest path around the figure to every node in the figure in a MASSIVE dictionary

        global num_contained_polygons
        num_contained_polygons = len(self.tri.contained_polygons)
        g_star_bar = np.zeros(self.tri.num_triangles, dtype=np.float64) # creates a vector for each triangle
        perpendicular_edges_dict = {}
        for omega in range(self.tri.num_triangles): # Loops over each triangle
            edges = self.build_path_edges(shortest_paths[omega]) # Takes in a list of verticies (circumcenters) connecting omega_0 to the node, and builds an edge path
            flux_contributing_edges = []
            for edge in edges:
                flux_contributing_edges.append(tuple(self.get_perpendicular_edge(edge))) # This creates a sequence of verticies (triangle verticies) connecting omega_0 to the desired end vertex
            perpendicular_edges_dict[omega] = flux_contributing_edges # adds this (triangle vertex0) path to the dictionary 
            g_star_bar[omega] = self.flux_on_contributing_edges(flux_contributing_edges) # adds the flux for this path to whatever the g_star_bar vector is

        # Interpolate the value of pde_solution to get its values on the omegas
        pde_on_omega_values = [np.mean(self.tri.pde_values[self.tri.triangles[i]]) for i in range(self.tri.num_triangles)] # takes in the solution on the triangle vertices and gives a solution on the circumcenter/node
        period_gsb = np.max(g_star_bar) # the maximum value in the vector is stored, so i guess the largest flux
        # TODO: allow the last edge so we get all the
        global uniformization
        uniformization = np.exp(2 * np.pi / period_gsb * (pde_on_omega_values + 1j * g_star_bar)) # unused, and incomprehensible without the mathematical contex

        # Level curves for gsb
        global g_star_bar_interpolated_interior
        g_star_bar_interpolated_interior = np.array([np.mean(g_star_bar[poly]) for poly in self.tri.contained_polygons]) # vector of the average fluxes for each contained cell
        min_, max_ = np.min(g_star_bar_interpolated_interior), np.max(g_star_bar_interpolated_interior) # saves the maximum and minimum average flux for each contained cell
        global heights
        heights = np.linspace(min_, max_, num=100) # creates an array of 100 evenely spaced numbers between the max and minimum flux for the contained polygons

        contained_triangle_indicator = np.all(self.tri.vertex_boundary_markers[self.tri.triangles] == 0, axis=1)
        global slit_cell_vertices
        global contained_triangles
        contained_triangles = np.where(contained_triangle_indicator)[0]
        slit_cell_vertices = set(self.flatten_list_of_lists([self.tri.contained_polygons[cell] for cell in self.cell_path]))
        global contained_triangle_minus_slit
        contained_triangle_minus_slit = list(set(contained_triangles).difference(slit_cell_vertices))

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
    def build_path_edges(vertices): # specifically this takes in a list of verticies, and creates a list of edges connecting the verticies in order that they are stored
        edges = []
        for i in range(len(vertices) - 1):
            edge = [
                vertices[i],
                vertices[i + 1]
            ]
            edges.append(edge)
        return edges

    # Make mapping from edges on the cells to the perpendicular edges in triangulation
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

    def what2(self):
        self.tri.show(
            show_triangle_indices=True,
            highlight_triangles=contained_triangle_minus_slit
        )
        self.canvas.draw()

    def what3(self):
        level_set = []
        for i in range(len(contained_triangle_minus_slit)):
            triangle = self.tri.triangles[contained_triangle_minus_slit[i]]
            level_set_triangle = tri_level_sets(
                self.tri.vertices[triangle],
                g_star_bar_interpolated_interior[self.tri.original_to_contained_index[triangle]],
                heights
            )
            level_set.append(level_set_triangle)

        level_set_flattened = self.flatten_list_of_lists(level_set)
        level_set_filtered = [
            line_segment for line_segment in level_set_flattened if len(line_segment) > 0
        ]
        lines = [
            [
                tuple(line_segment[0]),
                tuple(line_segment[1])
            ] for line_segment in level_set_filtered
        ]
        line_collection = mc.LineCollection(lines, linewidths=1)
        line_collection.set(color=[1, 0, 0])
        self.tri.show(
            highlight_triangles=contained_triangle_minus_slit
        )
        axes = plt.gca()
        axes.add_collection(line_collection)
        self.canvas.draw() 

    def what4(self):
        self.tri.show(
            'test.png',
            show_edges=True,
            show_triangles=False
        )
        self.tri.show_voronoi_tesselation(
            'test.png',
            show_edges=True,
            show_polygons=False,
            fig=plt.gcf(),
            axes=plt.gca()
        )
        self.canvas.draw() 

    def what5(self):
        # Conjugate level curves with color
        def subsample_color_map(colormap, num_samples, start_color=0, end_color=255, reverse=False):
            sample_points_float = np.linspace(start_color, end_color, num_samples)
            sample_points = np.floor(sample_points_float).astype(np.int64)
            all_colors = colormap.colors
            if reverse:
                all_colors = np.flip(all_colors, axis=0)
            return all_colors[sample_points]

        # graded_level_curve_color_map = cm.lajolla
        conjugate_level_curve_color_map = cm.buda
        conjugate_level_curve_colors = subsample_color_map(
            conjugate_level_curve_color_map,
            len(heights),
            start_color=32,
            end_color=223,
            reverse=True
        )
        for height_index, height in enumerate(heights):
            level_set = []
            for i in range(len(contained_triangle_minus_slit)):
                triangle = self.tri.triangles[contained_triangle_minus_slit[i]]
                level_set_triangle = tri_level_sets(
                    self.tri.vertices[triangle],
                    g_star_bar_interpolated_interior[self.tri.original_to_contained_index[triangle]],
                    [height]
                )
                level_set.append(level_set_triangle)

            level_set_flattened = self.flatten_list_of_lists(level_set)
            level_set_filtered = [
                line_segment for line_segment in level_set_flattened if len(line_segment) > 0
            ]
            lines = [
                [
                    tuple(line_segment[0]),
                    tuple(line_segment[1])
                ] for line_segment in level_set_filtered
            ]
            line_collection = mc.LineCollection(lines, linewidths=1)
            line_collection.set(color=conjugate_level_curve_colors[height_index])
            self.axes.add_collection(line_collection)
        self.tri.show(
            show_level_curves=True,
            fig=self.fig,
            axes=self.axes
        )
        self.axes.add_collection(line_collection)
        self.canvas.draw()      

    def determinePolygon(self, x, y):
        polygon_coordinates = [
                np.array(
                    list(map(lambda x: self.tri.circumcenters[x], polygon))
                )
                for polygon in self.tri.contained_polygons
            ]
        barycenters = np.array(list(map(
            lambda x: np.mean(x, axis=0),
            polygon_coordinates
        )))
        distanceToBary = np.array([ # builds an array of angles between the circumcenter and line
            (bary[0] - x)**2 + (bary[1] - y)**2
            for bary in barycenters
        ])
        return np.argmin(distanceToBary)
    
    def show(self):
        self.fig.clear()
        self.axes.clear()
        self.fig, self.axes = plt.subplots()
        self.axes = plt.gca()
        self.fig.set_figheight(3)
        self.fig.set_figwidth(4)
        self.graphHolder.destroy()
        self.graphHolder = tk.Frame(self.gui, width=self.canvas_width, height=self.canvas_height , relief="ridge")
        self.graphHolder.grid(column=0, row=1)
        self.canvas = FigureCanvasTkAgg(self.fig, master = self.graphHolder)   
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.graphHolder)
        self.toolbar.update()
        #print(self.graphHolder.children)
        self.matCanvas = self.canvas.get_tk_widget()
        self.matCanvas.pack()

        self.tri.show(
            show_vertices=self.show_vertices_tri.get(),
            show_edges=self.show_edges_tri.get(),
            show_triangles=self.show_triangles_tri.get(),
            show_vertex_indices=self.show_vertex_indices_tri.get(),
            show_triangle_indices=self.show_triangle_indices_tri.get(),
            show_level_curves=self.show_level_curves_tri.get(),
            #show_singular_level_curves=self.show_singular_level_curves_tri,
            #highlight_vertices=self.highlight_vertices_tri,
            #highlight_edges=self.highlight_edges_tri,
            #highlight_triangles=self.highlight_triangles_tri,
            #face_color=self.face_color_tri,
            #highlight_triangles_color=self.highlight_triangles_color_tri,
            #num_level_curves=self.num_level_curves_tri,
            #line_width=self.line_width_tri,
            fig=self.fig,
            axes=self.axes
        )
        self.tri.show_voronoi_tesselation(
            show_vertex_indices=self.show_vertex_indices_vor.get(),
            show_polygon_indices=self.show_polygon_indices_vor.get(),
            show_vertices=self.show_vertices_vor.get(),
            show_edges=self.show_edges_vor.get(),
            show_polygons=self.show_polygons_vor.get(),
            show_region=self.show_region_vor.get(),
            #highlight_vertices=self.highlight_vertices_vor,
            #highlight_edges=self.highlight_edges_vor,
            #highlight_polygons=self.highlight_polygons_vor,
            #highlight_edges_color=self.highlight_edges_color_vor,
            #highlight_vertices_color=self.highlight_vertices_color_vor,
            #highlight_polygons_color=self.highlight_polygons_color_vor,
            fig=self.fig,
            axes=self.axes
        )
        self.canvas.draw()

    def showSlit(self):
        slit_cell_vertices = set(self.flatten_list_of_lists([self.tri.contained_polygons[cell] for cell in self.cell_path]))
        self.tri.show_voronoi_tesselation(
            highlight_polygons=self.cell_path,
            highlight_vertices=list(slit_cell_vertices),
            fig=self.fig,
            axes=self.axes
        )
        self.canvas.draw()

    def mainMenu(self):
        self.controls.grid_remove()
        mainMenu = tk.Frame(self.gui, width=self.canvas_width, height=self.canvas_height)
        mainMenu.columnconfigure(0, weight=1)
        mainMenu.rowconfigure(0, weight=1)
        mainMenu.grid(column=0, row=0)

        graphButton = tk.Button(mainMenu, height=int(self.canvas_height/540), width=int(self.canvas_height/10), text="Graph Display Options", command = self.graphConfig)
        graphButton.grid(column=0, row=0)

        fluxButton = tk.Button(mainMenu, height=int(self.canvas_height/540), width=int(self.canvas_height/10), text="Edit Flux", command = self.fluxConfig)
        fluxButton.grid(column=0, row=1)

        self.controls = mainMenu

    def showSlitPath(self):
        self.showSlitPathCalculate()
        self.mainMenu()

    def nearestEdge(self, x, y):
        distanceToMidPoints = np.array([ # builds an array of angles between the circumcenter and line
            ((((self.tri.circumcenters[edge[0]][0] + self.tri.circumcenters[edge[1]][0]) * .5) - x)**2 +
            (((self.tri.circumcenters[edge[0]][1] + self.tri.circumcenters[edge[1]][1]) * .5) - y)**2)
            for edge in self.tri.voronoi_edges
        ])

        index = np.argmin(distanceToMidPoints)

        plt.plot(x,y, 'bo', markersize = 2)
        plt.plot(self.tri.circumcenters[self.tri.voronoi_edges[index][0]][0], self.tri.circumcenters[self.tri.voronoi_edges[index][0]][1], 'bo', markersize = 2)
        plt.plot(self.tri.circumcenters[self.tri.voronoi_edges[index][1]][0], self.tri.circumcenters[self.tri.voronoi_edges[index][1]][1], 'bo', markersize = 2)
        plt.draw()

    def fluxFinder(self, event):
        #edge_coordinates = [
        #        np.array(
        #            list(map(lambda x: self.tri.voronoi_edges[x], polygon))
        #        )
        #    ]
        x = event.xdata
        y = event.ydata        
        self.nearestEdge(x, y)

    def fluxConfig(self):
        self.controls.grid_remove()
        self.fig.canvas.callbacks.disconnect(self.callbackName)
        self.callbackName = self.fig.canvas.callbacks.connect('button_press_event', self.fluxFinder)


    
    def graphConfig(self):
        self.controls.grid_remove()
        graphConfigs = tk.Frame(self.gui, width=self.canvas_width, height=self.canvas_height)
        graphConfigs.columnconfigure(0, weight=1)
        graphConfigs.rowconfigure(0, weight=1)
        graphConfigs.grid(column=0, row=0)

        checkButtonTri1 = tk.Checkbutton(graphConfigs, height=int(self.canvas_height/540), width=int(self.canvas_height/50), text="Show Vertices Tri", variable=self.show_vertices_tri)
        checkButtonTri1.grid(column=0, row=0)
        checkButtonTri2 = tk.Checkbutton(graphConfigs, height=int(self.canvas_height/540), width=int(self.canvas_height/50), text="Show Edges Tri", variable=self.show_edges_tri)
        checkButtonTri2.grid(column=1, row=0)
        checkButtonTri3 = tk.Checkbutton(graphConfigs, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Triangles Tri", variable=self.show_triangles_tri)
        checkButtonTri3.grid(column=2, row=0)
        checkButtonTri4 = tk.Checkbutton(graphConfigs, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Vertex Indices Tri", variable=self.show_vertex_indices_tri)
        checkButtonTri4.grid(column=3, row=0)
        checkButtonTri5 = tk.Checkbutton(graphConfigs, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Triangle Indices Tri", variable=self.show_triangle_indices_tri)
        checkButtonTri5.grid(column=4, row=0)
        checkButtonTri6 = tk.Checkbutton(graphConfigs, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Level Curves Tri", variable=self.show_level_curves_tri)
        checkButtonTri6.grid(column=5, row=0)
        checkButtonTri7 = tk.Checkbutton(graphConfigs, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Singular Level Curves Tri", variable=self.show_singular_level_curves_tri)
        checkButtonTri7.grid(column=6, row=0)
        checkButtonVor1 = tk.Checkbutton(graphConfigs, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Vertex Indices Vor", variable=self.show_vertex_indices_vor)
        checkButtonVor1.grid(column=0, row=1)
        checkButtonVor2 = tk.Checkbutton(graphConfigs, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Polygon Indices Vor", variable=self.show_polygon_indices_vor)
        checkButtonVor2.grid(column=1, row=1)
        checkButtonVor3 = tk.Checkbutton(graphConfigs, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Vertices Vor", variable=self.show_vertices_vor)
        checkButtonVor3.grid(column=2, row=1)
        checkButtonVor4 = tk.Checkbutton(graphConfigs, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Edges Vor", variable=self.show_edges_vor)
        checkButtonVor4.grid(column=3, row=1)
        checkButtonVor5 = tk.Checkbutton(graphConfigs, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Polygons Vor", variable=self.show_polygons_vor)
        checkButtonVor5.grid(column=4, row=1)
        checkButtonVor6 = tk.Checkbutton(graphConfigs, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Show Region Vor", variable=self.show_region_vor)
        checkButtonVor6.grid(column=5, row=1)
        drawButton = tk.Button(graphConfigs, height=int(self.canvas_height/540), width=int(self.canvas_height/40), text="Display Graph", command = self.show)
        drawButton.grid(column=6, row=1)
        slitButton = tk.Button(graphConfigs, height=int(self.canvas_height/540), width=int(self.canvas_height/10), text="Show Slit", command = self.showSlit)
        slitButton.grid(column=0, row=2)
        backButton = tk.Button(graphConfigs, height=int(self.canvas_height/540), width=int(self.canvas_height/10), text="Back", command = self.mainMenu)
        backButton.grid(column=0, row=3)

        self.controls = graphConfigs
        
if __name__ == "__main__":
    a = show_results()
    a.showResults()