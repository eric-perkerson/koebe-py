import tkinter as tk
from BGColors import BGColors

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
        self.show_g_bar_level_curves = tk.BooleanVar()
        self.show_g_bar_level_curves.set(False)

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
        return self.show_vertices_tri.get(), self.show_edges_tri.get(), self.show_triangles_tri.get(), self.show_vertex_indices_tri.get(), self.show_triangle_indices_tri.get(), self.show_level_curves_tri.get(), self.show_singular_level_curves_tri.get(), self.show_g_bar_level_curves.get()
    
    def getSlit(self):
        return self.showSlitBool.get()
    
    def setSlit(self, bool):
        self.showSlitBool.set(bool)
    
    def getFrame(self, parent):
        super().__init__(parent, width = self.canvas_width, height = self.canvas_height, bg=BGColors.BG_COLOR.value)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.grid(column=0, row=0)
        checkButtonTri1 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_width/60), text="Show Vertices Tri", variable=self.show_vertices_tri, bg=BGColors.BG_COLOR.value)
        checkButtonTri1.grid(column=0, row=0)
        checkButtonTri2 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_width/60), text="Show Edges Tri", variable=self.show_edges_tri, bg=BGColors.BG_COLOR.value)
        checkButtonTri2.grid(column=1, row=0)
        checkButtonTri3 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_width/80), text="Show Triangles Tri", variable=self.show_triangles_tri, bg=BGColors.BG_COLOR.value)
        checkButtonTri3.grid(column=2, row=0)
        checkButtonTri4 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_width/70), text="Show Vertex Indices Tri", variable=self.show_vertex_indices_tri, bg=BGColors.BG_COLOR.value)
        checkButtonTri4.grid(column=3, row=0)
        checkButtonTri5 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_width/60), text="Show Triangle Indices Tri", variable=self.show_triangle_indices_tri, bg=BGColors.BG_COLOR.value)
        checkButtonTri5.grid(column=4, row=0)
        checkButtonTri6 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_width/70), text="Show Level Curves Tri", variable=self.show_level_curves_tri, bg=BGColors.BG_COLOR.value)
        checkButtonTri6.grid(column=5, row=0)
        checkButtonTri7 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_width/60), text="Show Singular Level Curves Tri", variable=self.show_singular_level_curves_tri, bg=BGColors.BG_COLOR.value)
        checkButtonTri7.grid(column=6, row=0)
        checkButtonVor1 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_width/60), text="Show Vertex Indices Vor", variable=self.show_vertex_indices_vor, bg=BGColors.BG_COLOR.value)
        checkButtonVor1.grid(column=0, row=1)
        checkButtonVor2 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_width/60), text="Show Polygon Indices Vor", variable=self.show_polygon_indices_vor, bg=BGColors.BG_COLOR.value)
        checkButtonVor2.grid(column=1, row=1)
        checkButtonVor3 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_width/80), text="Show Vertices Vor", variable=self.show_vertices_vor, bg=BGColors.BG_COLOR.value)
        checkButtonVor3.grid(column=2, row=1)
        checkButtonVor4 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_width/70), text="Show Edges Vor", variable=self.show_edges_vor, bg=BGColors.BG_COLOR.value)
        checkButtonVor4.grid(column=3, row=1)
        checkButtonVor5 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_width/70), text="Show Polygons Vor", variable=self.show_polygons_vor, bg=BGColors.BG_COLOR.value)
        checkButtonVor5.grid(column=4, row=1)
        checkButtonVor6 = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_width/70), text="Show Region Vor", variable=self.show_region_vor, bg=BGColors.BG_COLOR.value)
        checkButtonVor6.grid(column=5, row=1)
        slitButton = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_width/60), text="Show Slit", variable=self.showSlitBool, bg=BGColors.BG_COLOR.value)
        slitButton.grid(column=6, row=1)
        gBarButton = tk.Checkbutton(self, height=int(self.canvas_height/540), width=int(self.canvas_width/60), text="Show g Bar level curves", variable=self.show_g_bar_level_curves, bg=BGColors.BG_COLOR.value)
        gBarButton.grid(column=0, row=2)
        return self