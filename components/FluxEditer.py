import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from components.BGColors import BGColors

class FluxEditor(tk.Frame):
    def __init__(self, root, lambdaGraph, tri, fig, exitCommand, show, callback, height, width):
        super().__init__(root, width = width, height = height, bg=BGColors.BG_COLOR.value)
        self.gui = root
        self.lambdaGraph = lambdaGraph
        self.tri = tri
        self.fig = fig
        self.canvas_height = height
        self.canvas_width = width
        self.show = show
        self.callbackName = callback

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.grid(column=0, row=0, columnspan=5)
        instructions = tk.Label(self, height=int(self.canvas_height/540), width=int(self.canvas_width/15), text="Click on an edge to edit it's flux. Press enter to set value.", bg=BGColors.BG_COLOR.value)
        instructions.grid(column=0, row=0, columnspan=5)
        backButton = tk.Button(self, height=int(self.canvas_height/540), width=int(self.canvas_width/30), text="Back", command = exitCommand, bg=BGColors.BG_COLOR.value)
        backButton.grid(column=0, row=1, columnspan=5)

        self.fig.canvas.callbacks.disconnect(self.callbackName)
        # and adds a new click that finds nearest edge in the voronai graph
        self.callbackName = self.fig.canvas.callbacks.connect('button_press_event', self.fluxFinder)

    def fluxFinder(self, event):
        if (self.fig.canvas.toolbar.mode != ''):
            #print(self.fig.canvas.toolbar.mode)
            return
        #self.updateLambdaGraph()
        x = event.xdata
        y = event.ydata        
        if x is None or y is None:
            return
        # finds index of the edge closest to the mouse click
        selectedIndex = self.nearestEdge(x, y)

        # adds a entry and button to input user data to the flux graph, and places them at a point in the middle of the graph
        editor = tk.Frame(self.gui, height = int(self.canvas_height/50), width=int(self.canvas_width/70), bg=BGColors.BG_COLOR.value)
        fluxValue = tk.StringVar()
        reg = self.gui.register(self.validateText)
        currentFlux = self.lambdaGraph.edges[self.tri.voronoi_edges[selectedIndex][0], self.tri.voronoi_edges[selectedIndex][1]]['weight']
        fluxValue.set(str(currentFlux))
        fluxInput = tk.Entry(editor, width=int(self.canvas_width/70), bg=BGColors.BLACK.value, validate='key', validatecommand= (reg, '%P', '%i'), textvariable = fluxValue)
        fluxInput.grid(column=0, row=0)
        sendButton = tk.Button(editor, height=1, width=1, bg=BGColors.BG_COLOR.value, command= lambda: self.editFluxGraph(editor, selectedIndex))
        sendButton.grid(column=1, row=0)
        editor.place(x=int(self.canvas_width / 2), y=int(self.canvas_height/2))
        # disables back button until data is entered
        self.children['!button']['state'] = 'disabled'
        # disables clicking entirely
        self.fig.canvas.callbacks.disconnect(self.callbackName)

    def editFluxGraph(self, editor, selectedIndex):
        if editor.children['!entry'] is None:
            return
        if editor.children['!entry'].get() != '':
            #print('a', self.editor.children['!entry'].get(), 'b')
            # edits edge flux in lambda graph
            self.lambdaGraph.edges[self.tri.voronoi_edges[selectedIndex][0], self.tri.voronoi_edges[selectedIndex][1]]['weight'] = float(editor.children['!entry'].get())
            # removes popup, and connects call back back to the edge finder, renables back button
            editor.destroy()
            editor = None
            #self.callbackName = self.fig.canvas.callbacks.connect('button_press_event', self.fluxFinder)
            self.children['!button']['state'] = 'normal'
            self.show()
            #self.fig.canvas.callbacks.disconnect(self.callbackName)
            self.callbackName = self.fig.canvas.callbacks.connect('button_press_event', self.fluxFinder)

    def nearestEdge(self, x, y):
        distanceToMidPoints = np.array([ # builds an array of distance between click and edge midpoints
            ((((self.tri.circumcenters[edge[0]][0] + self.tri.circumcenters[edge[1]][0]) * .5) - x)**2 +
            (((self.tri.circumcenters[edge[0]][1] + self.tri.circumcenters[edge[1]][1]) * .5) - y)**2)
            for edge in self.tri.voronoi_edges
        ])

        # finds the smallest distance
        index = np.argmin(distanceToMidPoints)
        plt.plot(x,y, "ro", markersize = 2)
        plt.plot(self.tri.circumcenters[self.tri.voronoi_edges[index][0]][0], self.tri.circumcenters[self.tri.voronoi_edges[index][0]][1], "ro", markersize = 2)
        plt.plot(self.tri.circumcenters[self.tri.voronoi_edges[index][1]][0], self.tri.circumcenters[self.tri.voronoi_edges[index][1]][1], "ro", markersize = 2)
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
    
    def getLambda(self):
        return self.lambdaGraph

