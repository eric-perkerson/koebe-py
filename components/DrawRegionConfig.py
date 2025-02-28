import tkinter as tk
from components.BGColors import BGColors

class DrawRegionConfig(tk.Frame):
    def __init__(self, parent, width, height):
        super().__init__(parent, width = width, height = height, bg=BGColors.BG_COLOR.value)
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

        instructLabel = tk.Label(self, height=int(self.canvas_height/540), width=int(self.canvas_width/15), text="Select option, then click calcultate to generate a new figure", bg=BGColors.BG_COLOR.value)
        instructLabel.grid(column=2, row=0, columnspan=3)

        radiusOneLabel = tk.Label(self, width=int(self.canvas_width/80), height=int(self.canvas_height/600), text="Outer Radius", bg=BGColors.BG_COLOR.value)
        radiusOneLabel.grid(column=0, row=1)

        radiusOneEntry = tk.Entry(self, width=int(self.canvas_width/80), textvariable=self.outRad, bg=BGColors.BLACK.value)
        radiusOneEntry.grid(column=1, row=1)

        radiusTwoLabel = tk.Label(self, width=int(self.canvas_height/50), height=int(self.canvas_height/600), text="Inner Radius", bg=BGColors.BG_COLOR.value)
        radiusTwoLabel.grid(column=2, row=1)

        radiusTwoEntry = tk.Entry(self, width=int(self.canvas_width/80), textvariable=self.inRad, bg=BGColors.BLACK.value)
        radiusTwoEntry.grid(column=3, row=1)

        fileRootLabel = tk.Label(self, width=int(self.canvas_width/80), height=int(self.canvas_height/600), text="File Root", bg=BGColors.BG_COLOR.value)
        fileRootLabel.grid(column=4, row=1)

        fileRootEntry = tk.Entry(self, width=int(self.canvas_width/80), textvariable=self.fileRoot, bg=BGColors.BLACK.value)
        fileRootEntry.grid(column=5, row=1)

        outEdgeLabel = tk.Label(self, width=int(self.canvas_width/80), height=int(self.canvas_height/600), text="Outer Number of Edges", bg=BGColors.BG_COLOR.value)
        outEdgeLabel.grid(column=0, row=2)

        outEdgeEntry = tk.Entry(self, width=int(self.canvas_width/80), textvariable=self.outEdgeNum, bg=BGColors.BLACK.value)
        outEdgeEntry.grid(column=1, row=2)

        inEdgeLabel = tk.Label(self, width=int(self.canvas_width/80), height=int(self.canvas_height/600), text="Inner Number of Edges", bg=BGColors.BG_COLOR.value)
        inEdgeLabel.grid(column=2, row=2)

        inEdgeEntry = tk.Entry(self, width=int(self.canvas_width/80), textvariable=self.inEdgeNum, bg=BGColors.BLACK.value)
        inEdgeEntry.grid(column=3, row=2)

        fileNameLabel = tk.Label(self, width=int(self.canvas_width/80), height=int(self.canvas_height/600), text="File Name", bg=BGColors.BG_COLOR.value)
        fileNameLabel.grid(column=4, row=2)

        fileNameEntry = tk.Entry(self, width=int(self.canvas_width/80), textvariable=self.fileName, bg=BGColors.BLACK.value)
        fileNameEntry.grid(column=5, row=2)

        TriangleNumLabel = tk.Label(self, width=int(self.canvas_width/80), height=int(self.canvas_height/600), text="Number of Triangles", bg=BGColors.BG_COLOR.value)
        TriangleNumLabel.grid(column=0, row=3)

        reg = self.register(self.isNumber)
        TriangleNumEntry = tk.Entry(self, width=int(self.canvas_width/80), textvariable=self.triCount, validate='key', validatecommand= (reg, '%P', '%i'), bg=BGColors.BLACK.value)
        TriangleNumEntry.grid(column=1, row=3)

        freeDrawButton = tk.Checkbutton(self, height=int(self.canvas_height/600), width=int(self.canvas_width/80), text="Free Draw", variable=self.freeDraw, bg=BGColors.BG_COLOR.value)
        freeDrawButton.grid(column=2, row=3)

        randomButton = tk.Checkbutton(self, height=int(self.canvas_height/600), width=int(self.canvas_width/80), text="Randomize Vertices", variable=self.randomSet, bg=BGColors.BG_COLOR.value)
        randomButton.grid(column=3, row=3)

        inOrOutButton = tk.Checkbutton(self, height=int(self.canvas_height/600), width=int(self.canvas_width/70), text="Inscribe the polygon or Not", variable=self.inOrOut, bg=BGColors.BG_COLOR.value)
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

