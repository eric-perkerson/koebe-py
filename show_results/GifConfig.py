import tkinter as tk
from BGColors import BGColors

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
        controls = tk.Frame(parent, width=self.canvas_width, height=self.canvas_height, bg=BGColors.BG_COLOR.value)
        controls.columnconfigure(0, weight=1)
        controls.rowconfigure(0, weight=1)
        controls.grid(column=0, row=0)

        instructLabel = tk.Label(controls, height=int(self.canvas_height/540), width=int(self.canvas_width/15), text="Select options, then click start to generate a new sequence of figures, WARNING, it takes about 15-30 seconds per step to generate", bg=BGColors.BG_COLOR.value)
        instructLabel.grid(column=2, row=0, columnspan=3)

        iEdgeLabel = tk.Label(controls, width=int(self.canvas_width/70), height=int(self.canvas_height/600), text="Starting Edge Count", bg=BGColors.BG_COLOR.value)
        iEdgeLabel.grid(column=0, row=1)

        iEdgeEntry = tk.Entry(controls, width=int(self.canvas_width/70), textvariable=self.initEdge, bg=BGColors.BLACK.value)
        iEdgeEntry.grid(column=1, row=1)

        fEdgeLabel = tk.Label(controls, width=int(self.canvas_width/70), height=int(self.canvas_height/600), text="Final Edge Count", bg=BGColors.BG_COLOR.value)
        fEdgeLabel.grid(column=2, row=1)

        fEdgeEntry = tk.Entry(controls, width=int(self.canvas_width/70), textvariable=self.finEdge, bg=BGColors.BLACK.value)
        fEdgeEntry.grid(column=3, row=1)

        outRadiusLabel = tk.Label(controls, width=int(self.canvas_width/70), height=int(self.canvas_height/600), text="Outer Radius", bg=BGColors.BG_COLOR.value)
        outRadiusLabel.grid(column=4, row=1)

        outRadiusLabel = tk.Entry(controls, width=int(self.canvas_width/70), textvariable=self.outRad, bg=BGColors.BLACK.value)
        outRadiusLabel.grid(column=5, row=1)

        initInRadiusLabel = tk.Label(controls, width=int(self.canvas_width/70), height=int(self.canvas_height/600), text="Initial Inner Radius", bg=BGColors.BG_COLOR.value)
        initInRadiusLabel.grid(column=0, row=2)

        initInRadiusEntry = tk.Entry(controls, width=int(self.canvas_width/70), textvariable=self.initInRad, bg=BGColors.BLACK.value)
        initInRadiusEntry.grid(column=1, row=2)

        finInRadiusLabel = tk.Label(controls, width=int(self.canvas_width/70), height=int(self.canvas_height/600), text="Final Inner Radius", bg=BGColors.BG_COLOR.value)
        finInRadiusLabel.grid(column=2, row=2)

        finInRadiusEntry = tk.Entry(controls, width=int(self.canvas_width/70), textvariable=self.finInRad, bg=BGColors.BLACK.value)
        finInRadiusEntry.grid(column=3, row=2)

        stepCountLabel = tk.Label(controls, width=int(self.canvas_width/70), height=int(self.canvas_height/600), text="Number of Steps to shrink Inner Radius", bg=BGColors.BG_COLOR.value)
        stepCountLabel.grid(column=4, row=2)

        stepCountEntry = tk.Entry(controls, width=int(self.canvas_width/70), textvariable=self.stepCount, bg=BGColors.BLACK.value)
        stepCountEntry.grid(column=5, row=2)

        fileRootLabel = tk.Label(controls, width=int(self.canvas_width/70), height=int(self.canvas_height/600), text="File Root", bg=BGColors.BG_COLOR.value)
        fileRootLabel.grid(column=0, row=3)

        fileRootEntry = tk.Entry(controls, width=int(self.canvas_width/70), textvariable=self.fileRoot, bg=BGColors.BLACK.value)
        fileRootEntry.grid(column=1, row=3)

        triCountInitLabel = tk.Label(controls, width=int(self.canvas_width/70), height=int(self.canvas_height/600), text="Triangle Count Initial", bg=BGColors.BG_COLOR.value)
        triCountInitLabel.grid(column=2, row=3)

        triCountInitEntry = tk.Entry(controls, width=int(self.canvas_width/70), textvariable=self.triCountInit, bg=BGColors.BLACK.value)
        triCountInitEntry.grid(column=3, row=3)

        triCountFinLabel = tk.Label(controls, width=int(self.canvas_width/70), height=int(self.canvas_height/600), text="Triangle Count Final", bg=BGColors.BG_COLOR.value)
        triCountFinLabel.grid(column=4, row=3)

        triCountFinEntry = tk.Entry(controls, width=int(self.canvas_width/70), textvariable=self.triCountFinal, bg=BGColors.BLACK.value)
        triCountFinEntry.grid(column=5, row=3)

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