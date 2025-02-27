"""This module lets the user draw a region using a simple GUI."""
from sys import argv
import tkinter as tk
from PIL import Image,ImageTk
from tkinter import messagebox
from pathlib import Path
import numpy as np
import warnings
import unicodedata
import subprocess
import random
import math

from region import Region

EXAMPLE_DIRECTORY = Path('regions/vertex')

# Define colors
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


FG_COLOR = WHITE # No idea actually
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
FILL_COLORS = [WHITE] + (len(BDRY_COLORS) - 1) * [BG_COLOR] # sets the fill color for the shape, The second part is so holes show up, any calls to this with 
# verticies other than the first color fill in the middle with the background color, or add a hole

def is_number(string):
    """Checks a string to see if that string represents a number

    Parameters
    ----------
    s : str
        The string to test

    Returns
    -------
    result : bool
        True if `s` represents a number
    """
    try:
        float(string)
        return True
    except ValueError:
        pass

    try:
        unicodedata.numeric(string)
        return True
    except (TypeError, ValueError):
        pass
    return False


def get_unused_file_name(poly_file, poly_root):
    """Takes the poly_file name and if it already in use, generates an unused file name of the form
    `poly_file` + '_' + `i` where `i` is a number.

    Parameters
    ----------
    poly_file : str
        The poly file name that we want to use

    Returns
    -------
    poly_file
        A new poly file name that is not currently used
    """
    suffix = 1
    path = Path('regions') / poly_root / poly_file
    already_exists = path.with_suffix('.poly').exists() # boolean for if theres already a file with that name
    while path.with_suffix('.poly').exists(): # loops until a new file is created
        parts = path.stem.split('_')
        first_part = '_'.join(parts[:-1])
        last_part = parts[-1]
        if is_number(last_part):
            poly_file = first_part + '_' + str(suffix)
        else:
            poly_file += '_' + str(suffix)
        path = Path('regions') / poly_root / poly_file
        suffix += 1

    if already_exists:
        warnings.warn(f'File name already exists. Changing file name to {poly_file}')
    return poly_file


def draw_region(poly_file, poly_root=None):
    """Launches a GUI used to draw a polygonal region with polygonal holes. Writes the result to
    `poly_file`.

    Parameters
    ----------
    poly_file : str, optional
        The name of the poly file to generate, by default 'test'
    """
    
    #poly_file = get_unused_file_name(poly_file, poly_root)
    if poly_root != None:
        example_directory = ((Path('regions') / poly_root) / poly_file) / poly_file
        # print(example_directory)
        if not ((Path('regions') / poly_root) / poly_file).is_dir():
            ((Path('regions') / poly_root) / poly_file).mkdir(parents=True, exist_ok=True)
        poly_path = example_directory.with_suffix('.poly')
        # print(poly_path)\
    else:
        example_directory = (Path('regions') / poly_file) / poly_file
        # print(example_directory)
        if not (Path('regions') / poly_file).is_dir():
            (Path('regions') / poly_file).mkdir(parents=True, exist_ok=True)
        poly_path = example_directory.with_suffix('.poly')
        # print(poly_path)\
    # The above sets up the file to be written too.

    components = [[]]
    last_deleted = []
    gridLines = []

    gui = tk.Tk() # initialized Tk
    gui['bg'] = BG_COLOR # sets the background color to that grey
    gui.title("Define a Polygonal Region with Polygonal Holes") 
    gui.columnconfigure(0, weight=1)
    gui.rowconfigure(0, weight=1)
    canvas_width = gui.winfo_screenwidth() 
    canvas_height = gui.winfo_screenheight() # this and above set height and width variables that fill the screen
    
    controls = tk.Frame(gui, width=canvas_width, height=canvas_height/5 , relief="ridge", bg=BG_COLOR)
    controls.columnconfigure(0, weight=1)
    controls.rowconfigure(0, weight=1)
    controls.grid(column=0, row=0)

    canvas = tk.Canvas(gui, width=canvas_width, height=4/5 * canvas_height, bg=BG_COLOR) # puts a canvas into gui, having it fill the screen, and have that grey color
    canvas.grid(column=0, row=1)
    
    def undo():
        """Undoes the last action taken by the user, repeatable.
        """
        if (len(components) != 1) or (len(components[len(components) - 1]) != 0): # only undo if there are points on the screen
            #print(len(components))
            if ((len(components[len(components) - 1]) == 0) and (len(components) >= 0)):
                del components[len(components) - 1]
                last_deleted.append([]) # If we delete the last vertex of a color, add a spacer so redo knows to go back to that color
            oTag = "Oval" + str(components[len(components) - 1][len(components[len(components) - 1]) - 1][0]) + str(components[len(components) - 1][len(components[len(components) - 1]) - 1][1])
            # creates a tag for the ovals so undo can remove them
            last_deleted.append([components[len(components) - 1][len(components[len(components) - 1]) - 1][0], components[len(components) - 1][len(components[len(components) - 1]) - 1][1]])
            # append the removed point to the list of removed points
            del components[len(components) - 1][len(components[len(components) - 1]) - 1] 
            # delete the point from the list of components
            canvas.delete(oTag)
            if len(components[len(components) - 1]) >= 3: # If there are more than 2 verticies, start drawing the domain
                tag = "Poly" + str(len(components) - 1) # creates an identifier for this shape
                canvas.delete(tag)  # Deletes the old polygon every time a new vertex is added
                canvas.create_polygon(
                    flatten_list(components[len(components) - 1]),
                    fill=FILL_COLORS[len(components) - 1],
                    tag=tag
                    )
            if len(components[len(components) - 1]) < 3:
                tag = "Poly" + str(len(components) - 1)
                canvas.delete(tag)
                # If you go under 3 points, remove the polygon
    
    def redo():
        """Redoes the last action taken by the user, repeatable.
        """
        if len(last_deleted) != 0: # only redo if there are points that have been deleted
            #print(last_deleted)
            if len(last_deleted[len(last_deleted) - 1]) == 0:
                components.append([])
                del last_deleted[len(last_deleted) - 1]
                # If you enounter a spacer, shift to the next color of vertex
            paint(last_deleted[len(last_deleted) - 1][0], last_deleted[len(last_deleted) - 1][1]) # put the removed point back on the board
            del last_deleted[len(last_deleted) - 1]
            if len(components[len(components) - 1]) >= 3: # If there are more than 2 verticies, start drawing the domain
                tag = "Poly" + str(len(components) - 1) # creates an identifier for this shape
                canvas.delete(tag)  # Deletes the old polygon every time a new vertex is added
                canvas.create_polygon(
                    flatten_list(components[len(components) - 1]),
                    fill=FILL_COLORS[len(components) - 1],
                    tag=tag
                    )
    
    def deleteLines():
        for line in gridLines:
            canvas.delete(line)
    
    def gridSet():
        if (grid.get() == "FreeForm"):
            deleteLines()
        elif (grid.get() == "Square"):
            deleteLines()
            for x in range(30):
                gridLines.append(canvas.create_line(50 * x, 0, 50 * x, 770))
            for y in range(17):
                gridLines.append(canvas.create_line(0, 50 * y, 1466, 50 * y))
        elif (grid.get() == "Triangle"):
            deleteLines()
            for x in range(30):
                gridLines.append(canvas.create_line(50 * x, 0, (50 * x) + 385, 770))
                gridLines.append(canvas.create_line(50 * x, 0, (50 * x) - 385, 770))
            for y in range(17):
                gridLines.append(canvas.create_line(0, 50 * y, 1466, 50 * y))
            for y in range(8):
                gridLines.append(canvas.create_line(0, 100 * y, (770 - 100 * y) / 2, 770))
                gridLines.append(canvas.create_line(1500, 100 * y, 1500 - ((770 - 100 * y) / 2), 770))
        else:
            print("How")
        return

    def concentricPolygonRandom():
        # TODO
        # The circumscribing case, putting the circle in the polygon, is not quite flushed out. It needs 2 changes:
        # One: It seems to be breaking with larger edge counts. I think this has to do with problems it faces with angle values larger
        # than edge count, which I'm going to see if theres a simple solution in a second
        # Two: The outer radius function needs to be completely reworked. Right now it ensures that the outer radius does not 
        # intersect the inner radius, however with the circumscribing case the outer radius needs to make sure it encapsulates whatever weird
        # values it comes up with. I think having a maximum radius for the inner section would work to reduce this difficulty, but it still needs
        # to be edited in this way.
        xValue = int(canvas_width/2)
        yValue = int(3*canvas_height/7)
        print(xValue, yValue)
        angleCoef = (2*np.pi/int(edges.get()))
        valid = False
        while not valid:
            angles = []
            while len(angles) != int(edges.get()):
                theta = random.uniform(0, int(edges.get()))
                angles.append(theta)
            angles.sort()
            flag = True
            for i in range(len(angles)):
                theta = angles[i]
                x = xValue + int(polyRadiusOne.get()) * np.cos(theta * angleCoef)
                y = yValue + int(polyRadiusOne.get()) * np.sin(theta * angleCoef)
                theta = angles[i - 1]
                prevx = xValue + int(polyRadiusOne.get()) * np.cos(theta * angleCoef)
                prevy = yValue + int(polyRadiusOne.get()) * np.sin(theta * angleCoef)
                midx = (x + prevx) / 2
                midy = (y + prevy) / 2
                if math.sqrt((midx - xValue) ** 2 + (midy - yValue) ** 2) < int(polyRadiusTwo.get()):
                    flag = False
                if i == len(angles) - 1:
                    theta = angles[0]
                    nextx = xValue + int(polyRadiusOne.get()) * np.cos(theta * angleCoef)
                    nexty = yValue + int(polyRadiusOne.get()) * np.sin(theta * angleCoef)
                    if math.sqrt((nextx - xValue) ** 2 + (nexty - yValue) ** 2) < int(polyRadiusTwo.get()):
                        flag = False
            totalAngleChange = 0
            max = 0
            print("in outer radius, angles are: ", angles)
            for i in range(len(angles) - 1):
                totalAngleChange += abs(angles[i] - angles[i + 1])
                if abs(angles[i] - angles[i + 1]) > max:
                    max = abs(angles[i] - angles[i + 1])
            if abs((int(edges.get()) - angles[-1]) + angles[0]) > max:
                max = abs((int(edges.get()) - angles[-1]) + angles[0])
            totalAngleChange -= max
            if max > int(edges.get()) / 2:
               flag = False
            valid = flag
        outerVertices = []
        for theta in angles:
            point = [xValue + int(polyRadiusOne.get()) * np.cos(theta * angleCoef),
                yValue + int(polyRadiusOne.get()) * np.sin(theta * angleCoef)]
            outerVertices.append(point)
        angles = []
        for point in outerVertices:
                paint(point[0], point[1])
        components.append([])
        if inOrOut.get():
            for i in range(0, int(edges.get())):
                angles.append(random.uniform(0, int(edges.get())))
            angles.sort()
            for theta in angles:
                paint(xValue + int(polyRadiusTwo.get()) * np.cos(theta * angleCoef),
                    yValue + int(polyRadiusTwo.get()) * np.sin(theta * angleCoef)
                    )
        else:
            radialCoef = (1 / (np.cos(np.pi / int(edges.get())))) * int(polyRadiusTwo.get())
            theta = random.uniform(0, int(edges.get()))
            angles.append(theta)
            radius = []
            radius.append(radialCoef)
            totalMult = 0
            paint(xValue + radialCoef * np.cos(theta * angleCoef),
                   yValue + radialCoef * np.sin(theta * angleCoef))
            for i in range(2, int(edges.get())):
                if totalMult > 1.75 * int(edges.get()):
                    mult = random.uniform(.5, .8) + 1
                else:
                    mult = random.uniform(.5, 1.5) + 1
                totalMult += mult
                print("radius at ", i, ": ", radius[i - 2])
                change = np.arccos(int(polyRadiusTwo.get()) / radius[i - 2]) * (int(edges.get()) / (2 * np.pi))
                difference = mult * change
                tanAngle = theta + change
                theta += difference
                angles.append(theta)
                a = xValue + radius[i - 2] * np.cos((tanAngle - change) * angleCoef)
                # a = 735 + (100 * 1.1547005383792515) * cos((t - pi/6) * 2pi / 6)
                b = yValue + radius[i - 2] * np.sin((tanAngle - change) * angleCoef)
                c = xValue + int(polyRadiusTwo.get()) * np.cos(tanAngle * angleCoef)
                d = yValue + int(polyRadiusTwo.get()) * np.sin(tanAngle * angleCoef)
                m = (d - b) / (c - a)
                print("angle", theta)
                xOne = (m * c - d - xValue * np.tan(theta * angleCoef) + yValue) / (m - np.tan(theta * angleCoef))
                yOne = m * (xOne - c) + d
                print("point", xOne, yOne)
                paint(xOne, yOne)
                rad = np.sqrt((xOne - xValue) ** 2 + (yOne - yValue) ** 2)
                radius.append(rad)
            print(radius)
            print(angles)
            for i in range(len(angles)):
                valid = False
                while not valid:
                    valid = True
                    if angles[i] > int(edges.get()):
                        angles[i] -= int(edges.get())
                        valid = False
            print(angles)
            change = np.arccos(int(polyRadiusTwo.get()) / radius[-1])
            theta = angles[0] - change
            x = [0,0]
            y = [0,0]
            a = xValue + int(polyRadiusTwo.get()) * np.cos(theta * angleCoef)
            b = yValue + int(polyRadiusTwo.get()) * np.sin(theta * angleCoef)
            print("a0,b0", a,b)
            x[0] = xValue + radialCoef * np.cos(angles[0] * angleCoef)
            y[0] = yValue + radialCoef * np.sin(angles[0] * angleCoef)
            print("x0,y0", x[0], y[0])
            backSlope = (b - y[0]) / (a - x[0])
            print("back", backSlope)
            theta = angles[-1] + change
            a = xValue + int(polyRadiusTwo.get()) * np.cos(theta * angleCoef)
            b = yValue + int(polyRadiusTwo.get()) * np.sin(theta * angleCoef)
            print("a1,b1", a,b)
            x[1] = xValue + radius[-1] * np.cos(angles[-1] * angleCoef)
            y[1] = yValue + radius[-1] * np.sin(angles[-1] * angleCoef)
            print("x1,y1", x[1], y[1])
            frontSlope = (b - y[1]) / (a - x[1])
            print("front", frontSlope)
            intersectionPoint = [0,0]
            intersectionPoint[0] = ((frontSlope * x[1]) - (backSlope * x[0]) + y[0] - y[1]) / (frontSlope - backSlope)
            intersectionPoint[1] = frontSlope * (intersectionPoint[0] - x[1]) + y[1]
            print("inter", intersectionPoint)
            paint(intersectionPoint[0], intersectionPoint[1])
            
            # use intersection point to get angle, use as maximum for deciding third to last point
            # Then find the point between the first and third to last point which has midpoints on the circle when connected

            # There is still an issue with ordering, i think for larger values it manages to wrap around itself, and unfortunatly I don't think this is solvable with sort.
        for i in range(400):
            angleCoef = (2*np.pi/400)
            epsilon = 5
            pointX = xValue + int(polyRadiusTwo.get()) * np.cos(i * angleCoef)
            pointY = yValue + int(polyRadiusTwo.get()) * np.sin(i * angleCoef)
            x_1, y_1 = (pointX - epsilon), (pointY - epsilon)
            x_2, y_2 = (pointX + epsilon), (pointY + epsilon) # this and above create edges for the oval that will be set at that x and y
            oTag = "Oval" + str(pointX) + str(pointY)
            #print(oTag)
            oval = canvas.create_oval(
                x_1,
                y_1,
                x_2,
                y_2,
                tags=oTag,
                fill=BDRY_COLORS[len(components) - 2],
                outline=''
            ) # creates a little oval to make the vertex more visible
            canvas.tag_raise(oval) # moves the oval to the top of the display list, I think its unnessecary though
            
    def concentricPolygon():
        xValue = int(canvas_width/2)
        yValue = int(3*canvas_height/7)
        angleCoef = (2*np.pi/int(edges.get()))
        for theta in range(0, int(edges.get())):
            paint(xValue + int(polyRadiusOne.get()) * np.cos(theta * angleCoef),
                yValue + int(polyRadiusOne.get()) * np.sin(theta * angleCoef)
                )
        components.append([])
        if inOrOut.get():
            for theta in range(0, int(edges.get())):
                paint(xValue + int(polyRadiusTwo.get()) * np.cos(theta * angleCoef),
                    yValue + int(polyRadiusTwo.get()) * np.sin(theta * angleCoef)
                    )
        else:
            radialCoef = (1 / (np.cos(np.pi / int(edges.get()))))
            for theta in range(0, int(edges.get())):
                paint(xValue + (int(polyRadiusTwo.get()) * radialCoef) * np.cos(theta * angleCoef),
                    yValue + (int(polyRadiusTwo.get()) * radialCoef) * np.sin(theta * angleCoef)
                    )
        # for i in range(400):
        #     angleCoef = (2*np.pi/400)
        #     epsilon = 5
        #     pointX = xValue + int(polyRadiusTwo.get()) * np.cos(i * angleCoef)
        #     pointY = yValue + int(polyRadiusTwo.get()) * np.sin(i * angleCoef)
        #     x_1, y_1 = (pointX - epsilon), (pointY - epsilon)
        #     x_2, y_2 = (pointX + epsilon), (pointY + epsilon) # this and above create edges for the oval that will be set at that x and y
        #     oTag = "Oval" + str(pointX) + str(pointY)
        #     #print(oTag)
        #     oval = canvas.create_oval(
        #         x_1,
        #         y_1,
        #         x_2,
        #         y_2,
        #         tags=oTag,
        #         fill=BDRY_COLORS[len(components) - 1],
        #         outline=''
        #     ) # creates a little oval to make the vertex more visible
        #     canvas.tag_raise(oval) # moves the oval to the top of the display list, I think its unnessecary though
    
    def ellipse():
        xValue = int(canvas_width/2)
        yValue = int(3*canvas_height/7)
        angleCoef = (2*np.pi/int(edges.get()))
        for theta in range(0, int(edges.get())):
            paint(xValue + int(polyRadiusOne.get()) * np.cos(theta * angleCoef),
                yValue + int(polyRadiusTwo.get()) * np.sin(theta * angleCoef)
                )
        components.append([])
        for theta in range(0, int(edges.get())):
            paint(xValue + ratio.get() * int(polyRadiusOne.get()) * np.cos(theta * angleCoef),
                yValue + ratio.get() * int(polyRadiusTwo.get()) * np.sin(theta * angleCoef)
                )
            
    def polygon():
        # TODO add inner != outer side numbers
        if shape.get() == 'circle':
            if randomSet.get():
                concentricPolygonRandom()
            else:
                concentricPolygon()
        else:
            ellipse()

    global polygonBuilder
    polygonBuilder = tk.Frame(controls, height=int(canvas_height/7), width=int(canvas_width/12))
    polygonBuilder.columnconfigure(0, weight=1)
    polygonBuilder.rowconfigure(0, weight=1)
    polygonBuilder.grid(column=2, row=0)
    grid = tk.StringVar()
    grid.set("FreeForm")
    edges = tk.StringVar()
    polyRadiusOne = tk.StringVar()
    polyRadiusTwo = tk.StringVar()
    shape = tk.StringVar()
    shape.set("circle")
    randomSet = tk.BooleanVar()
    inOrOut = tk.BooleanVar()
    ratio = tk.DoubleVar()
    imgUndo = Image.open("draw_region_assets/UNDO.png")
    resized_imageUndo = imgUndo.resize((int(canvas_width/7), int(canvas_height/7)), Image.Resampling.LANCZOS)
    new_imgUndo = ImageTk.PhotoImage(resized_imageUndo, master=gui)
    imgRedo = Image.open("draw_region_assets/REDO.png")
    resized_imageRedo = imgRedo.resize((int(canvas_width/7), int(canvas_height/7)), Image.Resampling.LANCZOS)
    new_imgRedo = ImageTk.PhotoImage(resized_imageRedo, master=gui)
    imgFill = Image.open("draw_region_assets/FILLER.png")
    resized_imageFill = imgFill.resize((int(canvas_width/7), int(canvas_height/7)), Image.Resampling.LANCZOS)
    new_imgFill = ImageTk.PhotoImage(resized_imageFill, master=gui)
    # resizes the images to fit correctly in the buttons

    def configs():
        
        undo_button = tk.Button(controls, height=int(canvas_height/7), width=int(canvas_width/10), image=new_imgUndo, command=undo)
        undo_button.grid(column=0, row=0)
        
        gridSelect = tk.Frame(controls, height=int(canvas_height/7), width=int(canvas_width/10))
        gridSelect.columnconfigure(0, weight=1)
        gridSelect.rowconfigure(0, weight=1)
        gridSelect.grid(column=1, row=0)
        
        noGrid = tk.Radiobutton(gridSelect, width=int(canvas_width/80), variable=grid, value='FreeForm', text="FreeForm Grid", state="active", command=gridSet)
        noGrid.grid(column=0, row=0)
        
        squareGrid = tk.Radiobutton(gridSelect,  width=int(canvas_width/80), variable=grid, value="Square", text="Square Grid", state="normal", command=gridSet)
        squareGrid.grid(column=0, row=1)
        
        triangleGrid = tk.Radiobutton(gridSelect, width=int(canvas_width/80), variable=grid, value="Triangle", text="Triangle Grid", state="normal", command=gridSet)
        triangleGrid.grid(column=0, row=2)
        # these are the selections for grid

        global polygonBuilder
        polygonBuilder.destroy()
        polygonBuilder = tk.Frame(controls, height=int(canvas_height/7), width=int(canvas_width/12))
        polygonBuilder.columnconfigure(0, weight=1)
        polygonBuilder.rowconfigure(0, weight=1)
        polygonBuilder.grid(column=2, row=0)

        if shape.get() == 'circle':
            edgeLabel = tk.Label(polygonBuilder, width=int(canvas_width/90), text="Number of Edges")
            edgeLabel.grid(column=0, row=0)
            edgeEntry = tk.Entry(polygonBuilder, width=int(canvas_width/100), textvariable=edges)
            edgeEntry.grid(column=1, row=0)
            radiusOneLabel = tk.Label(polygonBuilder, width=int(canvas_width/90), text="Outer Radius")
            radiusOneLabel.grid(column=0, row=1)
            radiusOneEntry = tk.Entry(polygonBuilder, width=int(canvas_width/100), textvariable=polyRadiusOne)
            radiusOneEntry.grid(column=1, row=1)
            radiusTwoLabel = tk.Label(polygonBuilder, width=int(canvas_width/90), text="Inner Radius")
            radiusTwoLabel.grid(column=0, row=2)
            radiusTwoEntry = tk.Entry(polygonBuilder, width=int(canvas_width/100), textvariable=polyRadiusTwo)
            radiusTwoEntry.grid(column=1, row=2)
        else:
            edgeLabel = tk.Label(polygonBuilder, width=int(canvas_width/110), text="Number of Edges")
            edgeLabel.grid(column=0, row=0, columnspan=2)
            edgeEntry = tk.Entry(polygonBuilder, width=int(canvas_width/200), textvariable=edges)
            edgeEntry.grid(column=0, row=1, columnspan=2)
            radiusOneLabel = tk.Label(polygonBuilder, width=int(canvas_width/180), text="X Axis Size")
            radiusOneLabel.grid(column=0, row=2)
            radiusOneEntry = tk.Entry(polygonBuilder, width=int(canvas_width/200), textvariable=polyRadiusOne)
            radiusOneEntry.grid(column=1, row=2)
            radiusTwoLabel = tk.Label(polygonBuilder, width=int(canvas_width/180), text="Y Axis Size")
            radiusTwoLabel.grid(column=2, row=2)
            radiusTwoEntry = tk.Entry(polygonBuilder, width=int(canvas_width/200), textvariable=polyRadiusTwo)
            radiusTwoEntry.grid(column=3, row=2)
            ratioLabel = tk.Label(polygonBuilder, width=int(canvas_width/80), text="Fraction for Inner")
            ratioLabel.grid(column=2, row=0, columnspan=2)
            ratioEntry = tk.Entry(polygonBuilder, width=int(canvas_width/200), textvariable=ratio)
            ratioEntry.grid(column=2, row=1, columnspan=2)

        createPolygon = tk.Button(controls, height=int(canvas_height/56), width=int(canvas_width/80), text="Insert Polygon", command=polygon)
        createPolygon.grid(column=3, row=0)

        shapeSelect = tk.Frame(controls, height=int(canvas_height/10), width=int(canvas_width/10))
        shapeSelect.columnconfigure(0, weight=1)
        shapeSelect.rowconfigure(0, weight=1)
        shapeSelect.grid(column=4, row=0)

        circleShape = tk.Radiobutton(shapeSelect, width=int(canvas_width/80), variable=shape, value='circle', text="Circle", state="normal", command=configs)
        circleShape.grid(column=0, row=0)
        ellipseShape = tk.Radiobutton(shapeSelect, width=int(canvas_width/80), variable=shape, value='ellipse', text="Ellipse", state="normal", command=configs)
        ellipseShape.grid(column=0, row=1)

        checkButtonFrame = tk.Frame(controls, height=int(canvas_height/7), width=int(canvas_width/10))
        checkButtonFrame.columnconfigure(0, weight=1)
        checkButtonFrame.rowconfigure(0, weight=1)
        checkButtonFrame.grid(column=5, row=0)
        randomButton = tk.Checkbutton(checkButtonFrame, width=int(canvas_width/70), variable=randomSet, text='Randomize Vertices')
        randomButton.grid(column=0, row=1)
        inOutButton = tk.Checkbutton(checkButtonFrame, width=int(canvas_width/70), variable=inOrOut, text='Inscribed or Circumscribed')
        inOutButton.grid(column=0, row=0)
        if shape.get() != 'circle':
            randomButton["state"] = tk.DISABLED
            inOutButton["state"] = tk.DISABLED
        else:
            randomButton["state"] = tk.ACTIVE
            inOutButton["state"] = tk.ACTIVE
        
        redo_button = tk.Button(controls, height=int(canvas_height/7), width=int(canvas_width/12), image=new_imgRedo, command=redo)
        redo_button.grid(column=6, row=0)
    
    configs()
    

    def flatten_list(list_of_lists):
        """Takes in a list of lists and combines all its lists into a single list.

        Parameters
        ----------
        list_of_lists : list, required
            The list of lists to be flattened
            
        Returns
        -------
        list
            The flattoned list
        """
        return [item for sublist in list_of_lists for item in sublist] # horrific looking, but combines all items in all lists in the list of lists to a single list

    print
    def paint(x, y):
        """Adds verticies to the domain, and fills in the area between them if theres 3 or more.

        Parameters
        ----------
        x : x, required
            The x value of the vertex
        y : y, required
            The y value of the vertex
        epsilon : number, optional
            The size of the pins representing verticies
        """
        epsilon = 5
        pointX = 0
        pointY = 0
        if (grid.get() == "FreeForm"):
            pointX = x
            pointY = y
        elif (grid.get() == "Square"):
            pointX = 100*round((x/100)*2)/2 # Simple rounding to the nearest multiple of 50
            pointY = 100*round((y/100)*2)/2
        elif (grid.get() == "Triangle"):
            shearedX = x - y / 2 # change of basis
            shearedY = y
            roundedShearX = 100*round((shearedX/100)*2)/2
            roundedShearY = 100*round((shearedY/100)*2)/2 # round in that basis
            pointX = roundedShearX + roundedShearY / 2 # change the basis back
            pointY = roundedShearY
            #print("Triangle")
        else:
            print("what")
        
        x_1, y_1 = (pointX - epsilon), (pointY - epsilon)
        x_2, y_2 = (pointX + epsilon), (pointY + epsilon) # this and above create edges for the oval that will be set at that x and y
        components[len(components) - 1].append([pointX, pointY]) # adds the coordinates to the components list at the current "color" basically
        #print([pointX, pointY])
        oTag = "Oval" + str(pointX) + str(pointY)
        #print(oTag)
        oval = canvas.create_oval(
            x_1,
            y_1,
            x_2,
            y_2,
            tags=oTag,
            fill=BDRY_COLORS[len(components) - 1],
            outline=''
        ) # creates a little oval to make the vertex more visible
        canvas.tag_raise(oval) # moves the oval to the top of the display list, I think its unnessecary though
        if len(components[len(components) - 1]) >= 3: # If there are more than 2 verticies, start drawing the domain
            tag = "Poly" + str(len(components) - 1) # creates an identifier for this shape
            canvas.delete(tag)  # Deletes the old polygon every time a new vertex is added
            canvas.create_polygon(
                flatten_list(components[len(components) - 1]),
                fill=FILL_COLORS[len(components) - 1],
                tag=tag
            ) # adds the polygon, with the flattoned list of verticies, the respective fill color, and the identifier
        last_deleted = []
    
    def paintE(event, epsilon=5): # I seperated paint from the function called when clicking to make paint more abstract in use
        """Adds verticies to the domain, and fills in the area between them if theres 3 or more.

        Parameters
        ----------
        event : event, required
            Event that will call this function
        epsilon : number, optional
            The size of the pins representing verticies
        """
        while (len(last_deleted) != 0): # delete the cashe of removed points whenever a new point is added
            del last_deleted[len(last_deleted) - 1]
        paint(event.x, event.y)
            
    

    def new_component(event):
        """Switches to the next kind of component, changing the color of vertex.
        """
        components.append([])
        paintE(event)

    def on_closing():
        response = messagebox.askyesnocancel( # prompts the user to save
            "Save Before Quitting", "Do you want to save before quitting?"
        )
        if response is None:
            pass
        elif response:
            print('Saving as ' + str(poly_path))
            region = Region.region_from_components(components) # creates a region object from the components the user added, the components being the verticies
            with open(poly_path, 'w', encoding='utf-8') as file:
                region.write(file) # writes the region to the polyfile
            gui.destroy()
        else:
            gui.destroy()

    canvas.bind("<ButtonRelease 1>", paintE)
    canvas.bind("<ButtonRelease 2>", new_component)  # For mac
    canvas.bind("<ButtonRelease 3>", new_component)  # For windows

    message = tk.Label(
        gui,
        bg=BG_COLOR,
        fg=FG_COLOR,
        text=(
            "Click for a new vertex of current boundary component. "
            + "Right click to start a new component."
        )
    )
    message.grid(column=0, row=2)

    gui.protocol("WM_DELETE_WINDOW", on_closing)

    tk.mainloop()

    return poly_file

def draw_region_back(fileName, inSideNum, outSideNum, inRad, outRad, x=None, y=None, randomSet=False, fileRoot = None):
    """Draws and saves a polygon without manual clicking

        Parameters
        ----------
        fileRoot : name of file overall for that shape
        fileName : name of specific file for that specific shape's state
        inSideNum : number of sides for the inner shape
        outSideNum : number of sides for the outer shape
        inRad : radius of the hole/inside radius
        outRad : radius of the annulus/outside radius
        x : x value of where the hole is placed inside the shape
        y : y value of where the hole is placed inside the shape
        randomSet : whether to randomize the vertices on the polygon
        """

    # TODO Add an option for the polygon to be inscribed or to be circumscribed by the annuli radii

    if randomSet:
        valid = False
        while not valid:
            angles = []
            while len(angles) != int(outSideNum):
                theta = random.uniform(0, int(outSideNum))
                angles.append(theta)
            angles.sort()
            flag = True
            for i in range(len(angles)):
                theta = angles[i]
                x = int(int(outRad) * np.cos(theta * (2*np.pi/int(outSideNum))))
                y = int(int(outRad) * np.sin(theta * (2*np.pi/int(outSideNum))))
                theta = angles[i - 1]
                prevx = int(int(outRad) * np.cos(theta * (2*np.pi/int(outSideNum))))
                prevy = int(int(outRad) * np.sin(theta * (2*np.pi/int(outSideNum))))
                midx = (x + prevx) / 2
                midy = (y + prevy) / 2
                if math.sqrt((midx) ** 2 + (midy) ** 2) < int(inRad):
                    flag = False
                if i == len(angles) - 1:
                    theta = angles[0]
                    nextx = int(int(outRad) * np.cos(theta * (2*np.pi/int(outSideNum))))
                    nexty = int(int(outRad) * np.sin(theta * (2*np.pi/int(outSideNum))))
                    if math.sqrt((nextx) ** 2 + (nexty) ** 2) < int(inRad):
                        flag = False
            totalAngleChange = 0
            max = 0
            for i in range(len(angles)):
                totalAngleChange += abs(angles[i - 1] - angles[i])
                if abs(angles[i - 1] - angles[i]) > max:
                    max = abs(angles[i - 1] - angles[i])
            totalAngleChange -= max
            if totalAngleChange < int(outSideNum) / 2:
                flag = False
            valid = flag
        components = [[]]
        for theta in angles:
            components[len(components) - 1].append([int(int(outRad) * np.cos(theta * (2*np.pi/int(outSideNum)))),
                                                    int(int(outRad) * np.sin(theta * (2*np.pi/int(outSideNum))))])
        components.append([])
        angles = []
        for i in range(0, int(inSideNum)):
            angles.append(random.uniform(0, int(inSideNum)))
        angles.sort()
        for theta in angles:
            components[len(components) - 1].append([int(int(inRad) * np.cos(theta * (2*np.pi/int(inSideNum)))),
                                                    int(int(inRad) * np.sin(theta * (2*np.pi/int(inSideNum))))])
    else:
        components = [[]]
        print(outRad)
        for theta in range(0, int(outSideNum)):
            components[len(components) - 1].append([int(int(outRad) * np.cos(theta * (2*np.pi/int(outSideNum)))),
                                                    int(int(outRad) * np.sin(theta * (2*np.pi/int(outSideNum))))])
        components.append([])
        for theta in range(0, int(inSideNum)):
            components[len(components) - 1].append([int(int(inRad) * np.cos(theta * (2*np.pi/int(inSideNum)))),
                                                    int(int(inRad) * np.sin(theta * (2*np.pi/int(inSideNum))))])

    if fileRoot != None:
        print('Saving as ' + fileRoot + '/' + fileName)
        region = Region.region_from_components(components) # creates a region object from the components the user added, the components being the verticies
        example_directory = Path('regions/' + fileRoot) / fileName
    else:
        print('Saving as ' + fileName)
        region = Region.region_from_components(components) # creates a region object from the components the user added, the components being the verticies
        example_directory = Path('regions/') / fileName
    if not example_directory.is_dir():
        example_directory.mkdir(parents=True, exist_ok=True)
    poly_path = (example_directory / fileName).with_suffix('.poly')
    with open(poly_path, 'w', encoding='utf-8') as file:
        region.write(file) # writes the region to the polyfile

if __name__ == "__main__":
    if len(argv) > 1:
        draw_region(argv[1], argv[2])
    else:
        draw_region("torst")
