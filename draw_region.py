"""This module lets the user draw a region using a simple GUI."""
from sys import argv
import tkinter as tk
from PIL import Image,ImageTk
from tkinter import messagebox
from pathlib import Path
import warnings
import unicodedata

from region import Region


EXAMPLE_DIRECTORY = Path('regions')

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


def get_unused_file_name(poly_file):
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
    path = EXAMPLE_DIRECTORY / poly_file
    already_exists = path.with_suffix('.poly').exists() # boolean for if theres already a file with that name
    while path.with_suffix('.poly').exists(): # loops until a new file is created
        parts = path.stem.split('_')
        first_part = '_'.join(parts[:-1])
        last_part = parts[-1]
        if is_number(last_part):
            poly_file = first_part + '_' + str(suffix)
        else:
            poly_file += '_' + str(suffix)
        path = EXAMPLE_DIRECTORY / poly_file
        suffix += 1

    if already_exists:
        warnings.warn(f'File name already exists. Changing file name to {poly_file}')
    return poly_file


def draw_region(poly_file='test'):
    """Launches a GUI used to draw a polygonal region with polygonal holes. Writes the result to
    `poly_file`.

    Parameters
    ----------
    poly_file : str, optional
        The name of the poly file to generate, by default 'test'
    """
    
    poly_file = get_unused_file_name(poly_file)
    example_directory = EXAMPLE_DIRECTORY / poly_file
    if not example_directory.is_dir():
        example_directory.mkdir(parents=True, exist_ok=True)
    poly_path = (example_directory / poly_file).with_suffix('.poly')
    # The above sets up the file to be written too.

    components = [[]]
    last_deleted = []

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
    
    def undo():
        """Undoes the last action taken by the user, repeatable.
        """
        if (len(components) != 1) or (len(components[len(components) - 1]) != 0): # only undo if there are points on the screen
            print(len(components))
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
        return
    
    def redo():
        """Redoes the last action taken by the user, repeatable.
        """
        if len(last_deleted) != 0: # only redo if there are points that have been deleted
            print(last_deleted)
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
        return
    
    imgUndo = Image.open("draw_region_assets/UNDO.png")
    resized_imageUndo = imgUndo.resize((int(canvas_height/7), int(canvas_height/7)), Image.Resampling.LANCZOS)
    new_imgUndo = ImageTk.PhotoImage(resized_imageUndo, master=gui)
    imgRedo = Image.open("draw_region_assets/REDO.png")
    resized_imageRedo = imgRedo.resize((int(canvas_height/7), int(canvas_height/7)), Image.Resampling.LANCZOS)
    new_imgRedo = ImageTk.PhotoImage(resized_imageRedo, master=gui)
    imgFill = Image.open("draw_region_assets/FILLER.png")
    resized_imageFill = imgFill.resize((int(canvas_height/7), int(canvas_height/7)), Image.Resampling.LANCZOS)
    new_imgFill = ImageTk.PhotoImage(resized_imageFill, master=gui)
    # resizes the images to fit correctly in the buttons
    
    undo_button = tk.Button(controls, height=int(canvas_height/7), width=int(canvas_height/7), image=new_imgUndo, command=undo)
    undo_button.grid(column=0, row=0)
    
    button1 = tk.Button(controls, height=int(canvas_height/7), width=int(canvas_height/7), image=new_imgFill)
    button1.grid(column=1, row=0)
    
    button2 = tk.Button(controls, height=int(canvas_height/7), width=int(canvas_height/7), image=new_imgFill)
    button2.grid(column=2, row=0)
    
    button3 = tk.Button(controls, height=int(canvas_height/7), width=int(canvas_height/7), image=new_imgFill)
    button3.grid(column=3, row=0)
    
    button4 = tk.Button(controls, height=int(canvas_height/7), width=int(canvas_height/7), image=new_imgFill)
    button4.grid(column=4, row=0)
    
    button5 = tk.Button(controls, height=int(canvas_height/7), width=int(canvas_height/7), image=new_imgFill)
    button5.grid(column=5, row=0)
    
    redo_button = tk.Button(controls, height=int(canvas_height/7), width=int(canvas_height/7), image=new_imgRedo, command=redo)
    redo_button.grid(column=6, row=0)
    
    # For now I have a bunch of pointless buttons I will implement later
    

    canvas = tk.Canvas(gui, width=canvas_width, height=4/5 * canvas_height, bg=BG_COLOR) # puts a canvas into gui, having it fill the screen, and have that grey color
    canvas.grid(column=0, row=1)
    

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
        x_1, y_1 = (x - epsilon), (y - epsilon)
        x_2, y_2 = (x + epsilon), (y + epsilon) # this and above create edges for the oval that will be set at that x and y
        components[len(components) - 1].append([x, y]) # adds the coordinates to the components list at the current "color" basically
        #print([x, y])
        oTag = "Oval" + str(x) + str(y)
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


if __name__ == "__main__":
    if len(argv) > 1:
        draw_region(argv[1])
    else:
        draw_region()
