from triangulation import Triangulation
from pathlib import Path

path = Path("/Users/saarhersonsky/Dropbox/python_work/plannar-domains/regions/Eric/Eric.poly")
            
triangulation = Triangulation.read(path)
triangulation.show_triangulation()