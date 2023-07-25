from triangulation import Triangulation

t = Triangulation.read('regions/test3/test3.poly')
t.write('regions/test3/test3.output.poly')

print("finish running")